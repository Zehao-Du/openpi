import multiprocessing
import socket
import time
import urllib.request

import main
import numpy as np
from openpi_client import base_policy
from openpi_client import websocket_client_policy
from scipy.spatial.transform import Rotation

from openpi.serving import websocket_policy_server

_RANDOM_POLICY_SEED = 7
_ACTION_HORIZON = 10


class _RandomPolicy(base_policy.BasePolicy):
    def __init__(self, seed: int) -> None:
        self._rng = np.random.default_rng(seed)

    def infer(self, obs: dict) -> dict:
        assert set(obs) == {
            "observation/image",
            "observation/wrist_image",
            "observation/state",
            "prompt",
        }
        actions = self._rng.uniform(-1.0, 1.0, size=(_ACTION_HORIZON, 7)).astype(np.float32)
        return {"actions": actions}


def _serve_random_policy(port: int) -> None:
    server = websocket_policy_server.WebsocketPolicyServer(
        _RandomPolicy(_RANDOM_POLICY_SEED),
        host="127.0.0.1",
        port=port,
        metadata={"policy_name": "random"},
    )
    server.serve_forever()


def _unused_local_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _wait_until_server_is_ready(port: int, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    health_url = f"http://127.0.0.1:{port}/healthz"
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(health_url, timeout=0.2) as response:
                if response.status == 200:
                    return
        except OSError:
            time.sleep(0.01)
    raise TimeoutError(f"Random policy server did not become ready at {health_url}")


def test_make_policy_request_maps_realman_pika_observation() -> None:
    realman_tcp_pose = np.array([0.42, -0.18, 0.31, 0.1, -0.2, 0.3])
    observation = {
        **{key: float(value) for key, value in zip(main.STATE_ACTION_KEYS[:6], realman_tcp_pose, strict=True)},
        "gripper.pos": 0.04,
        "fisheye": np.zeros((8, 12, 3), dtype=np.uint8),
        "rgb": np.full((8, 12, 3), 255, dtype=np.uint8),
    }

    request = main.make_policy_request(observation, "pick the block", resize_size=4)

    expected_pika_pose = main.realman_tcp_pose_to_pika_gripper_pose(realman_tcp_pose)
    np.testing.assert_allclose(request["observation/state"][:6], expected_pika_pose, rtol=1e-6)
    assert request["observation/state"][6] == np.float32(0.04)
    assert request["observation/image"].shape == (4, 4, 3)
    assert request["observation/wrist_image"].shape == (4, 4, 3)
    assert request["prompt"] == "pick the block"


def test_absolute_chunk_to_local_delta() -> None:
    current_position = np.array([1.0, 2.0, 3.0])
    current_rotation = Rotation.from_euler("z", 90, degrees=True)
    local_position_delta = np.array([0.1, 0.2, 0.3])
    local_rotation_delta = Rotation.from_euler("x", 30, degrees=True)

    state = np.concatenate([current_position, current_rotation.as_rotvec(), [0.04]])
    target_position = current_position + current_rotation.apply(local_position_delta)
    target_rotation = current_rotation * local_rotation_delta
    actions = np.array([np.concatenate([target_position, target_rotation.as_rotvec(), [0.06]])])

    result = main.absolute_chunk_to_local_delta(actions, state)

    np.testing.assert_allclose(result[0, :3], local_position_delta, atol=1e-8)
    np.testing.assert_allclose(result[0, 3:6], local_rotation_delta.as_rotvec(), atol=1e-8)
    assert result[0, 6] == 0.06


def test_action_to_robot_dict_preserves_order() -> None:
    action = np.arange(7, dtype=np.float64)
    result = main.action_to_robot_dict(action)

    assert list(result) == list(main.STATE_ACTION_KEYS)
    np.testing.assert_array_equal(list(result.values()), action)


def test_random_policy_server_round_trip() -> None:
    """Exercise request serialization and inference through the real websocket client/server."""
    port = _unused_local_port()
    server_process = multiprocessing.Process(target=_serve_random_policy, args=(port,), daemon=True)
    server_process.start()
    client = None

    try:
        _wait_until_server_is_ready(port)
        client = websocket_client_policy.WebsocketClientPolicy("127.0.0.1", port)
        assert client.get_server_metadata() == {"policy_name": "random"}

        observation = {
            **{key: float(index) for index, key in enumerate(main.STATE_ACTION_KEYS)},
            "fisheye": np.zeros((8, 12, 3), dtype=np.uint8),
            "rgb": np.full((8, 12, 3), 255, dtype=np.uint8),
        }
        request = main.make_policy_request(observation, "test random policy", resize_size=4)
        result = client.infer(request)

        actions = result["actions"]
        expected_actions = np.random.default_rng(_RANDOM_POLICY_SEED).uniform(-1.0, 1.0, size=(_ACTION_HORIZON, 7))
        assert actions.shape == (_ACTION_HORIZON, 7)
        assert np.isfinite(actions).all()
        np.testing.assert_allclose(actions, expected_actions, rtol=1e-6)
        assert result["server_timing"]["infer_ms"] >= 0
    finally:
        if client is not None:
            client._ws.close()  # noqa: SLF001
        server_process.terminate()
        server_process.join(timeout=5)
        server_process.close()

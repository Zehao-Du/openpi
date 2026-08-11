import numpy as np
from scipy.spatial.transform import Rotation

import openpi.policies.realman_pika_policy as _policy


def test_tcp_delta_actions_round_trip() -> None:
    current_position = np.array([1.0, 2.0, 3.0])
    current_rotvec = np.array([0.0, 0.0, np.pi / 2.0])
    local_position_delta = np.array([0.1, 0.2, 0.3])
    local_rotation_delta = np.array([np.pi / 2.0, 0.0, 0.0])

    current_rotation = Rotation.from_rotvec(current_rotvec)
    target_position = current_position + current_rotation.apply(local_position_delta)
    target_rotvec = (current_rotation * Rotation.from_rotvec(local_rotation_delta)).as_rotvec()

    state = np.concatenate([current_position, current_rotvec, [0.25]])
    actions = np.zeros((2, 32), dtype=np.float64)
    actions[0, :7] = state
    actions[1, :7] = np.concatenate([target_position, target_rotvec, [0.75]])
    actions[:, 7:] = 42.0
    absolute_actions = actions.copy()

    data = _policy.DeltaTcpActions()({"state": np.pad(state, (0, 25)), "actions": actions})
    np.testing.assert_allclose(data["actions"][0, :6], 0.0, atol=1e-7)
    np.testing.assert_allclose(data["actions"][1, :3], local_position_delta, atol=1e-7)
    np.testing.assert_allclose(data["actions"][1, 3:6], local_rotation_delta, atol=1e-7)
    np.testing.assert_allclose(data["actions"][:, 6], [0.25, 0.75], atol=1e-7)
    np.testing.assert_allclose(data["actions"][:, 7:], 42.0, atol=1e-7)

    restored = _policy.AbsoluteTcpActions()(data)
    np.testing.assert_allclose(restored["actions"], absolute_actions, atol=1e-7)


def test_tcp_delta_actions_noop_without_actions() -> None:
    data = {"state": np.zeros(7)}
    assert _policy.DeltaTcpActions()(data) is data
    assert _policy.AbsoluteTcpActions()(data) is data


def test_realman_pika_outputs_keep_seven_action_dimensions() -> None:
    actions = np.zeros((10, 32), dtype=np.float32)
    output = _policy.RealmanPikaOutputs()({"actions": actions})
    assert output["actions"].shape == (10, 7)

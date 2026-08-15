"""Run an OpenPI policy on a RealMan arm with a Pika gripper.

The hardware implementation lives in the ``myfork/main`` branch of the sibling
LeRobot repository. By default this client performs one policy query, prints the
predicted action chunk, and exits without moving the robot. Pass ``--execute``
to enable action execution.
"""

from __future__ import annotations

import dataclasses
import logging
import pathlib
import time
from typing import Any, Literal

from lerobot.robots.realman_pika.transforms import realman_tcp_pose_to_pika_gripper_pose
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy
from scipy.spatial.transform import Rotation
import tyro

STATE_ACTION_KEYS = (
    "eef_x.pos",
    "eef_y.pos",
    "eef_z.pos",
    "eef_rx.pos",
    "eef_ry.pos",
    "eef_rz.pos",
    "gripper.pos",
)


@dataclasses.dataclass
class Args:
    # OpenPI policy server.
    host: str = "127.0.0.1"
    port: int = 8000
    prompt: str = "pick all blocks into the drawer"
    resize_size: int = 224

    # Rollout behavior. Execution is opt-in for hardware safety.
    execute: bool = False
    confirm_each_chunk: bool = True
    replan_steps: int = 5
    control_hz: float = 30.0
    max_steps: int = 300
    server_action_mode: Literal["absolute", "delta"] = "absolute"

    # RealMan + Pika hardware.
    robot_ip: str = "192.168.1.18"
    gripper_serial_port: str = "/dev/ttyUSB60"
    rgb_camera_serial: str = "419122270755"
    fisheye_camera_device: str = "/dev/video60"
    calibration_dir: pathlib.Path = dataclasses.field(
        default_factory=lambda: pathlib.Path("~/.cache/lerobot/realman_pika").expanduser()
    )

    # Conservative controller and safety settings.
    max_pos_speed: float = 0.01
    max_rot_speed: float = 0.10
    max_relative_pos: float = 0.03
    max_relative_rot: float = 0.15
    table_collision_enabled: bool = True
    table_height_m: float = 0.23
    robot_action_latency: float = 0.03
    gripper_action_latency: float = 0.0


def _realman_tcp_pose_from_observation(observation: dict[str, Any]) -> np.ndarray:
    missing = [key for key in STATE_ACTION_KEYS[:6] if key not in observation]
    if missing:
        raise KeyError(f"RealmanPika observation is missing TCP pose keys: {missing}")
    tcp_pose = np.asarray([observation[key] for key in STATE_ACTION_KEYS[:6]], dtype=np.float64)
    if tcp_pose.shape != (6,) or not np.isfinite(tcp_pose).all():
        raise ValueError(f"Expected a finite 6D RealMan TCP pose, got {tcp_pose}")
    return tcp_pose


def _state_from_observation(observation: dict[str, Any]) -> np.ndarray:
    if STATE_ACTION_KEYS[6] not in observation:
        raise KeyError(f"RealmanPika observation is missing state key: {STATE_ACTION_KEYS[6]}")
    realman_tcp_pose = _realman_tcp_pose_from_observation(observation)
    pika_pose = realman_tcp_pose_to_pika_gripper_pose(realman_tcp_pose)
    state = np.concatenate([pika_pose, np.asarray([observation[STATE_ACTION_KEYS[6]]])]).astype(np.float32)
    if state.shape != (7,) or not np.isfinite(state).all():
        raise ValueError(f"Expected a finite 7D Pika state, got {state}")
    return state


def _prepare_image(image: Any, resize_size: int) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"Expected an HWC RGB image, got shape {image.shape}")
    return image_tools.convert_to_uint8(image_tools.resize_with_pad(image, resize_size, resize_size))


def make_policy_request(observation: dict[str, Any], prompt: str, resize_size: int) -> dict[str, Any]:
    # input realman tcp pose
    missing_images = [key for key in ("fisheye", "rgb") if key not in observation]
    if missing_images:
        raise KeyError(f"RealmanPika observation is missing camera keys: {missing_images}")

    # Match the conversion script: Pika fisheye is the primary/global view and
    # RealSense RGB is the narrower wrist view.
    return {
        "observation/image": _prepare_image(observation["fisheye"], resize_size),
        "observation/wrist_image": _prepare_image(observation["rgb"], resize_size),
        "observation/state": _state_from_observation(observation),  # convert realman tcp pose to pika pose
        "prompt": prompt,
    }


def _validate_action_chunk(actions: Any) -> np.ndarray:
    actions = np.asarray(actions, dtype=np.float64)
    if actions.ndim != 2 or actions.shape[1] != 7:
        raise ValueError(f"Expected policy actions with shape (horizon, 7), got {actions.shape}")
    if len(actions) == 0 or not np.isfinite(actions).all():
        raise ValueError("Policy returned an empty action chunk or non-finite values")
    return actions


def absolute_chunk_to_local_delta(actions: np.ndarray, state: np.ndarray) -> np.ndarray:
    """Convert absolute Pika poses into local deltas relative to one state snapshot."""
    actions = _validate_action_chunk(actions).copy()
    state = np.asarray(state, dtype=np.float64)
    if state.shape != (7,) or not np.isfinite(state).all():
        raise ValueError(f"Expected a finite state with shape (7,), got {state.shape}")

    current_rotation = Rotation.from_rotvec(state[3:6])
    actions[:, :3] = current_rotation.inv().apply(actions[:, :3] - state[:3])
    actions[:, 3:6] = (current_rotation.inv() * Rotation.from_rotvec(actions[:, 3:6])).as_rotvec()
    # Gripper width is absolute in both the policy dataset and the robot API.
    return actions


def action_to_robot_dict(action: np.ndarray) -> dict[str, float]:
    action = np.asarray(action, dtype=np.float64)
    if action.shape != (7,) or not np.isfinite(action).all():
        raise ValueError(f"Expected a finite action with shape (7,), got {action}")
    return {key: float(value) for key, value in zip(STATE_ACTION_KEYS, action, strict=True)}


def _create_robot(args: Args):
    try:
        from lerobot.cameras import Cv2Backends
        from lerobot.cameras.opencv import OpenCVCameraConfig
        from lerobot.cameras.realsense import RealSenseCameraConfig
        from lerobot.robots.realman_pika import RealmanPika
        from lerobot.robots.realman_pika import RealmanPikaConfig
    except ImportError as error:
        raise ImportError(
            "RealMan-Pika support was not found. Install the sibling LeRobot repository from "
            "its myfork/main branch in the client environment."
        ) from error

    cameras = {
        "rgb": RealSenseCameraConfig(
            serial_number_or_name=args.rgb_camera_serial,
            width=640,
            height=480,
            fps=30,
        ),
        "fisheye": OpenCVCameraConfig(
            index_or_path=pathlib.Path(args.fisheye_camera_device),
            width=640,
            height=480,
            fps=30,
            fourcc="MJPG",
            backend=Cv2Backends.V4L2,
        ),
    }
    config = RealmanPikaConfig(
        id="openpi",
        calibration_dir=args.calibration_dir,
        robot_ip=args.robot_ip,
        gripper_serial_port=args.gripper_serial_port,
        cameras=cameras,
        max_pos_speed=args.max_pos_speed,
        max_rot_speed=args.max_rot_speed,
        max_relative_pos=args.max_relative_pos,
        max_relative_rot=args.max_relative_rot,
        table_collision_enabled=args.table_collision_enabled,
        table_height_m=args.table_height_m,
        robot_action_latency=args.robot_action_latency,
        gripper_action_latency=args.gripper_action_latency,
    )
    return RealmanPika(config)


def _confirm_chunk() -> bool:
    return input("Execute this action chunk? [y/N]: ").strip().lower() in {"y", "yes"}


def run(args: Args) -> None:
    client = websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    logging.info("Policy server metadata: %s", client.get_server_metadata())
    robot = _create_robot(args)
    executed_steps = 0

    try:
        robot.connect()
        logging.info("RealMan-Pika connected. execute=%s", args.execute)

        while executed_steps < args.max_steps:
            # get obesrvation
            observation = robot.get_observation()
            realman_tcp_pose = _realman_tcp_pose_from_observation(observation)
            # repack into policy input
            request = make_policy_request(observation, args.prompt, args.resize_size)
            pika_state = request["observation/state"]
            action_chunk = _validate_action_chunk(client.infer(request)["actions"])

            if args.server_action_mode == "absolute":
                robot_action_chunk = absolute_chunk_to_local_delta(action_chunk, pika_state)
            else:
                robot_action_chunk = action_chunk

            steps_this_chunk = min(args.replan_steps, len(robot_action_chunk), args.max_steps - executed_steps)
            print(f"Predicted action chunk ({len(robot_action_chunk)} steps):\n{action_chunk}")
            print(
                f"Robot-local actions to execute ({steps_this_chunk} steps):\n{robot_action_chunk[:steps_this_chunk]}"
            )

            if not args.execute:  # debug
                logging.info("Query-only mode: no action was sent to the robot.")
                return
            if args.confirm_each_chunk and not _confirm_chunk():
                logging.info("Execution stopped by the operator.")
                return

            # Every action in the predicted chunk is relative to the state sent
            # to the policy, so freeze exactly that hardware pose as the origin.
            robot.set_action_reference_from_realman_tcp_pose(realman_tcp_pose)
            try:
                for action in robot_action_chunk[:steps_this_chunk]:
                    robot.send_action(action_to_robot_dict(action))
                    executed_steps += 1
                    time.sleep(1.0 / args.control_hz)
            finally:
                robot.clear_action_reference()

    except KeyboardInterrupt:
        logging.info("Interrupted by operator.")
    finally:
        if robot.is_connected:
            robot.disconnect()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    run(tyro.cli(Args))

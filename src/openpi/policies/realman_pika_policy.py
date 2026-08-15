import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_realmanpika_example() -> dict:
    """Creates a random input example for the Realman Pika policy."""
    return {
        "observation/state": np.random.rand(7),
        "observation/image": np.random.randint(256, size=(480, 640, 3), dtype=np.uint8),
        "observation/wrist_image": np.random.randint(256, size=(480, 640, 3), dtype=np.uint8),
        "prompt": "do something",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


def _rotvec_to_quaternion(rotvec: np.ndarray) -> np.ndarray:
    """Convert rotation vectors to unit quaternions in xyzw order."""
    rotvec = np.asarray(rotvec, dtype=np.float64)
    angle = np.linalg.norm(rotvec, axis=-1, keepdims=True)
    half_angle = 0.5 * angle
    small = angle < 1e-8
    scale = np.empty_like(angle)
    scale[small] = 0.5 - angle[small] ** 2 / 48.0
    scale[~small] = np.sin(half_angle[~small]) / angle[~small]
    return np.concatenate([rotvec * scale, np.cos(half_angle)], axis=-1)


def _quaternion_to_rotvec(quaternion: np.ndarray) -> np.ndarray:
    """Convert xyzw quaternions to shortest-path rotation vectors."""
    quaternion = np.asarray(quaternion, dtype=np.float64)
    quaternion = quaternion / np.linalg.norm(quaternion, axis=-1, keepdims=True)
    quaternion = np.where(quaternion[..., 3:4] < 0.0, -quaternion, quaternion)
    xyz = quaternion[..., :3]
    xyz_norm = np.linalg.norm(xyz, axis=-1, keepdims=True)
    angle = 2.0 * np.arctan2(xyz_norm, quaternion[..., 3:4])
    small = xyz_norm < 1e-8
    scale = np.empty_like(xyz_norm)
    scale[small] = 2.0
    scale[~small] = angle[~small] / xyz_norm[~small]
    return xyz * scale


def _quaternion_conjugate(quaternion: np.ndarray) -> np.ndarray:
    return np.concatenate([-quaternion[..., :3], quaternion[..., 3:4]], axis=-1)


def _quaternion_multiply(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Compose xyzw quaternions so the result applies rhs, then lhs."""
    lhs_xyz, lhs_w = lhs[..., :3], lhs[..., 3:4]
    rhs_xyz, rhs_w = rhs[..., :3], rhs[..., 3:4]
    xyz = lhs_w * rhs_xyz + rhs_w * lhs_xyz + np.cross(lhs_xyz, rhs_xyz)
    w = lhs_w * rhs_w - np.sum(lhs_xyz * rhs_xyz, axis=-1, keepdims=True)
    return np.concatenate([xyz, w], axis=-1)


def _rotate_vector(quaternion: np.ndarray, vector: np.ndarray) -> np.ndarray:
    """Rotate vectors with unit xyzw quaternions."""
    quaternion_xyz = quaternion[..., :3]
    twice_cross = 2.0 * np.cross(quaternion_xyz, vector)
    return vector + quaternion[..., 3:4] * twice_cross + np.cross(quaternion_xyz, twice_cross)


def _validate_tcp_shapes(state: np.ndarray, actions: np.ndarray) -> None:
    if state.shape[-1] < 7 or actions.shape[-1] < 7:
        raise ValueError(f"TCP state/action must have at least 7 dimensions, got {state.shape} and {actions.shape}")
    if actions.ndim != state.ndim + 1 or actions.shape[:-2] != state.shape[:-1]:
        raise ValueError(f"Expected actions shape (..., horizon, dim) for state (..., dim), got {actions.shape} and {state.shape}")


@dataclasses.dataclass(frozen=True)
class DeltaTcpActions(transforms.DataTransformFn):
    """Convert absolute TCP targets to current end-effector-frame deltas.

    State/action layout is ``[position(3), rotation_vector(3), gripper(1)]``.
    Position and orientation are transformed with SE(3); gripper stays absolute.
    """

    def __call__(self, data: dict) -> dict:
        if "actions" not in data:
            return data

        state = np.asarray(data["state"])
        actions = np.asarray(data["actions"]).copy()
        _validate_tcp_shapes(state, actions)

        current_position = state[..., :3]
        current_quaternion = _rotvec_to_quaternion(state[..., 3:6])
        current_quaternion_inv = _quaternion_conjugate(current_quaternion)
        target_quaternion = _rotvec_to_quaternion(actions[..., 3:6])

        world_position_delta = actions[..., :3] - current_position[..., np.newaxis, :]
        actions[..., :3] = _rotate_vector(current_quaternion_inv[..., np.newaxis, :], world_position_delta)
        relative_quaternion = _quaternion_multiply(
            current_quaternion_inv[..., np.newaxis, :], target_quaternion
        )
        actions[..., 3:6] = _quaternion_to_rotvec(relative_quaternion)
        data["actions"] = actions
        return data


@dataclasses.dataclass(frozen=True)
class AbsoluteTcpActions(transforms.DataTransformFn):
    """Convert current end-effector-frame TCP deltas back to absolute targets."""

    def __call__(self, data: dict) -> dict:
        if "actions" not in data:
            return data

        state = np.asarray(data["state"])
        actions = np.asarray(data["actions"]).copy()
        _validate_tcp_shapes(state, actions)

        current_position = state[..., :3]
        current_quaternion = _rotvec_to_quaternion(state[..., 3:6])
        relative_quaternion = _rotvec_to_quaternion(actions[..., 3:6])

        actions[..., :3] = current_position[..., np.newaxis, :] + _rotate_vector(
            current_quaternion[..., np.newaxis, :], actions[..., :3]
        )
        target_quaternion = _quaternion_multiply(
            current_quaternion[..., np.newaxis, :], relative_quaternion
        )
        actions[..., 3:6] = _quaternion_to_rotvec(target_quaternion)
        data["actions"] = actions
        return data


@dataclasses.dataclass(frozen=True)
class RealmanPikaInputs(transforms.DataTransformFn):
    """
    This class is used to convert inputs to the model to the expected format. It is used for both training and inference.

    For your own dataset, you can copy this class and modify the keys based on the comments below to pipe
    the correct elements of your dataset into the model.
    """

    # Determines which model will be used.
    # Do not change this for your own dataset.
    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        # Possibly need to parse images to uint8 (H,W,C) since LeRobot automatically
        # stores as float32 (C,H,W), gets skipped for policy inference.
        # Keep this for your own dataset, but if your dataset stores the images
        # in a different key than "observation/image" or "observation/wrist_image",
        # you should change it below.
        # Pi0 models support three image inputs at the moment: one third-person view,
        # and two wrist views (left and right). If your dataset does not have a particular type
        # of image, e.g. wrist images, you can comment it out here and replace it with zeros like we do for the
        # right wrist image below.
        base_image = _parse_image(data["observation/image"])
        wrist_image = _parse_image(data["observation/wrist_image"])

        # Create inputs dict. Do not change the keys in the dict below.
        inputs = {
            "state": data["observation/state"],
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                # Pad any non-existent images with zero-arrays of the appropriate shape.
                "right_wrist_0_rgb": np.zeros_like(base_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                # We only mask padding images for pi0 model, not pi0-FAST. Do not change this for your own dataset.
                "right_wrist_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_,
            },
        }

        # Pad actions to the model action dimension. Keep this for your own dataset.
        # Actions are only available during training.
        if "actions" in data:
            inputs["actions"] = data["actions"]

        # Pass the prompt (aka language instruction) to the model.
        # Keep this for your own dataset (but modify the key if the instruction is not
        # stored in "prompt"; the output dict always needs to have the key "prompt").
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class RealmanPikaOutputs(transforms.DataTransformFn):
    """
    This class is used to convert outputs from the model back the the dataset specific format. It is
    used for inference only.

    For your own dataset, you can copy this class and modify the action dimension based on the comments below.
    """

    def __call__(self, data: dict) -> dict:
        # Only return the first N actions -- since we padded actions above to fit the model action
        # dimension, we need to now parse out the correct number of actions in the return dict.
        # For Libero, we only return the first 7 actions (since the rest is padding).
        # For your own dataset, replace `7` with the action dimension of your dataset.
        return {"actions": np.asarray(data["actions"][..., :7])}

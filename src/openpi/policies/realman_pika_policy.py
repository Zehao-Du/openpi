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


def _decode_grounding_bitmask(bitmask) -> np.ndarray:
    """Decodes a (..., H, W) uint8 grounding bitmask into (..., H, W, 1) float supervision maps.

    The single channel means "what the currently active visual prompt refers to" (pick frames:
    the highlighted object; place frames: the release marker). Bit layout: bit0 foreground,
    bit1 unlabeled. Decoded values: 1 foreground, 0 background, -1 unlabeled.
    Accepts the raw 2D map or an image-feature layout (CHW / HWC, uint8 or float in [0, 1]).
    """
    bits = np.asarray(bitmask)
    bits = np.rint(255.0 * bits).astype(np.uint8) if np.issubdtype(bits.dtype, np.floating) else bits.astype(np.uint8)
    if bits.ndim == 3 and bits.shape[0] in (1, 2, 3) and bits.shape[-1] not in (1, 2, 3):
        bits = np.moveaxis(bits, 0, -1)  # CHW -> HWC
    if bits.ndim == 3:  # replicated-channel image feature: all channels carry the same bitmask
        bits = bits[..., 0]
    fg = (bits >> 0) & 1
    unlabeled = (bits >> 1) & 1
    return np.where(unlabeled, -1.0, fg).astype(np.float32)[..., None]


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
        raise ValueError(
            f"Expected actions shape (..., horizon, dim) for state (..., dim), got {actions.shape} and {state.shape}"
        )


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
        relative_quaternion = _quaternion_multiply(current_quaternion_inv[..., np.newaxis, :], target_quaternion)
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
        target_quaternion = _quaternion_multiply(current_quaternion[..., np.newaxis, :], relative_quaternion)
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
    include_original_images: bool = False

    def __call__(self, data: dict) -> dict:
        if self.include_original_images:
            camera_sources = {
                "base_0_rgb": "observation/image_original",
                "left_wrist_0_rgb": "observation/wrist_image_original",
                "base_0_rgb_recolor": "observation/image",
                "left_wrist_0_rgb_recolor": "observation/wrist_image",
            }
            missing = set(camera_sources.values()) - data.keys()
            if missing:
                raise ValueError(f"Four-camera inputs missing keys: {sorted(missing)}")
            images = {key: _parse_image(data[source]) for key, source in camera_sources.items()}
            masks = dict.fromkeys(camera_sources, np.True_)
        else:
            base_image = _parse_image(data["observation/image"])
            images = {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": _parse_image(data["observation/wrist_image"]),
                "right_wrist_0_rgb": np.zeros_like(base_image),
            }
            masks = {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.bool_(self.model_type == _model.ModelType.PI0_FAST),
            }
        inputs = {"state": data["observation/state"], "image": images, "image_mask": masks}

        # B1 grounding auxiliary labels (training data only; absent at inference). The dataset stores
        # a per-frame uint8 bitmask; supervise the original (non-recolor) base camera.
        if "grounding_mask" in data:
            inputs["grounding_masks"] = {"base_0_rgb": _decode_grounding_bitmask(data["grounding_mask"])}
            # Dual-cam VP datasets carry a second bitmask for the aux (fisheye) stream; supervise
            # the wrist slot with it so both visual prompts get grounding supervision.
            if "grounding_mask_fisheye" in data and "left_wrist_0_rgb" in images:
                inputs["grounding_masks"]["left_wrist_0_rgb"] = _decode_grounding_bitmask(
                    data["grounding_mask_fisheye"]
                )

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

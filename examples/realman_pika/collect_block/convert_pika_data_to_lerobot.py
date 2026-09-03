"""
Script for converting a pika collected dataset to LeRobot format.

Usage:
uv run examples/realman_pika/collect_block/convert_pika_data_to_lerobot.py \
    --data-dir /absolute/path/to/collect_blocks_0824

List matched episodes without converting anything:
uv run examples/realman_pika/collect_block/convert_pika_data_to_lerobot.py \
    --data-dir /absolute/path/to/collect_blocks_0824 --test-mode

If you want to push your dataset to the Hugging Face Hub, you can use the following command:
uv run examples/realman_pika/collect_block/convert_pika_data_to_lerobot.py \
    --data-dir /absolute/path/to/collect_blocks_0824 --push-to-hub

The resulting dataset will get saved to the $HF_LEROBOT_HOME directory.
"""

from pathlib import Path
import re
import shutil

import h5py
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
from openpi_client import image_tools
from PIL import Image
from scipy.spatial.transform import Rotation
from tqdm.auto import tqdm
import tyro

REPO_NAME = "Zehao123/pika_collect_blocks_0824_224_224"
TASK_PROMPT = "pick all blocks into the drawer"
FPS = 30
IMAGE_SIZE = 224
DEFAULT_DATA_DIR = Path(
    "/inspire/hdd/project/robot-dna/baojiachun-CZXS25130063/zehao/dataset/pika/collect_blocks_0824"
)

TCP_KEY = "localization/pose/pika"
GRIPPER_KEY = "gripper/encoderDistance/pika"
FISHEYE_KEY = "camera/color/pikaFisheyeCamera"
DEPTH_CAMERA_RGB_KEY = "camera/color/pikaDepthCamera"


def _episode_sort_key(path: Path, data_dir: Path) -> tuple[str, int, int | str]:
    match = re.fullmatch(r"episode(\d+)", path.name)
    parent = path.parent.relative_to(data_dir).as_posix()
    return (parent, 0, int(match.group(1))) if match else (parent, 1, path.name)


def _find_episode_dirs(data_dir: Path, *, test_mode: bool = False) -> list[Path]:
    episode_dirs = sorted(
        (
            hdf5_path.parent
            for hdf5_path in data_dir.rglob("data.hdf5")
            if re.fullmatch(r"episode\d+", hdf5_path.parent.name)
        ),
        key=lambda path: _episode_sort_key(path, data_dir),
    )
    if not episode_dirs:
        raise FileNotFoundError(f"No **/episode*/data.hdf5 directories found under {data_dir}")
    if test_mode:
        print(f"Found {len(episode_dirs)} episode directories:")
        for episode_dir in episode_dirs:
            print(episode_dir.relative_to(data_dir))
    return episode_dirs


def _decode_hdf5_path(value: object) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def _read_rgb(episode_dir: Path, value: object) -> np.ndarray:
    image_path = episode_dir / _decode_hdf5_path(value)
    if not image_path.is_file():
        raise FileNotFoundError(image_path)
    with Image.open(image_path) as image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    return image_tools.resize_with_pad(rgb, IMAGE_SIZE, IMAGE_SIZE)


def _read_state(file: h5py.File, *, test_mode: bool = False) -> np.ndarray:
    poses = np.asarray(file[TCP_KEY][:], dtype=np.float64)
    gripper = np.asarray(file[GRIPPER_KEY][:], dtype=np.float64).reshape(-1, 1)
    if poses.ndim != 2 or poses.shape[1] != 6:
        raise ValueError(f"Expected {TCP_KEY} to have shape (T, 6), got {poses.shape}")
    if len(poses) != len(gripper):
        raise ValueError(f"TCP/gripper length mismatch: {len(poses)} != {len(gripper)}")
    if len(poses) == 0:
        raise ValueError("State is empty")

    rotvec = Rotation.from_euler("xyz", poses[:, 3:6]).as_rotvec()
    state = np.concatenate((poses[:, :3], rotvec, gripper), axis=-1).astype(np.float32)
    if not np.isfinite(state).all():
        raise ValueError("State contains NaN or Inf values")
    if test_mode:
        episode_name = Path(file.filename).parent.name
        sample_count = min(5, len(state))
        print(
            f"{episode_name}: state shape={state.shape}, dtype={state.dtype}, "
            f"min={np.array2string(state.min(axis=0), precision=5)}, "
            f"max={np.array2string(state.max(axis=0), precision=5)}\n"
            f"first {sample_count} states:\n{np.array2string(state[:sample_count], precision=5)}"
        )
    return state


def main(data_dir: str = str(DEFAULT_DATA_DIR), *, push_to_hub: bool = False, test_mode: bool = False):
    # resolve data path and episode path
    data_path = Path(data_dir).expanduser()
    if not data_path.is_absolute():
        raise ValueError(f"--data-dir must be an absolute path, got: {data_dir}")
    data_path = data_path.resolve()
    if not data_path.is_dir():
        raise NotADirectoryError(data_path)
    episode_dirs = _find_episode_dirs(data_path, test_mode=test_mode)

    # for debug
    if test_mode:
        print("\nValidating state arrays:")
        for episode_dir in episode_dirs:
            with h5py.File(episode_dir / "data.hdf5", "r") as file:
                _read_state(file, test_mode=True)
        return

    # Ask before deleting an existing dataset in the output directory.
    output_path = HF_LEROBOT_HOME / REPO_NAME
    if output_path.exists():
        try:
            response = input(f"Output directory already exists: {output_path}\nOverwrite it? [y/N]: ")
        except EOFError:
            print("No interactive input is available. Aborting without overwriting the dataset.")
            return
        if response.strip().lower() not in {"y", "yes"}:
            print("Aborted. The existing dataset was not modified.")
            return
        shutil.rmtree(output_path)

    # Create LeRobot dataset, define features to store
    # OpenPi assumes that proprio is stored in `state` and actions in `action`
    # LeRobot assumes that dtype of image data is `image`
    dataset = LeRobotDataset.create(
        repo_id=REPO_NAME,
        robot_type="realman arm with pika gripper",
        fps=FPS,
        features={
            "image": {
                "dtype": "image",  # Fisheye image: primary/global view.
                "shape": (IMAGE_SIZE, IMAGE_SIZE, 3),
                "names": ["height", "width", "channel"],
            },
            "wrist_image": {
                "dtype": "image",  # Depth camera RGB: narrow wrist view.
                "shape": (IMAGE_SIZE, IMAGE_SIZE, 3),
                "names": ["height", "width", "channel"],
            },
            "state": {
                "dtype": "float32",
                "shape": (7,),  # [pos(3), rotvec(3), gripper(1)]
                "names": ["x", "y", "z", "rx", "ry", "rz", "gripper"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (7,),  # [pos(3), rotvec(3), gripper(1)]
                "names": ["x", "y", "z", "rx", "ry", "rz", "gripper"],
            },
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )

    for episode_dir in tqdm(episode_dirs, desc="Converting Pika", unit="episode", dynamic_ncols=True):
        with h5py.File(episode_dir / "data.hdf5", "r") as file:
            required_keys = (TCP_KEY, GRIPPER_KEY, FISHEYE_KEY, DEPTH_CAMERA_RGB_KEY)
            missing_keys = [key for key in required_keys if key not in file]
            if missing_keys:
                raise KeyError(f"{episode_dir}: missing HDF5 keys {missing_keys}")

            state = _read_state(file)
            episode_length = len(state)
            if len(file[FISHEYE_KEY]) != episode_length or len(file[DEPTH_CAMERA_RGB_KEY]) != episode_length:
                raise ValueError(f"{episode_dir}: camera/state length mismatch")

            episode_label = episode_dir.relative_to(data_path).as_posix()
            for frame_index in tqdm(range(episode_length), desc=episode_label, unit="frame", leave=False):
                # As in the previous UMI dataset, the absolute pose trajectory is stored as action.
                # OpenPI samples future entries and converts the first six dimensions to relative actions.
                dataset.add_frame(
                    {
                        "image": _read_rgb(episode_dir, file[FISHEYE_KEY][frame_index]),
                        "wrist_image": _read_rgb(episode_dir, file[DEPTH_CAMERA_RGB_KEY][frame_index]),
                        "state": state[frame_index],
                        "actions": state[frame_index].copy(),
                        "task": TASK_PROMPT,
                    }
                )
        dataset.save_episode()

    dataset.stop_image_writer()
    print(f"Saved {len(episode_dirs)} episodes to {output_path}")

    # Optionally push to the Hugging Face Hub
    if push_to_hub:
        dataset.push_to_hub(
            tags=["pika", "realman", "manipulation"],
            private=False,
            push_videos=True,
            license="apache-2.0",
        )


if __name__ == "__main__":
    tyro.cli(main)

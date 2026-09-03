"""Convert Pika recordings into color-labeled single-grasp LeRobot episodes.

Each source recording may contain one or more completed open -> closed -> open
gripper cycles. Every detected cycle becomes one output episode, and its task is
``pick the {color} block into the drawer``, where ``color`` is classified from
pre-close wrist-camera frames.

Usage:
uv run examples/realman_pika/collect_block/convert_pika_three_grasps_to_lerobot.py \
    --data-dir /absolute/path/to/collect_blocks_0824

Validate splitting and color classification without writing a dataset:
uv run examples/realman_pika/collect_block/convert_pika_three_grasps_to_lerobot.py \
    --data-dir /absolute/path/to/collect_blocks_0824 --test-mode
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
import random
import re
import shutil
from typing import Any

import av
import h5py
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
from openpi_client import image_tools
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
from scipy.spatial.transform import Rotation
from split_pika_data_by_grasp import GraspCycle
from split_pika_data_by_grasp import detect_grasp_cycles
from tqdm.auto import tqdm
import tyro

DEFAULT_DATA_DIR = Path("/inspire/hdd/project/robot-dna/baojiachun-CZXS25130063/zehao/dataset/pika/collect_blocks_0824")
DEFAULT_REPO_ID = "Zehao123/pika_collect_blocks_0824_224_224_three_grasps_color"
TASK_PROMPT_TEMPLATE = "pick the {color} block into the drawer"
FPS = 30
IMAGE_SIZE = 224
DEBUG_PROMPT_BAR_HEIGHT = 48
DEBUG_PROMPT_FONT = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
EXPECTED_GRASPS_PER_RECORDING = 3

TCP_KEY = "localization/pose/pika"
GRIPPER_KEY = "gripper/encoderDistance/pika"
FISHEYE_KEY = "camera/color/pikaFisheyeCamera"
DEPTH_CAMERA_RGB_KEY = "camera/color/pikaDepthCamera"

COLOR_HUE_RANGES: dict[str, tuple[tuple[float, float], ...]] = {
    "red": ((300.0, 360.0), (0.0, 25.0)),
    "green": ((75.0, 165.0),),
    "blue": ((165.0, 300.0),),
}


@dataclasses.dataclass
class SplitConfig:
    open_threshold: float = 0.085
    closed_threshold: float = 0.075
    min_state_frames: int = 3
    post_release_frames: int = 10


@dataclasses.dataclass
class ColorDetectionConfig:
    candidate_colors: tuple[str, ...] = ("red", "green", "blue")
    reference_frame_offsets: tuple[int, ...] = (-20, -15, -10)
    roi_xyxy: tuple[float, float, float, float] = (0.28, 0.48, 0.72, 0.94)
    min_saturation: float = 0.45
    min_value: float = 0.35
    min_colored_fraction: float = 0.01
    min_confidence: float = 0.5


@dataclasses.dataclass(frozen=True)
class EpisodeSlice:
    output_index: int
    source_episode_dir: Path
    cycle: GraspCycle


@dataclasses.dataclass(frozen=True)
class ColorClassification:
    color: str
    confidence: float
    colored_fraction: float
    reference_frames: tuple[int, ...]
    scores: dict[str, float]


@dataclasses.dataclass
class Args:
    data_dir: Path = DEFAULT_DATA_DIR
    repo_id: str = DEFAULT_REPO_ID
    expected_grasps_per_recording: int | None = None
    max_recordings: int | None = None
    push_to_hub: bool = False
    test_mode: bool = False
    debug_video_dir: Path | None = None
    debug_seed: int = 0
    debug_overwrite: bool = False
    split: SplitConfig = dataclasses.field(default_factory=SplitConfig)
    color_detection: ColorDetectionConfig = dataclasses.field(default_factory=ColorDetectionConfig)


def _episode_sort_key(path: Path, data_dir: Path) -> tuple[str, int, int | str]:
    match = re.fullmatch(r"episode(\d+)", path.name)
    parent = path.parent.relative_to(data_dir).as_posix()
    return (parent, 0, int(match.group(1))) if match else (parent, 1, path.name)


def _find_episode_dirs(data_dir: Path) -> list[Path]:
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
    return episode_dirs


def _plan_episode_slices(
    episode_dirs: list[Path], config: SplitConfig, expected_grasps: int | None
) -> list[EpisodeSlice]:
    slices: list[EpisodeSlice] = []
    for episode_dir in episode_dirs:
        with h5py.File(episode_dir / "data.hdf5", "r") as file:
            if GRIPPER_KEY not in file:
                raise KeyError(f"{episode_dir}: missing HDF5 key {GRIPPER_KEY}")
            gripper = np.asarray(file[GRIPPER_KEY][:], dtype=np.float64)
        cycles = detect_grasp_cycles(
            gripper,
            open_threshold=config.open_threshold,
            closed_threshold=config.closed_threshold,
            min_state_frames=config.min_state_frames,
            post_release_frames=config.post_release_frames,
        )
        if not cycles:
            raise ValueError(f"{episode_dir}: found no completed grasps")
        if expected_grasps is not None and len(cycles) != expected_grasps:
            raise ValueError(f"{episode_dir}: expected exactly {expected_grasps} completed grasps, found {len(cycles)}")
        for cycle in cycles:
            slices.append(EpisodeSlice(len(slices), episode_dir, cycle))
    return slices


def _decode_hdf5_path(value: object) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def _read_native_rgb(episode_dir: Path, value: object) -> np.ndarray:
    image_path = episode_dir / _decode_hdf5_path(value)
    if not image_path.is_file():
        raise FileNotFoundError(image_path)
    with Image.open(image_path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8).copy()


def _read_rgb(episode_dir: Path, value: object) -> np.ndarray:
    return image_tools.resize_with_pad(_read_native_rgb(episode_dir, value), IMAGE_SIZE, IMAGE_SIZE)


def _read_state(file: h5py.File, start: int, end: int) -> np.ndarray:
    poses = np.asarray(file[TCP_KEY][start:end], dtype=np.float64)
    gripper = np.asarray(file[GRIPPER_KEY][start:end], dtype=np.float64).reshape(-1, 1)
    if poses.ndim != 2 or poses.shape[1] != 6:
        raise ValueError(f"Expected {TCP_KEY} to have shape (T, 6), got {poses.shape}")
    if len(poses) != len(gripper) or len(poses) == 0:
        raise ValueError(f"Invalid TCP/gripper lengths: {len(poses)} and {len(gripper)}")
    rotvec = Rotation.from_euler("xyz", poses[:, 3:6]).as_rotvec()
    state = np.concatenate((poses[:, :3], rotvec, gripper), axis=-1).astype(np.float32)
    if not np.isfinite(state).all():
        raise ValueError("State contains NaN or Inf values")
    return state


def _validate_color_config(config: ColorDetectionConfig) -> None:
    if not config.candidate_colors:
        raise ValueError("color_detection.candidate_colors must not be empty")
    unknown = sorted(set(config.candidate_colors) - set(COLOR_HUE_RANGES))
    if unknown:
        raise ValueError(f"Unknown candidate colors: {unknown}")
    if not config.reference_frame_offsets:
        raise ValueError("color_detection.reference_frame_offsets must not be empty")
    x_min, y_min, x_max, y_max = config.roi_xyxy
    if not (0 <= x_min < x_max <= 1 and 0 <= y_min < y_max <= 1):
        raise ValueError(f"Invalid normalized ROI: {config.roi_xyxy}")


def _classify_color(
    images: list[np.ndarray], reference_frames: tuple[int, ...], config: ColorDetectionConfig
) -> ColorClassification:
    _validate_color_config(config)
    scores = dict.fromkeys(config.candidate_colors, 0.0)
    colored_pixels = 0
    roi_pixels = 0
    x_min, y_min, x_max, y_max = config.roi_xyxy
    for image in images:
        height, width = image.shape[:2]
        roi = image[
            round(y_min * height) : round(y_max * height),
            round(x_min * width) : round(x_max * width),
        ]
        hsv = np.asarray(Image.fromarray(roi).convert("HSV"), dtype=np.float32)
        hue = hsv[..., 0] * (360.0 / 255.0)
        saturation = hsv[..., 1] / 255.0
        value = hsv[..., 2] / 255.0
        valid = (saturation >= config.min_saturation) & (value >= config.min_value)
        weights = saturation * value
        any_candidate = np.zeros_like(valid)
        for color in config.candidate_colors:
            color_mask = np.zeros_like(valid)
            for hue_min, hue_max in COLOR_HUE_RANGES[color]:
                color_mask |= (hue >= hue_min) & (hue < hue_max)
            color_mask &= valid
            any_candidate |= color_mask
            scores[color] += float(weights[color_mask].sum())
        colored_pixels += int(any_candidate.sum())
        roi_pixels += valid.size

    total_score = sum(scores.values())
    if total_score <= 0:
        raise ValueError("No supported color found in the grasp reference region")
    color = max(scores, key=scores.__getitem__)
    confidence = scores[color] / total_score
    colored_fraction = colored_pixels / roi_pixels
    if colored_fraction < config.min_colored_fraction:
        raise ValueError(f"Colored fraction {colored_fraction:.3%} is below {config.min_colored_fraction:.3%}")
    if confidence < config.min_confidence:
        raise ValueError(f"Ambiguous block color: {color} confidence {confidence:.3f}; scores={scores}")
    return ColorClassification(
        color=color,
        confidence=confidence,
        colored_fraction=colored_fraction,
        reference_frames=reference_frames,
        scores={name: score / total_score for name, score in scores.items()},
    )


def _classify_slice(episode_slice: EpisodeSlice, file: h5py.File, config: ColorDetectionConfig) -> ColorClassification:
    cycle = episode_slice.cycle
    reference_frames = tuple(
        sorted(
            {min(max(cycle.close + offset, cycle.start), cycle.end - 1) for offset in config.reference_frame_offsets}
        )
    )
    images = [
        _read_native_rgb(episode_slice.source_episode_dir, file[DEPTH_CAMERA_RGB_KEY][index])
        for index in reference_frames
    ]
    try:
        return _classify_color(images, reference_frames, config)
    except ValueError as error:
        raise ValueError(f"{episode_slice.source_episode_dir}, grasp {cycle.start}:{cycle.end}: {error}") from error


def _create_dataset(repo_id: str) -> LeRobotDataset:
    return LeRobotDataset.create(
        repo_id=repo_id,
        robot_type="realman arm with pika gripper",
        fps=FPS,
        features={
            "image": {
                "dtype": "image",
                "shape": (IMAGE_SIZE, IMAGE_SIZE, 3),
                "names": ["height", "width", "channel"],
            },
            "wrist_image": {
                "dtype": "image",
                "shape": (IMAGE_SIZE, IMAGE_SIZE, 3),
                "names": ["height", "width", "channel"],
            },
            "state": {
                "dtype": "float32",
                "shape": (7,),
                "names": ["x", "y", "z", "rx", "ry", "rz", "gripper"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (7,),
                "names": ["x", "y", "z", "rx", "ry", "rz", "gripper"],
            },
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )


def _classify_all(slices: list[EpisodeSlice], config: ColorDetectionConfig) -> dict[int, ColorClassification]:
    classifications: dict[int, ColorClassification] = {}
    for episode_slice in tqdm(slices, desc="Classifying colors", unit="grasp"):
        with h5py.File(episode_slice.source_episode_dir / "data.hdf5", "r") as file:
            if DEPTH_CAMERA_RGB_KEY not in file:
                raise KeyError(f"{episode_slice.source_episode_dir}: missing {DEPTH_CAMERA_RGB_KEY}")
            classification = _classify_slice(episode_slice, file, config)
        classifications[episode_slice.output_index] = classification
        tqdm.write(
            f"output episode {episode_slice.output_index}: {classification.color} "
            f"(confidence={classification.confidence:.3f}, "
            f"source={episode_slice.source_episode_dir.name} "
            f"frames={episode_slice.cycle.start}:{episode_slice.cycle.end})"
        )
    return classifications


def _write_manifest(
    output_path: Path,
    slices: list[EpisodeSlice],
    classifications: dict[int, ColorClassification],
) -> None:
    records: list[dict[str, Any]] = []
    for episode_slice in slices:
        classification = classifications[episode_slice.output_index]
        records.append(
            {
                "episode_index": episode_slice.output_index,
                "source_episode": str(episode_slice.source_episode_dir),
                "source_start": episode_slice.cycle.start,
                "source_close": episode_slice.cycle.close,
                "source_release": episode_slice.cycle.release,
                "source_end": episode_slice.cycle.end,
                **dataclasses.asdict(classification),
                "task_prompt": TASK_PROMPT_TEMPLATE.format(color=classification.color),
            }
        )
    with (output_path / "grasp_color_manifest.json").open("w", encoding="utf-8") as file:
        json.dump(records, file, indent=2)
        file.write("\n")


def _make_debug_frame(
    fisheye_image: np.ndarray,
    wrist_image: np.ndarray,
    prompt: str,
) -> np.ndarray:
    """Place both camera views side by side and draw the task prompt."""
    canvas = Image.new("RGB", (IMAGE_SIZE * 2, IMAGE_SIZE + DEBUG_PROMPT_BAR_HEIGHT), "black")
    canvas.paste(Image.fromarray(fisheye_image), (0, 0))
    canvas.paste(Image.fromarray(wrist_image), (IMAGE_SIZE, 0))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, 66, 15), fill="black")
    draw.text((3, 2), "fisheye", fill="white")
    draw.rectangle((IMAGE_SIZE, 0, IMAGE_SIZE + 46, 15), fill="black")
    draw.text((IMAGE_SIZE + 3, 2), "wrist", fill="white")
    text_bbox = draw.textbbox((0, 0), prompt, font=DEBUG_PROMPT_FONT)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    prompt_x = max(4, (IMAGE_SIZE * 2 - text_width) // 2)
    prompt_y = IMAGE_SIZE + (DEBUG_PROMPT_BAR_HEIGHT - text_height) // 2 - text_bbox[1]
    draw.text((prompt_x, prompt_y), prompt, fill="white", font=DEBUG_PROMPT_FONT)
    return np.asarray(canvas, dtype=np.uint8)


class _DebugVideoWriter:
    def __init__(self, path: Path, fps: int) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._container = av.open(str(path), mode="w")
        self._stream = self._container.add_stream("libx264", rate=fps)
        self._stream.width = IMAGE_SIZE * 2
        self._stream.height = IMAGE_SIZE + DEBUG_PROMPT_BAR_HEIGHT
        self._stream.pix_fmt = "yuv420p"

    def add_frame(self, image: np.ndarray) -> None:
        frame = av.VideoFrame.from_ndarray(image, format="rgb24")
        for packet in self._stream.encode(frame):
            self._container.mux(packet)

    def close(self) -> None:
        for packet in self._stream.encode():
            self._container.mux(packet)
        self._container.close()


def _write_debug_videos(
    output_dir: Path,
    selected_dirs: list[Path],
    slices: list[EpisodeSlice],
    classifications: dict[int, ColorClassification],
    *,
    overwrite: bool,
) -> None:
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    recording_indices = {episode_dir: index for index, episode_dir in enumerate(selected_dirs)}
    grasp_indices: dict[Path, int] = dict.fromkeys(selected_dirs, 0)
    items: list[tuple[EpisodeSlice, Path]] = []
    for episode_slice in slices:
        episode_dir = episode_slice.source_episode_dir
        grasp_index = grasp_indices[episode_dir]
        grasp_indices[episode_dir] += 1
        name = f"sample_{recording_indices[episode_dir]:02d}_{episode_dir.name}_grasp_{grasp_index}.mp4"
        items.append((episode_slice, output_dir / name))

    existing = sorted(output_dir.glob("*.mp4"))
    if existing and not overwrite:
        raise FileExistsError(
            f"{len(existing)} MP4 files already exist in {output_dir}; use --debug-overwrite to replace them"
        )
    for path in existing:
        path.unlink()

    for episode_slice, output_path in tqdm(items, desc="Writing debug videos", unit="video"):
        classification = classifications[episode_slice.output_index]
        prompt = TASK_PROMPT_TEMPLATE.format(color=classification.color)
        cycle = episode_slice.cycle
        writer = _DebugVideoWriter(output_path, FPS)
        try:
            with h5py.File(episode_slice.source_episode_dir / "data.hdf5", "r") as file:
                for source_index in range(cycle.start, cycle.end):
                    fisheye = _read_rgb(episode_slice.source_episode_dir, file[FISHEYE_KEY][source_index])
                    wrist = _read_rgb(
                        episode_slice.source_episode_dir,
                        file[DEPTH_CAMERA_RGB_KEY][source_index],
                    )
                    writer.add_frame(_make_debug_frame(fisheye, wrist, prompt))
        finally:
            writer.close()
        tqdm.write(f"Saved {output_path.name}: {prompt}")

    print(f"Saved {len(items)} debug videos to {output_dir}")


def main(args: Args) -> None:
    if args.expected_grasps_per_recording is not None and args.expected_grasps_per_recording < 1:
        raise ValueError("--expected-grasps-per-recording must be at least 1")
    if args.max_recordings is not None and args.max_recordings < 1:
        raise ValueError("--max-recordings must be at least 1")
    if not args.repo_id.strip():
        raise ValueError("--repo-id must not be empty")

    data_path = args.data_dir.expanduser()
    if not data_path.is_absolute():
        raise ValueError(f"--data-dir must be an absolute path, got {args.data_dir}")
    data_path = data_path.resolve()
    if not data_path.is_dir():
        raise NotADirectoryError(data_path)
    episode_dirs = _find_episode_dirs(data_path)
    if args.max_recordings is not None:
        episode_dirs = episode_dirs[: args.max_recordings]
    if args.debug_video_dir is not None:
        eligible_dirs = []
        for episode_dir in episode_dirs:
            try:
                _plan_episode_slices([episode_dir], args.split, EXPECTED_GRASPS_PER_RECORDING)
            except ValueError:
                continue
            eligible_dirs.append(episode_dir)
        if len(eligible_dirs) < 3:
            raise ValueError(
                "Debug video mode needs at least 3 recordings with exactly 3 completed grasps, "
                f"found {len(eligible_dirs)}"
            )
        episode_dirs = random.Random(args.debug_seed).sample(eligible_dirs, k=3)
        print("Randomly selected debug recordings:")
        for episode_dir in episode_dirs:
            print(f"  {episode_dir.relative_to(data_path)}")
        expected_grasps = EXPECTED_GRASPS_PER_RECORDING
    else:
        expected_grasps = args.expected_grasps_per_recording
    slices = _plan_episode_slices(episode_dirs, args.split, expected_grasps)
    classifications = _classify_all(slices, args.color_detection)
    print(f"Planned {len(slices)} single-grasp episodes from {len(episode_dirs)} recordings")
    if args.test_mode:
        return
    if args.debug_video_dir is not None:
        _write_debug_videos(
            args.debug_video_dir,
            episode_dirs,
            slices,
            classifications,
            overwrite=args.debug_overwrite,
        )
        return

    output_path = HF_LEROBOT_HOME / args.repo_id
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

    dataset = _create_dataset(args.repo_id)
    try:
        for episode_slice in tqdm(slices, desc="Converting Pika grasps", unit="grasp"):
            episode_dir = episode_slice.source_episode_dir
            cycle = episode_slice.cycle
            classification = classifications[episode_slice.output_index]
            task_prompt = TASK_PROMPT_TEMPLATE.format(color=classification.color)
            with h5py.File(episode_dir / "data.hdf5", "r") as file:
                required_keys = (TCP_KEY, GRIPPER_KEY, FISHEYE_KEY, DEPTH_CAMERA_RGB_KEY)
                missing = [key for key in required_keys if key not in file]
                if missing:
                    raise KeyError(f"{episode_dir}: missing HDF5 keys {missing}")
                source_length = len(file[TCP_KEY])
                if any(len(file[key]) != source_length for key in required_keys[1:]):
                    raise ValueError(f"{episode_dir}: camera/state length mismatch")
                state = _read_state(file, cycle.start, cycle.end)
                for local_index, source_index in enumerate(
                    tqdm(
                        range(cycle.start, cycle.end),
                        desc=f"output episode {episode_slice.output_index}",
                        unit="frame",
                        leave=False,
                    )
                ):
                    dataset.add_frame(
                        {
                            "image": _read_rgb(episode_dir, file[FISHEYE_KEY][source_index]),
                            "wrist_image": _read_rgb(episode_dir, file[DEPTH_CAMERA_RGB_KEY][source_index]),
                            "state": state[local_index],
                            "actions": state[local_index].copy(),
                            "task": task_prompt,
                        }
                    )
            dataset.save_episode()
    finally:
        dataset.stop_image_writer()

    _write_manifest(output_path, slices, classifications)
    print(f"Saved {len(slices)} single-grasp episodes to {output_path}")
    if args.push_to_hub:
        dataset.push_to_hub(
            tags=["pika", "realman", "manipulation", "single-grasp", "color-prompt"],
            private=False,
            push_videos=True,
            license="apache-2.0",
        )


if __name__ == "__main__":
    main(tyro.cli(Args))

"""Convert raw collect-blocks Pika recordings to visual-prompt LeRobot data.

Each raw recording is split in memory into completed open -> closed -> open
grasp cycles. Pre-close RealSense frames identify the color of the block that
the robot is about to grasp. That color becomes both the episode task text and
its SAM 3 prompt. Both cameras are resize-padded to 224x224, segmented, and
recolored before being written to the final LeRobot dataset. No intermediate
single-grasp dataset is created.

Usage:
uv run --project examples/realman_pika python \
    examples/realman_pika/visualprompt_convert_pika_data_to_lerobot.py \
    --data-dir /absolute/path/to/collect_blocks_0824

Validate episode discovery and state arrays without loading SAM 3:
uv run --project examples/realman_pika python \
    examples/realman_pika/visualprompt_convert_pika_data_to_lerobot.py \
    --data-dir /absolute/path/to/collect_blocks_0824 --test-mode

Classify every episode without loading SAM 3 or writing a dataset:
uv run --project examples/realman_pika python \
    examples/realman_pika/visualprompt_convert_pika_data_to_lerobot.py \
    --data-dir /absolute/path/to/collect_blocks_0824 --classify-only

Customize SAM 3 and the offline batch size:
uv run --project examples/realman_pika python \
    examples/realman_pika/visualprompt_convert_pika_data_to_lerobot.py \
    --data-dir /absolute/path/to/collect_blocks_0824 \
    --sam-batch-size 8 \
    --sam3.checkpoint ../foundation_models/SAM3 \
    --sam3.prompts "{color} block" "{color} cube" \
    --sam3.device cuda

Add ``--push-to-hub`` to upload the completed dataset.
"""

from __future__ import annotations

from collections.abc import Sequence
import colorsys
import dataclasses
import json
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any

import h5py
from image_preprocessing import ImagePreprocessor
from image_preprocessing import Sam3EpisodeTrackerPreprocessor
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.constants import HF_LEROBOT_HOME
import numpy as np
from openpi_client import image_tools
from PIL import Image
from PIL import ImageDraw
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation
from split_pika_data_by_grasp import GraspCycle
from split_pika_data_by_grasp import detect_grasp_cycles
from tqdm.auto import tqdm
import tyro

DEFAULT_DATA_DIR = Path(
    "/inspire/hdd/project/robot-dna/baojiachun-CZXS25130063/zehao/dataset/pika/collect_blocks_0824"
)
DEFAULT_REPO_ID = "Zehao123/pika_collect_blocks_224_224_visualprompt_new_0824"
DEFAULT_SAM3_CHECKPOINT = Path(__file__).resolve().parents[3] / "foundation_models" / "SAM3"
TASK_PROMPT_TEMPLATE = "grasp the {color} block and place it into the drawer"
FPS = 30
IMAGE_SIZE = 224
LEROBOT_OUTPUT_VERSION = "v2.1"
LEROBOT_CHUNKS_SIZE = 1000

TCP_KEY = "localization/pose/pika"
GRIPPER_KEY = "gripper/encoderDistance/pika"
FISHEYE_KEY = "camera/color/pikaFisheyeCamera"
DEPTH_CAMERA_RGB_KEY = "camera/color/pikaDepthCamera"

# Hue ranges are in degrees and were measured from the grasp-approach frames
# in collect_blocks.  Red wraps around zero; pink is deliberately separated
# from the dataset's deeper red blocks at 350 degrees.
COLOR_HUE_RANGES: dict[str, tuple[tuple[float, float], ...]] = {
    "red": ((350.0, 360.0), (0.0, 25.0)),
    "green": ((75.0, 165.0),),
    "blue": ((165.0, 260.0),),
    "pink": ((300.0, 350.0),),
}


@dataclasses.dataclass
class Sam3Config:
    checkpoint: Path = DEFAULT_SAM3_CHECKPOINT
    prompts: tuple[str, ...] = ("{color} block",)
    target_rgb: tuple[int, int, int] = (0, 0, 255)
    device: str = "cuda"
    score_threshold: float = 0.5
    fisheye_score_threshold: float = 0.4
    mask_threshold: float = 0.3
    alpha: float = 0.9
    min_component_area: int = 64
    model_input_size: int = IMAGE_SIZE
    redetect_area_ratio: float = 0.5
    redetect_reference_decay: float = 0.98
    redetect_cooldown_frames: int = 15


@dataclasses.dataclass
class ColorDetectionConfig:
    candidate_colors: tuple[str, ...] = ("red", "green", "blue", "pink")
    reference_frame_offsets: tuple[int, ...] = (-20, -15, -10)
    roi_xyxy: tuple[float, float, float, float] = (0.28, 0.48, 0.72, 0.94)
    min_saturation: float = 0.45
    min_value: float = 0.35
    min_colored_fraction: float = 0.01
    min_confidence: float = 0.5


@dataclasses.dataclass
class SplitConfig:
    open_threshold: float = 0.085
    closed_threshold: float = 0.075
    min_state_frames: int = 3
    post_release_frames: int = 10


@dataclasses.dataclass(frozen=True)
class EpisodeSlice:
    """One output episode backed by a half-open interval in a source episode."""

    output_index: int
    source_episode_dir: Path
    cycle: GraspCycle

    @property
    def output_name(self) -> str:
        return f"episode{self.output_index}"


@dataclasses.dataclass(frozen=True)
class GraspColorClassification:
    color: str
    confidence: float
    colored_fraction: float
    reference_frames: tuple[int, ...]
    scores: dict[str, float]


@dataclasses.dataclass
class Args:
    data_dir: Path = DEFAULT_DATA_DIR
    repo_id: str = DEFAULT_REPO_ID
    sam_batch_size: int = 8
    max_episodes: int | None = None
    preview_video: Path | None = None
    push_to_hub: bool = False
    test_mode: bool = False
    classify_only: bool = False
    sam3: Sam3Config = dataclasses.field(default_factory=Sam3Config)
    color_detection: ColorDetectionConfig = dataclasses.field(default_factory=ColorDetectionConfig)
    split: SplitConfig = dataclasses.field(default_factory=SplitConfig)


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


def _plan_episode_slices(
    episode_dirs: Sequence[Path],
    config: SplitConfig,
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
            raise ValueError(f"{episode_dir}: no completed open -> closed -> open grasp cycle found")
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


def _validate_color_detection_config(config: ColorDetectionConfig) -> None:
    if not config.candidate_colors:
        raise ValueError("color_detection.candidate_colors must not be empty")
    unknown_colors = sorted(set(config.candidate_colors) - set(COLOR_HUE_RANGES))
    if unknown_colors:
        raise ValueError(f"Unknown candidate colors {unknown_colors}; supported colors are {sorted(COLOR_HUE_RANGES)}")
    if not config.reference_frame_offsets:
        raise ValueError("color_detection.reference_frame_offsets must not be empty")
    x_min, y_min, x_max, y_max = config.roi_xyxy
    if not (0.0 <= x_min < x_max <= 1.0 and 0.0 <= y_min < y_max <= 1.0):
        raise ValueError(f"color_detection.roi_xyxy must be normalized x_min,y_min,x_max,y_max, got {config.roi_xyxy}")
    for name, value in (
        ("min_saturation", config.min_saturation),
        ("min_value", config.min_value),
        ("min_colored_fraction", config.min_colored_fraction),
        ("min_confidence", config.min_confidence),
    ):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"color_detection.{name} must be between 0 and 1, got {value}")


def _classify_grasp_color(
    reference_images: Sequence[np.ndarray],
    reference_frames: Sequence[int],
    config: ColorDetectionConfig,
) -> GraspColorClassification:
    """Classify the centered grasp target from pre-close RealSense frames."""
    _validate_color_detection_config(config)
    if not reference_images or len(reference_images) != len(reference_frames):
        raise ValueError("reference_images and reference_frames must have the same non-zero length")

    scores = dict.fromkeys(config.candidate_colors, 0.0)
    colored_pixel_count = 0
    roi_pixel_count = 0
    x_min, y_min, x_max, y_max = config.roi_xyxy
    for image in reference_images:
        image_array = np.asarray(image)
        if image_array.ndim != 3 or image_array.shape[-1] != 3 or image_array.dtype != np.uint8:
            raise ValueError(f"Expected an HWC RGB uint8 reference image, got {image_array.shape}/{image_array.dtype}")
        height, width = image_array.shape[:2]
        roi = image_array[
            round(y_min * height) : round(y_max * height),
            round(x_min * width) : round(x_max * width),
        ]
        if roi.size == 0:
            raise ValueError(f"color_detection.roi_xyxy produced an empty crop for image shape {image_array.shape}")

        hsv = np.asarray(Image.fromarray(roi).convert("HSV"), dtype=np.float32)
        hue = hsv[..., 0] * (360.0 / 255.0)
        saturation = hsv[..., 1] / 255.0
        value = hsv[..., 2] / 255.0
        valid = (saturation >= config.min_saturation) & (value >= config.min_value)
        weights = saturation * value
        roi_pixel_count += valid.size

        any_candidate = np.zeros_like(valid)
        for color in config.candidate_colors:
            color_mask = np.zeros_like(valid)
            for hue_min, hue_max in COLOR_HUE_RANGES[color]:
                color_mask |= (hue >= hue_min) & (hue < hue_max)
            color_mask &= valid
            any_candidate |= color_mask
            scores[color] += float(weights[color_mask].sum())
        colored_pixel_count += int(any_candidate.sum())

    total_score = sum(scores.values())
    if total_score <= 0.0:
        raise ValueError("No supported high-saturation color was found in the grasp reference region")
    color = max(scores, key=scores.__getitem__)
    confidence = scores[color] / total_score
    colored_fraction = colored_pixel_count / roi_pixel_count
    if colored_fraction < config.min_colored_fraction:
        raise ValueError(
            f"Only {colored_fraction:.3%} of grasp-region pixels matched a supported color; "
            f"minimum is {config.min_colored_fraction:.3%}"
        )
    if confidence < config.min_confidence:
        raise ValueError(
            f"Ambiguous grasped-block color: {color} confidence {confidence:.3f} is below "
            f"{config.min_confidence:.3f}; scores={scores}"
        )
    normalized_scores = {name: score / total_score for name, score in scores.items()}
    return GraspColorClassification(
        color=color,
        confidence=confidence,
        colored_fraction=colored_fraction,
        reference_frames=tuple(reference_frames),
        scores=normalized_scores,
    )


def _classify_episode_grasp_color(
    episode_dir: Path,
    file: h5py.File,
    cycle: GraspCycle,
    config: ColorDetectionConfig,
) -> GraspColorClassification:
    episode_length = len(file[DEPTH_CAMERA_RGB_KEY])
    if not (0 <= cycle.start < cycle.close < cycle.release < cycle.end <= episode_length):
        raise ValueError(f"{episode_dir}: invalid grasp cycle {cycle} for length {episode_length}")
    reference_frames = tuple(
        sorted(
            {min(max(cycle.close + offset, cycle.start), cycle.end - 1) for offset in config.reference_frame_offsets}
        )
    )
    reference_images = [
        _read_native_rgb(episode_dir, file[DEPTH_CAMERA_RGB_KEY][frame_index]) for frame_index in reference_frames
    ]
    try:
        return _classify_grasp_color(reference_images, reference_frames, config)
    except ValueError as error:
        raise ValueError(f"{episode_dir}: failed to classify grasped block: {error}") from error


def _read_state(
    file: h5py.File,
    start: int = 0,
    end: int | None = None,
    *,
    test_mode: bool = False,
) -> np.ndarray:
    poses = np.asarray(file[TCP_KEY][start:end], dtype=np.float64)
    gripper = np.asarray(file[GRIPPER_KEY][start:end], dtype=np.float64).reshape(-1, 1)
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


def _resolve_data_path(data_dir: Path) -> Path:
    data_path = data_dir.expanduser()
    if not data_path.is_absolute():
        raise ValueError(f"--data-dir must be an absolute path, got: {data_dir}")
    data_path = data_path.resolve()
    if not data_path.is_dir():
        raise NotADirectoryError(data_path)
    return data_path


def _create_dataset(repo_id: str, root: Path | None = None) -> LeRobotDataset:
    return LeRobotDataset.create(
        repo_id=repo_id,
        root=root,
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


def _read_parquet_directory(path: Path) -> pa.Table:
    files = sorted(path.glob("chunk-*/file-*.parquet"))
    if not files:
        raise FileNotFoundError(f"No chunk-*/file-*.parquet files found under {path}")
    return pa.concat_tables([pq.read_table(file) for file in files], promote_options="default")


def _write_jsonlines(path: Path, records: Sequence[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            json.dump(record, file, separators=(",", ":"))
            file.write("\n")


def _v21_episode_stats(record: dict[str, Any], feature_names: Sequence[str]) -> dict[str, Any]:
    stats: dict[str, dict[str, Any]] = {}
    for feature_name in feature_names:
        feature_stats = {
            statistic: record[f"stats/{feature_name}/{statistic}"]
            for statistic in ("min", "max", "mean", "std", "count")
            if f"stats/{feature_name}/{statistic}" in record
        }
        if feature_stats:
            stats[feature_name] = feature_stats
    return {"episode_index": record["episode_index"], "stats": stats}


def _with_v21_huggingface_metadata(table: pa.Table) -> pa.Table:
    metadata = dict(table.schema.metadata or {})
    serialized_features = metadata.get(b"huggingface")
    if serialized_features is None:
        raise ValueError("Staging Parquet schema has no Hugging Face feature metadata")
    huggingface_metadata = json.loads(serialized_features)

    def replace_list_type(value: Any) -> None:
        if isinstance(value, dict):
            if value.get("_type") == "List":
                value["_type"] = "Sequence"
            for child in value.values():
                replace_list_type(child)
        elif isinstance(value, list):
            for child in value:
                replace_list_type(child)

    replace_list_type(huggingface_metadata)
    huggingface_metadata.pop("fingerprint", None)
    metadata[b"huggingface"] = json.dumps(huggingface_metadata, separators=(",", ":")).encode()
    return table.replace_schema_metadata(metadata)


def _convert_lerobot_v3_to_v21(source_root: Path, destination_root: Path) -> None:
    """Rewrite a LeRobot v3 image dataset using the v2.1 layout used by OpenPI."""
    with (source_root / "meta/info.json").open(encoding="utf-8") as file:
        source_info = json.load(file)
    if source_info.get("codebase_version") != "v3.0":
        raise ValueError(f"Expected a LeRobot v3.0 staging dataset, got {source_info.get('codebase_version')!r}")

    destination_meta = destination_root / "meta"
    destination_data = destination_root / "data"
    destination_meta.mkdir(parents=True)
    destination_data.mkdir()

    tasks = sorted(
        pq.read_table(source_root / "meta/tasks.parquet").to_pylist(),
        key=lambda record: record["task_index"],
    )
    episodes = sorted(
        _read_parquet_directory(source_root / "meta/episodes").to_pylist(),
        key=lambda record: record["episode_index"],
    )
    expected_episode_indices = list(range(len(episodes)))
    actual_episode_indices = [record["episode_index"] for record in episodes]
    if actual_episode_indices != expected_episode_indices:
        raise ValueError(
            f"LeRobot staging episode indices must be contiguous from zero, got {actual_episode_indices[:10]}"
        )

    records_by_data_file: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for record in episodes:
        data_file = (record["data/chunk_index"], record["data/file_index"])
        records_by_data_file.setdefault(data_file, []).append(record)

    for (chunk_index, file_index), file_episodes in sorted(records_by_data_file.items()):
        source_file = source_root / "data" / f"chunk-{chunk_index:03d}" / f"file-{file_index:03d}.parquet"
        source_table = pq.read_table(source_file)
        for record in file_episodes:
            episode_index = record["episode_index"]
            episode_table = source_table.filter(pc.equal(source_table["episode_index"], episode_index))
            if len(episode_table) != record["length"]:
                raise ValueError(f"Episode {episode_index} has {len(episode_table)} rows, expected {record['length']}")
            episode_chunk = episode_index // LEROBOT_CHUNKS_SIZE
            episode_dir = destination_data / f"chunk-{episode_chunk:03d}"
            episode_dir.mkdir(exist_ok=True)
            pq.write_table(
                _with_v21_huggingface_metadata(episode_table),
                episode_dir / f"episode_{episode_index:06d}.parquet",
            )

    features = source_info["features"]
    total_episodes = len(episodes)
    total_frames = sum(record["length"] for record in episodes)
    output_info = {
        "codebase_version": LEROBOT_OUTPUT_VERSION,
        "robot_type": source_info.get("robot_type"),
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "total_tasks": len(tasks),
        "total_videos": 0,
        "total_chunks": (total_episodes + LEROBOT_CHUNKS_SIZE - 1) // LEROBOT_CHUNKS_SIZE,
        "chunks_size": LEROBOT_CHUNKS_SIZE,
        "fps": source_info["fps"],
        "splits": {"train": f"0:{total_episodes}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": features,
    }
    with (destination_meta / "info.json").open("w", encoding="utf-8") as file:
        json.dump(output_info, file, indent=4)
        file.write("\n")
    _write_jsonlines(destination_meta / "tasks.jsonl", tasks)
    _write_jsonlines(
        destination_meta / "episodes.jsonl",
        [
            {
                "episode_index": record["episode_index"],
                "tasks": record["tasks"],
                "length": record["length"],
            }
            for record in episodes
        ],
    )
    _write_jsonlines(
        destination_meta / "episodes_stats.jsonl",
        [_v21_episode_stats(record, tuple(features)) for record in episodes],
    )


def _make_preview_frame(
    originals: dict[str, np.ndarray],
    processed: dict[str, np.ndarray],
    frame_index: int,
    task_prompt: str,
) -> np.ndarray:
    panels = (
        (originals[f"{frame_index}:image"], "fisheye original"),
        (processed[f"{frame_index}:image"], "fisheye SAM3"),
        (originals[f"{frame_index}:wrist_image"], "RealSense original"),
        (processed[f"{frame_index}:wrist_image"], "RealSense SAM3"),
    )
    canvas = np.zeros((IMAGE_SIZE * 2, IMAGE_SIZE * 2, 3), dtype=np.uint8)
    for panel_index, (panel, _label) in enumerate(panels):
        row, column = divmod(panel_index, 2)
        y, x = row * IMAGE_SIZE, column * IMAGE_SIZE
        canvas[y : y + IMAGE_SIZE, x : x + IMAGE_SIZE] = panel

    preview = Image.fromarray(canvas)
    draw = ImageDraw.Draw(preview)
    for panel_index, (_, label) in enumerate(panels):
        row, column = divmod(panel_index, 2)
        y, x = row * IMAGE_SIZE, column * IMAGE_SIZE
        draw.rectangle((x, y, x + 112, y + 15), fill=(0, 0, 0))
        draw.text((x + 3, y + 2), label, fill=(255, 255, 255))
    draw.rectangle((0, IMAGE_SIZE * 2 - 17, IMAGE_SIZE * 2, IMAGE_SIZE * 2), fill=(0, 0, 0))
    draw.text((3, IMAGE_SIZE * 2 - 15), task_prompt, fill=(255, 255, 255))
    return np.asarray(preview, dtype=np.uint8)


class _PreviewVideoWriter:
    def __init__(self, path: Path, fps: int) -> None:
        try:
            import av
        except ImportError as error:
            raise ImportError("Preview video output requires PyAV, which should be installed by LeRobot") from error

        path.parent.mkdir(parents=True, exist_ok=True)
        self._container = av.open(str(path), mode="w")
        self._stream = self._container.add_stream("libx264", rate=fps)
        self._stream.width = IMAGE_SIZE * 2
        self._stream.height = IMAGE_SIZE * 2
        self._stream.pix_fmt = "yuv420p"
        self._av = av

    def add_frame(self, image: np.ndarray) -> None:
        frame = self._av.VideoFrame.from_ndarray(image, format="rgb24")
        for packet in self._stream.encode(frame):
            self._container.mux(packet)

    def close(self) -> None:
        for packet in self._stream.encode():
            self._container.mux(packet)
        self._container.close()


def _preprocess_frame_batch(
    episode_dir: Path,
    file: h5py.File,
    frame_indices: Sequence[int],
    image_preprocessor: ImagePreprocessor,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    images: dict[str, np.ndarray] = {}
    for frame_index in frame_indices:
        images[f"{frame_index}:image"] = _read_rgb(episode_dir, file[FISHEYE_KEY][frame_index])
        images[f"{frame_index}:wrist_image"] = _read_rgb(episode_dir, file[DEPTH_CAMERA_RGB_KEY][frame_index])

    if getattr(image_preprocessor, "requires_sequential_frames", False):
        processed: dict[str, np.ndarray] = {}
        for frame_index in frame_indices:
            frame_images = {
                "image": images[f"{frame_index}:image"],
                "wrist_image": images[f"{frame_index}:wrist_image"],
            }
            frame_processed = image_preprocessor.preprocess(frame_images)
            if set(frame_processed) != set(frame_images):
                raise ValueError(f"SAM 3 changed image keys from {set(frame_images)} to {set(frame_processed)}")
            processed.update({f"{frame_index}:{camera_name}": image for camera_name, image in frame_processed.items()})
    else:
        processed = image_preprocessor.preprocess(images)
    if set(processed) != set(images):
        raise ValueError(f"SAM 3 changed image keys from {set(images)} to {set(processed)}")
    for key, image in processed.items():
        image_array = np.asarray(image)
        if image_array.shape != (IMAGE_SIZE, IMAGE_SIZE, 3) or image_array.dtype != np.uint8:
            raise ValueError(
                f"SAM 3 returned {key!r} with shape/dtype {image_array.shape}/{image_array.dtype}; "
                f"expected {(IMAGE_SIZE, IMAGE_SIZE, 3)}/uint8"
            )
        processed[key] = image_array
    return images, processed


def _grasp_detection_anchor_index(state: np.ndarray, grasp_close_local_index: int) -> int:
    """Select the lowest TCP z frame during the approach-to-grasp phase."""
    if state.ndim != 2 or state.shape[1] < 3 or len(state) == 0:
        raise ValueError(f"Expected a non-empty (T, >=3) state array, got {state.shape}")
    if not 0 <= grasp_close_local_index < len(state):
        raise ValueError(
            f"Grasp close index {grasp_close_local_index} is outside an episode with {len(state)} frames"
        )
    return int(np.argmin(state[: grasp_close_local_index + 1, 2]))


def _detection_candidate_local_indices(episode_length: int, preferred_anchor: int) -> tuple[int, ...]:
    """Order every frame by distance from the preferred SAM 3 detection anchor."""
    if episode_length < 1:
        raise ValueError(f"Episode length must be positive, got {episode_length}")
    if not 0 <= preferred_anchor < episode_length:
        raise ValueError(
            f"Detection anchor {preferred_anchor} is outside an episode with {episode_length} frames"
        )
    return tuple(
        sorted(range(episode_length), key=lambda index: (abs(index - preferred_anchor), index))
    )


def _write_episode_frames(
    dataset: Any,
    episode_dir: Path,
    file: h5py.File,
    state: np.ndarray,
    source_start: int,
    image_preprocessor: ImagePreprocessor,
    sam_batch_size: int,
    task_prompt: str,
    output_name: str,
    preview_writer: _PreviewVideoWriter | None = None,
    detection_anchor_local_index: int = 0,
) -> bool:
    episode_length = len(state)
    if not 0 <= detection_anchor_local_index < episode_length:
        raise ValueError(
            f"Detection anchor {detection_anchor_local_index} is outside an episode with "
            f"{episode_length} frames"
        )

    originals_by_key: dict[str, np.ndarray] = {}
    processed_by_key: dict[str, np.ndarray] = {}

    with tqdm(total=episode_length, desc=output_name, unit="frame", leave=False) as frame_progress:
        def process_sequence(source_indices: Sequence[int], *, store: bool) -> None:
            for batch_start in range(0, len(source_indices), sam_batch_size):
                batch_indices = source_indices[batch_start : batch_start + sam_batch_size]
                originals, processed = _preprocess_frame_batch(
                    episode_dir, file, batch_indices, image_preprocessor
                )
                if store:
                    originals_by_key.update(originals)
                    processed_by_key.update(processed)
                    frame_progress.update(len(batch_indices))

        if getattr(image_preprocessor, "requires_sequential_frames", False):
            start_episode = getattr(image_preprocessor, "start_episode", None)
            if not callable(start_episode):
                raise TypeError("A sequential image preprocessor must provide start_episode()")
            has_active_trackers = getattr(image_preprocessor, "has_active_trackers", None)
            if not callable(has_active_trackers):
                raise TypeError(
                    "A sequential image preprocessor must provide has_active_trackers()"
                )

            source_end = source_start + episode_length
            anchor_source_index: int | None = None
            for candidate_local_index in _detection_candidate_local_indices(
                episode_length, detection_anchor_local_index
            ):
                candidate_source_index = source_start + candidate_local_index
                start_episode()
                candidate_originals, candidate_processed = _preprocess_frame_batch(
                    episode_dir, file, (candidate_source_index,), image_preprocessor
                )
                if has_active_trackers(("image", "wrist_image")):
                    anchor_source_index = candidate_source_index
                    originals_by_key.update(candidate_originals)
                    processed_by_key.update(candidate_processed)
                    frame_progress.update(1)
                    if candidate_local_index != detection_anchor_local_index:
                        tqdm.write(
                            f"{output_name}: SAM 3 initialization recovered on adjacent source "
                            f"frame {candidate_source_index}"
                        )
                    break
                tqdm.write(
                    f"{output_name}: no valid two-camera SAM 3 initialization on source frame "
                    f"{candidate_source_index}; trying the next adjacent frame"
                )

            if anchor_source_index is None:
                tqdm.write(
                    f"{output_name}: skipping episode because SAM 3 could not initialize both "
                    "camera trackers on any frame"
                )
                return False

            # Continue from the successful detection frame toward the episode end.
            process_sequence(tuple(range(anchor_source_index + 1, source_end)), store=True)

            if anchor_source_index > source_start:
                # Reinitialize at the same anchor, then feed earlier frames in reverse
                # chronological order so every output frame receives a tracker mask.
                start_episode()
                process_sequence((anchor_source_index,), store=False)
                if not has_active_trackers(("image", "wrist_image")):
                    tqdm.write(
                        f"{output_name}: skipping episode because SAM 3 could not reinitialize "
                        f"both camera trackers on source frame {anchor_source_index}"
                    )
                    return False
                process_sequence(
                    tuple(range(anchor_source_index - 1, source_start - 1, -1)),
                    store=True,
                )
        else:
            process_sequence(tuple(range(source_start, source_start + episode_length)), store=True)

        for local_index in range(episode_length):
            source_index = source_start + local_index
            dataset.add_frame(
                {
                    "image": processed_by_key[f"{source_index}:image"],
                    "wrist_image": processed_by_key[f"{source_index}:wrist_image"],
                    "state": state[local_index],
                    "actions": state[local_index].copy(),
                    "task": task_prompt,
                }
            )
            if preview_writer is not None:
                preview_writer.add_frame(
                    _make_preview_frame(originals_by_key, processed_by_key, source_index, task_prompt)
                )
    return True


def _format_color_prompts(prompt_templates: tuple[str, ...], color: str) -> tuple[str, ...]:
    if not prompt_templates:
        raise ValueError("sam3.prompts must contain at least one prompt template")
    if any("{color}" not in template for template in prompt_templates):
        raise ValueError(f"Every sam3.prompts template must contain '{{color}}', got {prompt_templates}")
    prompts = tuple(template.format(color=color).strip() for template in prompt_templates)
    if any(not prompt for prompt in prompts):
        raise ValueError(f"sam3.prompts produced an empty prompt for color {color!r}")
    return prompts


def _target_rgb_color_name(target_rgb: tuple[int, int, int]) -> str:
    """Return a stable natural-language color name for the recoloring target."""
    if len(target_rgb) != 3 or any(channel < 0 or channel > 255 for channel in target_rgb):
        raise ValueError(f"sam3.target_rgb must contain three values in [0, 255], got {target_rgb}")

    red, green, blue = (channel / 255.0 for channel in target_rgb)
    hue, saturation, value = colorsys.rgb_to_hsv(red, green, blue)
    if value < 0.15:
        return "black"
    if saturation < 0.15:
        return "white" if value >= 0.85 else "gray"

    hue_degrees = hue * 360.0
    color_sectors = (
        (15.0, "red"),
        (45.0, "orange"),
        (70.0, "yellow"),
        (165.0, "green"),
        (195.0, "cyan"),
        (255.0, "blue"),
        (285.0, "purple"),
        (345.0, "pink"),
        (360.0, "red"),
    )
    return next(name for upper_bound, name in color_sectors if hue_degrees < upper_bound)


def _make_preprocessor(config: Sam3Config, initial_prompts: tuple[str, ...]) -> Sam3EpisodeTrackerPreprocessor:
    return Sam3EpisodeTrackerPreprocessor(
        config.checkpoint,
        prompts=initial_prompts,
        target_rgb=config.target_rgb,
        device=config.device,
        score_threshold=config.score_threshold,
        camera_score_thresholds={"image": config.fisheye_score_threshold},
        mask_threshold=config.mask_threshold,
        alpha=config.alpha,
        min_component_area=config.min_component_area,
        model_input_size=config.model_input_size,
        error_policy="raise",
        redetect_area_ratio=config.redetect_area_ratio,
        redetect_reference_decay=config.redetect_reference_decay,
        redetect_cooldown_frames=config.redetect_cooldown_frames,
    )


def _classify_episodes(
    episode_slices: Sequence[EpisodeSlice],
    config: ColorDetectionConfig,
) -> dict[int, GraspColorClassification]:
    classifications: dict[int, GraspColorClassification] = {}
    for episode_slice in tqdm(episode_slices, desc="Classifying grasp colors", unit="episode", dynamic_ncols=True):
        episode_dir = episode_slice.source_episode_dir
        with h5py.File(episode_dir / "data.hdf5", "r") as file:
            if DEPTH_CAMERA_RGB_KEY not in file:
                raise KeyError(f"{episode_dir}: missing HDF5 key {DEPTH_CAMERA_RGB_KEY}")
            classification = _classify_episode_grasp_color(episode_dir, file, episode_slice.cycle, config)
        classifications[episode_slice.output_index] = classification
        tqdm.write(
            f"{episode_slice.output_name} ({episode_dir.name} "
            f"frames {episode_slice.cycle.start}:{episode_slice.cycle.end}): {classification.color} "
            f"(confidence={classification.confidence:.3f}, "
            f"colored_fraction={classification.colored_fraction:.3f}, "
            f"frames={classification.reference_frames})"
        )
    return classifications


def _make_manifest(
    episode_slices: Sequence[EpisodeSlice],
    classifications: dict[int, GraspColorClassification],
    prompts_by_episode: dict[int, tuple[str, ...]],
    target_rgb: tuple[int, int, int],
    target_color: str,
) -> list[dict[str, Any]]:
    manifest: list[dict[str, Any]] = []
    for episode_slice in episode_slices:
        output_index = episode_slice.output_index
        classification = classifications[output_index]
        manifest.append(
            {
                "episode": episode_slice.output_name,
                "source_episode": episode_slice.source_episode_dir.name,
                "source_start": episode_slice.cycle.start,
                "source_close": episode_slice.cycle.close,
                "source_release": episode_slice.cycle.release,
                "source_end": episode_slice.cycle.end,
                **dataclasses.asdict(classification),
                "sam3_prompts": list(prompts_by_episode[output_index]),
                "recolor_target_rgb": list(target_rgb),
                "recolor_target_color": target_color,
                "task_prompt": TASK_PROMPT_TEMPLATE.format(color=target_color),
            }
        )
    return manifest


def _confirm_preview_video_overwrite(preview_video_path: Path) -> bool:
    if not preview_video_path.is_file():
        raise IsADirectoryError(f"Preview video path exists but is not a file: {preview_video_path}")
    try:
        response = input(
            f"Preview video already exists: {preview_video_path}\n"
            "Overwrite it? [y/N]: "
        )
    except EOFError:
        print("No interactive input is available. Aborting without overwriting the preview video.")
        return False
    if response.strip().lower() not in {"y", "yes"}:
        print("Aborted. The existing preview video was not modified.")
        return False
    return True


def main(args: Args) -> None:
    if args.sam_batch_size < 1:
        raise ValueError(f"--sam-batch-size must be at least 1, got {args.sam_batch_size}")
    if not args.repo_id.strip():
        raise ValueError("--repo-id must not be empty")
    if args.max_episodes is not None and args.max_episodes < 1:
        raise ValueError(f"--max-episodes must be at least 1, got {args.max_episodes}")
    preview_video_path = args.preview_video.expanduser().resolve() if args.preview_video else None
    if (
        preview_video_path is not None
        and preview_video_path.exists()
        and not _confirm_preview_video_overwrite(preview_video_path)
    ):
        return

    data_path = _resolve_data_path(args.data_dir)
    episode_dirs = _find_episode_dirs(data_path, test_mode=args.test_mode)
    episode_slices = _plan_episode_slices(episode_dirs, args.split)
    planned_episode_count = len(episode_slices)
    if args.max_episodes is not None:
        episode_slices = episode_slices[: args.max_episodes]
    print(
        f"Selected {len(episode_slices)} of {planned_episode_count} single-grasp episodes "
        f"from {len(episode_dirs)} recordings"
    )

    if args.test_mode:
        print("\nValidating sliced state arrays:")
        for episode_slice in episode_slices:
            with h5py.File(episode_slice.source_episode_dir / "data.hdf5", "r") as file:
                _read_state(
                    file,
                    episode_slice.cycle.start,
                    episode_slice.cycle.end,
                    test_mode=True,
                )
        return

    classifications = _classify_episodes(episode_slices, args.color_detection)
    prompts_by_episode = {
        episode_slice.output_index: _format_color_prompts(
            args.sam3.prompts, classifications[episode_slice.output_index].color
        )
        for episode_slice in episode_slices
    }
    target_color = _target_rgb_color_name(args.sam3.target_rgb)
    task_prompt = TASK_PROMPT_TEMPLATE.format(color=target_color)
    color_manifest = _make_manifest(
        episode_slices,
        classifications,
        prompts_by_episode,
        args.sam3.target_rgb,
        target_color,
    )
    if args.classify_only:
        print(json.dumps(color_manifest, indent=2))
        return
    manifest_by_output_index = {
        episode_slice.output_index: manifest_entry
        for episode_slice, manifest_entry in zip(episode_slices, color_manifest, strict=True)
    }

    output_path = HF_LEROBOT_HOME / args.repo_id
    overwrite = False
    if output_path.exists():
        try:
            response = input(f"Output directory already exists: {output_path}\nOverwrite it? [y/N]: ")
        except EOFError:
            print("No interactive input is available. Aborting without overwriting the dataset.")
            return
        if response.strip().lower() not in {"y", "yes"}:
            print("Aborted. The existing dataset was not modified.")
            return
        overwrite = True

    # Validate and load SAM 3 before touching any existing output dataset.
    image_preprocessor = _make_preprocessor(args.sam3, prompts_by_episode[0])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output_path.name}-staging-", dir=output_path.parent
    ) as staging_directory:
        staging_root = Path(staging_directory)
        staging_v3 = staging_root / "v3"
        staging_v21 = staging_root / "v2.1"
        dataset = _create_dataset(args.repo_id, root=staging_v3)
        preview_writer = _PreviewVideoWriter(preview_video_path, FPS) if preview_video_path else None
        saved_manifest: list[dict[str, Any]] = []
        skipped_episode_count = 0

        try:
            for episode_slice in tqdm(
                episode_slices, desc="Converting Pika with SAM 3", unit="episode", dynamic_ncols=True
            ):
                episode_dir = episode_slice.source_episode_dir
                image_preprocessor.set_prompts(prompts_by_episode[episode_slice.output_index])
                with h5py.File(episode_dir / "data.hdf5", "r") as file:
                    required_keys = (TCP_KEY, GRIPPER_KEY, FISHEYE_KEY, DEPTH_CAMERA_RGB_KEY)
                    missing_keys = [key for key in required_keys if key not in file]
                    if missing_keys:
                        raise KeyError(f"{episode_dir}: missing HDF5 keys {missing_keys}")

                    source_length = len(file[TCP_KEY])
                    camera_lengths = (
                        len(file[GRIPPER_KEY]),
                        len(file[FISHEYE_KEY]),
                        len(file[DEPTH_CAMERA_RGB_KEY]),
                    )
                    if any(length != source_length for length in camera_lengths):
                        raise ValueError(f"{episode_dir}: camera/state length mismatch")
                    state = _read_state(file, episode_slice.cycle.start, episode_slice.cycle.end)
                    grasp_close_local_index = episode_slice.cycle.close - episode_slice.cycle.start
                    detection_anchor_local_index = _grasp_detection_anchor_index(
                        state, grasp_close_local_index
                    )
                    tqdm.write(
                        f"{episode_slice.output_name}: SAM 3 anchor source frame "
                        f"{episode_slice.cycle.start + detection_anchor_local_index} "
                        f"(TCP z={state[detection_anchor_local_index, 2]:.5f})"
                    )

                    episode_written = _write_episode_frames(
                        dataset,
                        episode_dir,
                        file,
                        state,
                        episode_slice.cycle.start,
                        image_preprocessor,
                        args.sam_batch_size,
                        task_prompt,
                        episode_slice.output_name,
                        preview_writer,
                        detection_anchor_local_index,
                    )
                if episode_written:
                    dataset.save_episode()
                    manifest_entry = dict(
                        manifest_by_output_index[episode_slice.output_index]
                    )
                    manifest_entry["planned_episode"] = manifest_entry["episode"]
                    manifest_entry["episode"] = f"episode{len(saved_manifest)}"
                    saved_manifest.append(manifest_entry)
                else:
                    skipped_episode_count += 1
        finally:
            try:
                if saved_manifest:
                    dataset.finalize()
            finally:
                if preview_writer is not None:
                    preview_writer.close()

        if not saved_manifest:
            raise RuntimeError(
                "SAM 3 could not initialize both camera trackers in any selected episode; "
                "no LeRobot dataset was written"
            )
        _convert_lerobot_v3_to_v21(staging_v3, staging_v21)
        with (staging_v21 / "grasp_color_manifest.json").open("w", encoding="utf-8") as manifest_file:
            json.dump(saved_manifest, manifest_file, indent=2)
            manifest_file.write("\n")

        if overwrite:
            shutil.rmtree(output_path)
        shutil.move(staging_v21, output_path)

    print(
        f"Saved {len(saved_manifest)} episodes to {output_path}; skipped "
        f"{skipped_episode_count} episodes without valid two-camera SAM 3 initialization"
    )
    if preview_video_path is not None:
        print(f"Saved preview video to {preview_video_path}")
    if args.push_to_hub:
        from huggingface_hub import HfApi

        hub_api = HfApi()
        hub_api.create_repo(args.repo_id, repo_type="dataset", exist_ok=True)
        hub_api.upload_folder(
            repo_id=args.repo_id,
            repo_type="dataset",
            folder_path=output_path,
        )


if __name__ == "__main__":
    main(tyro.cli(Args))

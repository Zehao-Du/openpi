"""Convert double-close-delimited Pika pull-stick recordings to LeRobot.

The operator marks the boundary between two demonstrations by closing the
gripper twice quickly.  A boundary is placed halfway through the open interval
between those two short closures.  Recording start/end are retained, matching
the behavior of ``convert_pika_data_to_lerobot.py``.

The source episode directories must first contain the aligned ``data.hdf5``
produced by the Pika data tools.  Validate all detected slices without writing:

    uv run examples/realman_pika/pull_stick/convert_double_close_pika_data_to_lerobot.py \
        --test-mode

Build target LeRobot dataset D from two existing LeRobot datasets A and B,
followed by converted Pika recordings C:

    uv run --project examples/realman_pika --no-sync python \
        examples/realman_pika/pull_stick/convert_double_close_pika_data_to_lerobot.py \
        --append-from-repo-id Zehao123/A \
        --append-lerobot-repo-id Zehao123/B \
        --data-dir /absolute/path/to/C --repo-id Zehao123/D
"""

from __future__ import annotations

# Reuse the established pull-stick converter's I/O and LeRobot helpers.
# ruff: noqa: SLF001
import dataclasses
import importlib.util
import itertools
import json
from pathlib import Path
import shutil
import sys

import h5py
import numpy as np
from tqdm.auto import tqdm
import tyro

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("pika_pull_stick_converter", HERE / "convert_pika_data_to_lerobot.py")
pull = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = pull
spec.loader.exec_module(pull)
base = pull.base

DEFAULT_DATA_DIR = Path(
    "/inspire/hdd/project/robot-dna/baojiachun-CZXS25130063/zehao/dataset/pika/pull_stick/0827_37_gripper"
)
DEFAULT_REPO_ID = "Zehao123/pika_pull_stick_0827_37_gripper_224_224"


@dataclasses.dataclass
class SplitConfig:
    """Thresholds in aligned 30 Hz frames."""

    closed_threshold: float = 0.020
    gripper_median_window: int = 3
    min_quick_close_frames: int = 3
    max_quick_close_frames: int = 20
    max_between_closes_frames: int = 30
    min_episode_frames: int = 30
    min_sustained_close_frames: int = 30


@dataclasses.dataclass(frozen=True)
class Separator:
    left_start: int
    left_end: int
    right_start: int
    right_end: int
    boundary: int


@dataclasses.dataclass(frozen=True)
class PlannedSlice:
    source_dir: Path
    start: int
    end: int
    sustained_closes: tuple[tuple[int, int], ...]
    left_separator: Separator | None
    right_separator: Separator | None


@dataclasses.dataclass
class Args:
    data_dir: Path = DEFAULT_DATA_DIR
    repo_id: str = DEFAULT_REPO_ID
    overwrite: bool = False
    append_from_repo_id: str | None = None
    append_lerobot_repo_id: str | None = None
    task_prompt: str = pull.TASK_PROMPT
    rewrite_existing_prompt: bool = False
    rewrite_task_index: int = 0
    max_recordings: int | None = None
    push_to_hub: bool = False
    test_mode: bool = False
    debug_video_dir: Path | None = None
    debug_overwrite: bool = False
    strict_splitting: bool = False
    split: SplitConfig = dataclasses.field(default_factory=SplitConfig)


def _validate_config(config: SplitConfig) -> None:
    counts = (
        config.gripper_median_window,
        config.min_quick_close_frames,
        config.max_quick_close_frames,
        config.max_between_closes_frames,
        config.min_episode_frames,
        config.min_sustained_close_frames,
    )
    if min(counts) < 1:
        raise ValueError("All split frame counts must be positive")
    if config.gripper_median_window % 2 == 0:
        raise ValueError("gripper_median_window must be odd")
    if config.min_quick_close_frames > config.max_quick_close_frames:
        raise ValueError("min_quick_close_frames must not exceed max_quick_close_frames")


def _smooth_gripper(gripper: np.ndarray, window: int) -> np.ndarray:
    radius = window // 2
    padded = np.pad(gripper, (radius, radius), mode="edge")
    return np.median(np.lib.stride_tricks.sliding_window_view(padded, window), axis=-1)


def detect_double_close_slices(
    gripper: np.ndarray, config: SplitConfig
) -> tuple[list[tuple[int, int]], list[Separator], list[str]]:
    """Return slices separated by two adjacent quick gripper closures."""

    _validate_config(config)
    gripper = np.asarray(gripper, dtype=float).reshape(-1)
    if not len(gripper) or not np.isfinite(gripper).all():
        raise ValueError("gripper must be a finite, non-empty signal")

    smooth = _smooth_gripper(gripper, config.gripper_median_window)
    closed_runs = pull._runs(smooth <= config.closed_threshold, 1)
    quick_runs = [
        (start, end)
        for start, end in closed_runs
        if config.min_quick_close_frames <= end - start <= config.max_quick_close_frames
    ]

    separators: list[Separator] = []
    index = 0
    while index + 1 < len(quick_runs):
        left, right = quick_runs[index], quick_runs[index + 1]
        open_frames = right[0] - left[1]
        if 1 <= open_frames <= config.max_between_closes_frames:
            boundary = left[1] + open_frames // 2
            separators.append(Separator(*left, *right, boundary))
            index += 2
        else:
            index += 1

    boundaries = [0, *(item.boundary for item in separators), len(gripper)]
    slices: list[tuple[int, int]] = []
    warnings: list[str] = []
    for start, end in itertools.pairwise(boundaries):
        if end - start < config.min_episode_frames:
            warnings.append(f"[{start}:{end}] slice is shorter than {config.min_episode_frames} frames")
            continue
        sustained = [
            (run_start, run_end)
            for run_start, run_end in closed_runs
            if run_end - run_start >= config.min_sustained_close_frames
            and start <= (run_start + run_end - 1) // 2 < end
        ]
        if not sustained:
            warnings.append(f"[{start}:{end}] contains no sustained task closure")
            continue
        if len(sustained) > 1:
            warnings.append(
                f"[{start}:{end}] contains {len(sustained)} sustained task closures; "
                "a double-close separator may be missing"
            )
        slices.append((start, end))

    return slices, separators, warnings


def _plan(episode_dirs: list[Path], config: SplitConfig, *, strict: bool) -> tuple[list[PlannedSlice], list[str]]:
    planned: list[PlannedSlice] = []
    all_warnings: list[str] = []
    for episode_dir in episode_dirs:
        with h5py.File(episode_dir / "data.hdf5", "r") as file:
            if base.GRIPPER_KEY not in file:
                raise KeyError(f"{episode_dir}: missing HDF5 key {base.GRIPPER_KEY}")
            gripper = np.asarray(file[base.GRIPPER_KEY], dtype=float)
        slices, separators, warnings = detect_double_close_slices(gripper, config)
        if not separators:
            warnings.append("no double-close separator found")
        if not slices:
            warnings.append("no valid demonstration slice found")
        all_warnings.extend(f"{episode_dir}: {warning}" for warning in warnings)
        for start, end in slices:
            sustained = tuple(
                run
                for run in pull._runs(
                    _smooth_gripper(gripper.reshape(-1), config.gripper_median_window) <= config.closed_threshold,
                    config.min_sustained_close_frames,
                )
                if start <= (run[0] + run[1] - 1) // 2 < end
            )
            left = next((item for item in separators if item.boundary == start), None)
            right = next((item for item in separators if item.boundary == end), None)
            planned.append(PlannedSlice(episode_dir, start, end, sustained, left, right))
    if strict and all_warnings:
        raise ValueError(f"Splitting produced {len(all_warnings)} warnings: {all_warnings[:10]}")
    return planned, all_warnings


def _separator_dict(separator: Separator | None) -> dict[str, int] | None:
    return dataclasses.asdict(separator) if separator is not None else None


def _validate_append_dataset(dataset, output: Path) -> None:
    expected = {
        "image": (("image", "video"), (base.IMAGE_SIZE, base.IMAGE_SIZE, 3)),
        "wrist_image": (("image", "video"), (base.IMAGE_SIZE, base.IMAGE_SIZE, 3)),
        "state": (("float32",), (7,)),
        "actions": (("float32",), (7,)),
    }
    automatic = {"timestamp", "frame_index", "episode_index", "index", "task_index"}
    user_features = {
        name: feature
        for name, feature in dataset.meta.features.items()
        if name not in automatic
    }
    if set(user_features) != set(expected):
        raise ValueError(
            f"{output}: incompatible user features; expected {sorted(expected)}, "
            f"got {sorted(user_features)}"
        )
    if dataset.meta.fps != base.FPS:
        raise ValueError(f"{output}: expected {base.FPS} FPS, got {dataset.meta.fps}")
    for name, (allowed_dtypes, expected_shape) in expected.items():
        feature = user_features[name]
        if feature["dtype"] not in allowed_dtypes or tuple(feature["shape"]) != expected_shape:
            raise ValueError(
                f"{output}: incompatible feature {name!r}: {feature}; "
                f"expected dtype in {allowed_dtypes}, shape {expected_shape}"
            )


def _load_existing_manifest(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise ValueError(f"Expected a list of records in {path}")
    return value


def _tensor_to_numpy(value):
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _image_to_uint8_hwc(value) -> np.ndarray:
    image = _tensor_to_numpy(value)
    if image.ndim != 3:
        raise ValueError(f"Expected a 3D image, got shape {image.shape}")
    if image.shape[0] in (1, 3, 4) and image.shape[-1] not in (1, 3, 4):
        image = np.moveaxis(image, 0, -1)
    if np.issubdtype(image.dtype, np.floating):
        if not np.isfinite(image).all():
            raise ValueError("Image contains NaN or Inf")
        if image.size and float(image.max()) <= 1.0 + 1e-6:
            image = image * 255.0
        image = np.rint(image)
    return np.clip(image, 0, 255).astype(np.uint8)


def _user_features(dataset) -> dict:
    automatic = {"timestamp", "frame_index", "episode_index", "index", "task_index"}
    return {
        name: feature
        for name, feature in dataset.meta.features.items()
        if name not in automatic
    }


def _append_lerobot_dataset(target, source, features: dict) -> int:
    episodes = sorted(
        source.meta.episodes.to_list(),
        key=lambda record: record["episode_index"],
    )
    total_frames = sum(
        int(episode["dataset_to_index"]) - int(episode["dataset_from_index"])
        for episode in episodes
    )
    progress = tqdm(total=total_frames, desc="Appending LeRobot B", unit="frame")
    try:
        for episode in episodes:
            start = int(episode["dataset_from_index"])
            end = int(episode["dataset_to_index"])
            for frame_index in range(start, end):
                frame = source[frame_index]
                output = {"task": frame["task"]}
                for name, feature in features.items():
                    value = frame[name]
                    if feature["dtype"] in {"image", "video"}:
                        output[name] = _image_to_uint8_hwc(value)
                    elif hasattr(value, "detach"):
                        output[name] = value.detach().cpu().numpy()
                    else:
                        output[name] = value
                target.add_frame(output)
                progress.update()
            target.save_episode()
    finally:
        progress.close()
    return len(episodes)


def main(args: Args) -> None:
    if args.rewrite_existing_prompt:
        pull._rewrite_existing_prompt(
            args.repo_id,
            args.rewrite_task_index,
            args.task_prompt,
        )
        return
    root = args.data_dir.expanduser()
    if not root.is_absolute():
        raise ValueError("--data-dir must be absolute")
    root = root.resolve()
    direct_episode_dirs = [path for path in root.glob("episode*") if path.is_dir()]
    if direct_episode_dirs:
        missing_hdf5 = [path for path in direct_episode_dirs if not (path / "data.hdf5").is_file()]
        if missing_hdf5:
            raise FileNotFoundError(
                f"Found {len(direct_episode_dirs)} episode directories under {root}, but "
                f"{len(missing_hdf5)} have no data.hdf5. Run the Pika data_sync.py and "
                "data_to_hdf5.py tools first."
            )
        episode_dirs = sorted(direct_episode_dirs, key=lambda path: base._episode_sort_key(path, root))
    else:
        episode_dirs = base._find_episode_dirs(root)
    if args.max_recordings is not None:
        if args.max_recordings < 1:
            raise ValueError("--max-recordings must be positive")
        episode_dirs = episode_dirs[: args.max_recordings]

    slices, warnings = _plan(episode_dirs, args.split, strict=args.strict_splitting)
    for index, item in enumerate(slices):
        source = item.source_dir.relative_to(root)
        print(f"episode {index}: {source} [{item.start}:{item.end}] sustained_closes={list(item.sustained_closes)}")
    for warning in warnings:
        print(f"Warning: {warning}")
    print(
        f"Planned {len(slices)} demonstrations from {len(episode_dirs)} recordings; generated {len(warnings)} warnings"
    )
    if args.test_mode:
        return

    if args.debug_video_dir is not None:
        pull._write_debug_videos(
            args.debug_video_dir,
            [(item.source_dir, item.start, item.end) for item in slices],
            args.task_prompt,
            overwrite=args.debug_overwrite,
        )

    output = pull.HF_LEROBOT_HOME / args.repo_id
    append_source = None
    base_dataset = None
    if args.append_from_repo_id is not None:
        append_from_repo_id = args.append_from_repo_id.strip()
        if not append_from_repo_id:
            raise ValueError("--append-from-repo-id must not be empty")
        append_source = (pull.HF_LEROBOT_HOME / append_from_repo_id).resolve()
        if append_source == output.resolve():
            raise ValueError("--append-from-repo-id and --repo-id must be different")
        if not append_source.is_dir():
            raise FileNotFoundError(f"Append source dataset does not exist: {append_source}")
        base_dataset = base.LeRobotDataset(
            repo_id=append_from_repo_id,
            root=append_source,
        )
        _validate_append_dataset(base_dataset, append_source)
        print(
            f"Copy source A validated: {base_dataset.meta.total_episodes} episodes, "
            f"{base_dataset.meta.total_frames} frames"
        )

    lerobot_append_source = None
    lerobot_append_dataset = None
    if args.append_lerobot_repo_id is not None:
        if append_source is None:
            raise ValueError(
                "--append-lerobot-repo-id requires --append-from-repo-id "
                "so A can be copied before B is appended"
            )
        append_lerobot_repo_id = args.append_lerobot_repo_id.strip()
        if not append_lerobot_repo_id:
            raise ValueError("--append-lerobot-repo-id must not be empty")
        lerobot_append_source = (
            pull.HF_LEROBOT_HOME / append_lerobot_repo_id
        ).resolve()
        if lerobot_append_source in {append_source, output.resolve()}:
            raise ValueError("LeRobot A, B, and target D must be different datasets")
        if not lerobot_append_source.is_dir():
            raise FileNotFoundError(
                f"LeRobot append dataset B does not exist: {lerobot_append_source}"
            )
        lerobot_append_dataset = base.LeRobotDataset(
            repo_id=append_lerobot_repo_id,
            root=lerobot_append_source,
        )
        _validate_append_dataset(lerobot_append_dataset, lerobot_append_source)
        if lerobot_append_dataset.meta.robot_type != base_dataset.meta.robot_type:
            raise ValueError(
                "robot_type mismatch between LeRobot A and B: "
                f"{base_dataset.meta.robot_type!r} != "
                f"{lerobot_append_dataset.meta.robot_type!r}"
            )
        print(
            f"Append source B validated: "
            f"{lerobot_append_dataset.meta.total_episodes} episodes, "
            f"{lerobot_append_dataset.meta.total_frames} frames"
        )

    if output.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output exists: {output}. Pass --overwrite to replace it.")
        shutil.rmtree(output)

    manifest_path = output / "double_close_slice_manifest.json"
    if append_source is not None:
        print(f"Copying append source without re-encoding: {append_source} -> {output}")
        shutil.copytree(append_source, output)
        dataset = base.LeRobotDataset.resume(
            repo_id=args.repo_id,
            root=output,
            image_writer_threads=10,
            image_writer_processes=5,
        )
        starting_episode_index = dataset.meta.total_episodes
        manifest = _load_existing_manifest(manifest_path)
        print(f"Appending after {starting_episode_index} copied episodes in {output}")
    else:
        dataset = base._create_dataset(args.repo_id)
        starting_episode_index = 0
        manifest: list[dict] = []

    existing_manifest_count = len(manifest)

    try:
        if lerobot_append_dataset is not None:
            appended_episodes = _append_lerobot_dataset(
                dataset,
                lerobot_append_dataset,
                _user_features(dataset),
            )
            print(
                f"Appended {appended_episodes} episodes from LeRobot B; "
                f"target now has {dataset.meta.total_episodes} episodes"
            )
            starting_episode_index = dataset.meta.total_episodes

        for item in tqdm(slices, desc="Converting double-close slices"):
            with h5py.File(item.source_dir / "data.hdf5", "r") as file:
                required = (
                    base.TCP_KEY,
                    base.GRIPPER_KEY,
                    base.FISHEYE_KEY,
                    base.DEPTH_CAMERA_RGB_KEY,
                )
                if any(key not in file for key in required):
                    raise ValueError(f"{item.source_dir}: missing required data")
                source_length = len(file[base.TCP_KEY])
                if any(len(file[key]) != source_length for key in required[1:]):
                    raise ValueError(f"{item.source_dir}: misaligned required data")
                state = base._read_state(file, item.start, item.end)
                for local_index, source_index in enumerate(range(item.start, item.end)):
                    dataset.add_frame(
                        {
                            "image": base._read_rgb(item.source_dir, file[base.FISHEYE_KEY][source_index]),
                            "wrist_image": base._read_rgb(
                                item.source_dir, file[base.DEPTH_CAMERA_RGB_KEY][source_index]
                            ),
                            "state": state[local_index],
                            "actions": state[local_index].copy(),
                            "task": args.task_prompt,
                        }
                    )
            dataset.save_episode()
            manifest.append(
                {
                    "episode_index": (
                        starting_episode_index + len(manifest) - existing_manifest_count
                    ),
                    "source_episode": str(item.source_dir.relative_to(root)),
                    "start": item.start,
                    "end": item.end,
                    "sustained_closes": item.sustained_closes,
                    "left_separator": _separator_dict(item.left_separator),
                    "right_separator": _separator_dict(item.right_separator),
                }
            )
    finally:
        if hasattr(dataset, "stop_image_writer"):
            dataset.stop_image_writer()
        else:
            dataset.finalize()

    with manifest_path.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)
        file.write("\n")

    if args.push_to_hub:
        dataset.push_to_hub(
            tags=["pika", "realman", "pull-stick"],
            private=False,
            push_videos=True,
            license="apache-2.0",
        )


if __name__ == "__main__":
    main(tyro.cli(Args))

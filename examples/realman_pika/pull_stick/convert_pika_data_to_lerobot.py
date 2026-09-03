"""Convert continuous Pika pull-stick recordings into split LeRobot episodes.

Long, stable gripper closures identify pull events. The highest smoothed TCP z
between adjacent long-closure intervals becomes their shared boundary. The original
recording start and end are retained for the first and last slices.
"""

from __future__ import annotations

# Reuse the reference converter helpers intentionally.
# ruff: noqa: SLF001
import dataclasses
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import types

import h5py

try:
    from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
except ModuleNotFoundError:
    # examples/realman_pika pins a newer LeRobot fork with a shorter module path.
    import lerobot.datasets.lerobot_dataset as _lerobot_dataset
    from lerobot.utils.constants import HF_LEROBOT_HOME

    _lerobot_dataset.HF_LEROBOT_HOME = HF_LEROBOT_HOME
    common = types.ModuleType("lerobot.common")
    datasets = types.ModuleType("lerobot.common.datasets")
    sys.modules.setdefault("lerobot.common", common)
    sys.modules.setdefault("lerobot.common.datasets", datasets)
    sys.modules["lerobot.common.datasets.lerobot_dataset"] = _lerobot_dataset
import numpy as np
from tqdm.auto import tqdm
import tyro

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent
sys.path.insert(0, str(PARENT))
sys.path.insert(0, str(PARENT / "collect_block"))
spec = importlib.util.spec_from_file_location(
    "pika_three_grasps_converter", PARENT / "collect_block" / "convert_pika_three_grasps_to_lerobot.py"
)
base = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = base
spec.loader.exec_module(base)

DEFAULT_DATA_DIR = Path("/inspire/hdd/project/robot-dna/baojiachun-CZXS25130063/zehao/dataset/pika/pull_stick")
DEFAULT_REPO_ID = "Zehao123/pika_pull_stick_224_224"
TASK_PROMPT = "pull the stick"


@dataclasses.dataclass
class SplitConfig:
    closed_threshold: float = 0.020
    gripper_median_window: int = 5
    min_closed_frames: int = 30
    min_reopen_frames: int = 30
    min_episode_frames: int = 30
    z_smoothing_window: int = 7

@dataclasses.dataclass
class Args:
    data_dir: Path = DEFAULT_DATA_DIR
    repo_id: str = DEFAULT_REPO_ID
    task_prompt: str = TASK_PROMPT
    overwrite: bool = False
    rewrite_existing_prompt: bool = False
    rewrite_task_index: int = 0
    max_recordings: int | None = None
    push_to_hub: bool = False
    test_mode: bool = False
    debug_video_dir: Path | None = None
    debug_overwrite: bool = False
    strict_splitting: bool = False
    split: SplitConfig = dataclasses.field(default_factory=SplitConfig)


def _runs(mask, minimum):
    padded = np.pad(np.asarray(mask, dtype=bool), (1, 1))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return [(int(a), int(b)) for a, b in zip(edges[::2], edges[1::2], strict=True) if b - a >= minimum]


def _merge_short_reopenings(runs, minimum_reopen_frames):
    """Merge closures separated only by a brief gripper reopening."""
    merged = []
    for start, end in runs:
        if merged and start - merged[-1][1] < minimum_reopen_frames:
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))
    return merged


def detect_pull_slices(z, gripper, config):
    """Detect pulls from long closures and split at z maxima in the gaps between them."""
    z = np.asarray(z, dtype=float).reshape(-1)
    gripper = np.asarray(gripper, dtype=float).reshape(-1)
    if len(z) != len(gripper) or not len(z) or not np.isfinite(z).all() or not np.isfinite(gripper).all():
        raise ValueError("z/gripper must be equally sized, finite, non-empty signals")
    counts = (
        config.gripper_median_window,
        config.min_closed_frames,
        config.min_reopen_frames,
        config.min_episode_frames,
        config.z_smoothing_window,
    )
    if min(counts) < 1:
        raise ValueError("Split frame counts must be positive")
    if config.gripper_median_window % 2 == 0 or config.z_smoothing_window % 2 == 0:
        raise ValueError("Smoothing windows must be odd")

    gripper_radius = config.gripper_median_window // 2
    padded_gripper = np.pad(gripper, (gripper_radius, gripper_radius), mode="edge")
    smooth_gripper = np.median(
        np.lib.stride_tricks.sliding_window_view(
            padded_gripper, config.gripper_median_window
        ),
        axis=-1,
    )
    closed_runs = _merge_short_reopenings(
        _runs(smooth_gripper <= config.closed_threshold, 1),
        config.min_reopen_frames,
    )
    closed_runs = [
        (start, end)
        for start, end in closed_runs
        if end - start >= config.min_closed_frames
    ]
    closure_centers = [(run_start + run_end - 1) // 2 for run_start, run_end in closed_runs]

    z_radius = config.z_smoothing_window // 2
    smooth_z = np.convolve(
        np.pad(z, (z_radius, z_radius), mode="edge"),
        np.ones(config.z_smoothing_window) / config.z_smoothing_window,
        mode="valid",
    )
    z_maxima = []
    for left_run, right_run in zip(closed_runs, closed_runs[1:]):
        # Search only after the preceding closure has ended and before the next
        # closure begins. This keeps both closure events inside their own slice.
        gap_start, gap_end = left_run[1], right_run[0]
        boundary = gap_start + int(np.argmax(smooth_z[gap_start:gap_end]))
        z_maxima.append(boundary)
    boundaries = [0, *z_maxima, len(z)] if closed_runs else []

    accepted, warnings = [], []
    for closed_run, center, start, end in zip(
        closed_runs,
        closure_centers,
        boundaries[:-1],
        boundaries[1:],
        strict=True,
    ):
        if not start <= center < end:
            warnings.append((start, end, f"closure center {center} falls outside slice"))
            continue
        if end - start < config.min_episode_frames:
            warnings.append((start, end, "z-maximum-bounded slice too short"))
            continue
        anchor = (center, center + 1)
        accepted.append((start, end, closed_run[0], closed_run[1], anchor, anchor))
    return accepted, warnings

def _plan(dirs, config, strict):
    good, bad = [], []
    for directory in dirs:
        with h5py.File(directory / "data.hdf5") as f:
            pose, grip = np.asarray(f[base.TCP_KEY]), np.asarray(f[base.GRIPPER_KEY])
        cuts, rejected = detect_pull_slices(pose[:, 2], grip, config)
        if not cuts:
            message = f"{directory}: no complete pull cycle"
            if strict:
                raise ValueError(message)
            print(f"Warning: {message}; skipping recording")
            bad += [(directory, 0, len(grip), "no complete pull cycle")]
            continue
        good += [(directory, *cut, float(pose[:, 2].min())) for cut in cuts]
        bad += [(directory, *item) for item in rejected]
    if strict and bad:
        raise ValueError(f"Rejected {len(bad)} incomplete candidates: {bad[:10]}")
    return good, bad


def _write_debug_videos(output_dir: Path, slices, task_prompt: str, *, overwrite: bool) -> None:
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = list(output_dir.glob("*.mp4"))
    if existing and not overwrite:
        raise FileExistsError(
            f"{len(existing)} MP4 files already exist in {output_dir}; pass --debug-overwrite to replace them"
        )
    if overwrite:
        for path in existing:
            path.unlink()

    for index, item in enumerate(tqdm(slices, desc="Writing debug videos", unit="video")):
        source_dir, start, end = item[:3]
        name = f"episode_{index:04d}_{source_dir.parent.name}_{source_dir.name}_{start}_{end}.mp4"
        writer = base._DebugVideoWriter(output_dir / name, base.FPS)
        try:
            with h5py.File(source_dir / "data.hdf5") as file:
                required = (base.FISHEYE_KEY, base.DEPTH_CAMERA_RGB_KEY)
                missing = [key for key in required if key not in file]
                if missing:
                    raise KeyError(f"{source_dir}: missing camera keys {missing}")
                for frame_index in range(start, end):
                    fisheye = base._read_rgb(source_dir, file[base.FISHEYE_KEY][frame_index])
                    wrist = base._read_rgb(source_dir, file[base.DEPTH_CAMERA_RGB_KEY][frame_index])
                    writer.add_frame(base._make_debug_frame(fisheye, wrist, task_prompt))
        finally:
            writer.close()
    print(f"Saved {len(slices)} debug videos to {output_dir}")


def _rewrite_existing_prompt(repo_id: str, task_index: int, prompt: str) -> None:
    """Atomically replace one task prompt in an existing LeRobot dataset."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    if not repo_id.strip():
        raise ValueError("--repo-id must not be empty")
    if task_index < 0:
        raise ValueError("--rewrite-task-index must be non-negative")
    new_prompt = prompt.strip()
    if not new_prompt:
        raise ValueError("--task-prompt must not be empty")

    tasks_path = HF_LEROBOT_HOME / repo_id / "meta" / "tasks.parquet"
    if not tasks_path.is_file():
        raise FileNotFoundError(f"LeRobot task metadata not found: {tasks_path}")
    table = pq.read_table(tasks_path)
    required = {"task_index", "task"}
    if not required.issubset(table.column_names):
        raise ValueError(
            f"{tasks_path} must contain {sorted(required)}, got {table.column_names}"
        )
    task_indices = table["task_index"].to_pylist()
    matches = [
        row_index
        for row_index, value in enumerate(task_indices)
        if value == task_index
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one task_index={task_index} in {tasks_path}, "
            f"found {len(matches)}"
        )
    row_index = matches[0]
    tasks = table["task"].to_pylist()
    old_prompt = tasks[row_index]
    tasks[row_index] = new_prompt
    column_index = table.schema.get_field_index("task")
    field = table.schema.field(column_index)
    table = table.set_column(
        column_index,
        field,
        pa.array(tasks, type=field.type),
    )
    temporary_path = tasks_path.with_name(f".{tasks_path.name}.tmp")
    try:
        pq.write_table(table, temporary_path)
        temporary_path.replace(tasks_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    print(
        f"Updated task_index={task_index} in {tasks_path}: "
        f"{old_prompt!r} -> {new_prompt!r}"
    )


def main(args: Args):
    if args.rewrite_existing_prompt:
        _rewrite_existing_prompt(
            args.repo_id,
            args.rewrite_task_index,
            args.task_prompt,
        )
        return
    root = args.data_dir.expanduser()
    if not root.is_absolute():
        raise ValueError("--data-dir must be absolute")
    root = root.resolve()
    dirs = base._find_episode_dirs(root)
    if args.max_recordings is not None:
        if args.max_recordings < 1:
            raise ValueError("--max-recordings must be positive")
        dirs = dirs[: args.max_recordings]
    slices, rejected = _plan(dirs, args.split, args.strict_splitting)
    for i, (path, start, end, cs, ce, _sa, _ea, zmin) in enumerate(slices):
        print(f"episode {i}: {path.relative_to(root)} [{start}:{end}] closed=[{cs}:{ce}] z_min={zmin:.5f}")
    for path, start, end, reason in rejected:
        print(f"Warning {path.relative_to(root)} [{start}:{end}]: {reason}")
    print(f"Planned {len(slices)} pulls from {len(dirs)} recordings; rejected {len(rejected)} candidates")
    if args.test_mode:
        return
    if args.debug_video_dir is not None:
        _write_debug_videos(
            args.debug_video_dir,
            slices,
            args.task_prompt,
            overwrite=args.debug_overwrite,
        )
    output = HF_LEROBOT_HOME / args.repo_id
    if output.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output exists: {output}. Pass --overwrite to replace it.")
        shutil.rmtree(output)
    dataset = base._create_dataset(args.repo_id)
    try:
        for item in tqdm(slices, desc="Converting pulls"):
            path, start, end = item[:3]
            with h5py.File(path / "data.hdf5") as f:
                required = (base.TCP_KEY, base.GRIPPER_KEY, base.FISHEYE_KEY, base.DEPTH_CAMERA_RGB_KEY)
                if any(k not in f for k in required) or any(len(f[k]) != len(f[base.TCP_KEY]) for k in required[1:]):
                    raise ValueError(f"{path}: missing or misaligned data")
                state = base._read_state(f, start, end)
                for local, source in enumerate(range(start, end)):
                    dataset.add_frame(
                        {
                            "image": base._read_rgb(path, f[base.FISHEYE_KEY][source]),
                            "wrist_image": base._read_rgb(path, f[base.DEPTH_CAMERA_RGB_KEY][source]),
                            "state": state[local],
                            "actions": state[local].copy(),
                            "task": args.task_prompt,
                        }
                    )
            dataset.save_episode()
    finally:
        if hasattr(dataset, "stop_image_writer"):
            dataset.stop_image_writer()
        else:
            dataset.finalize()
    manifest = [
        {
            "episode_index": i,
            "source_episode": str(x[0].relative_to(root)),
            "start": x[1],
            "end": x[2],
            "closed_start": x[3],
            "closed_end": x[4],
            "start_anchor": x[5],
            "end_anchor": x[6],
            "z_min": x[7],
        }
        for i, x in enumerate(slices)
    ]
    with (output / "pull_slice_manifest.json").open("w") as f:
        json.dump(manifest, f, indent=2)
    if args.push_to_hub:
        dataset.push_to_hub(
            tags=["pika", "realman", "pull-stick"], private=False, push_videos=True, license="apache-2.0"
        )


if __name__ == "__main__":
    main(tyro.cli(Args))

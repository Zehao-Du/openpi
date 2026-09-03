"""Split Pika recordings into episodes containing exactly one completed pick-and-place.

The gripper encoder is treated as a hysteretic state signal.  A completed
operation is one stable open -> closed -> open cycle; leading/trailing device
startup values are ignored.  Every time-aligned HDF5 dataset is sliced, static
calibration datasets are preserved, and only files referenced by the slice are
copied.

Example:
uv run --project examples/realman_pika python \
    examples/realman_pika/collect_block/split_pika_data_by_grasp.py \
    --data-dir /absolute/path/to/collect_blocks \
    --output-dir /absolute/path/to/collect_blocks_single_grasp

Inspect the proposed boundaries without writing data by adding ``--dry-run``.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any

import h5py
import numpy as np
import tyro

GRIPPER_KEY = "gripper/encoderDistance/pika"
SIZE_KEY = "size"


@dataclasses.dataclass(frozen=True)
class GraspCycle:
    """Half-open source interval and key transitions for one pick-and-place."""

    start: int
    close: int
    release: int
    end: int


@dataclasses.dataclass
class Args:
    data_dir: Path
    output_dir: Path
    open_threshold: float = 0.085
    closed_threshold: float = 0.075
    min_state_frames: int = 3
    post_release_frames: int = 10
    overwrite: bool = False
    dry_run: bool = False


def _episode_sort_key(path: Path) -> tuple[int, int | str]:
    match = re.fullmatch(r"episode(\d+)", path.name)
    return (0, int(match.group(1))) if match else (1, path.name)


def _find_episode_dirs(data_dir: Path) -> list[Path]:
    episodes = sorted(
        (
            path
            for path in data_dir.iterdir()
            if path.is_dir() and path.name.startswith("episode") and (path / "data.hdf5").is_file()
        ),
        key=_episode_sort_key,
    )
    if not episodes:
        raise FileNotFoundError(f"No episode*/data.hdf5 directories found under {data_dir}")
    return episodes


def detect_grasp_cycles(
    gripper: np.ndarray,
    *,
    open_threshold: float = 0.085,
    closed_threshold: float = 0.075,
    min_state_frames: int = 3,
    post_release_frames: int = 10,
) -> list[GraspCycle]:
    """Detect stable open -> closed -> open cycles in a gripper-distance trace."""
    gripper = np.asarray(gripper, dtype=np.float64).reshape(-1)
    if not np.isfinite(gripper).all():
        raise ValueError("Gripper trace contains NaN or Inf values")
    if open_threshold <= closed_threshold:
        raise ValueError("open_threshold must be greater than closed_threshold")
    if min_state_frames < 1:
        raise ValueError("min_state_frames must be at least 1")
    if post_release_frames < 0:
        raise ValueError("post_release_frames must be non-negative")

    phase = "first_open"
    stable_count = 0
    first_open: int | None = None
    close: int | None = None
    transitions: list[tuple[int, int]] = []

    for index, value in enumerate(gripper):
        expected = value >= open_threshold if phase in {"first_open", "release"} else value <= closed_threshold
        stable_count = stable_count + 1 if expected else 0
        if stable_count < min_state_frames:
            continue

        transition = index - min_state_frames + 1
        stable_count = 0
        if phase == "first_open":
            first_open = transition
            phase = "close"
        elif phase == "close":
            close = transition
            phase = "release"
        else:
            assert close is not None
            transitions.append((close, transition))
            close = None
            phase = "close"

    if first_open is None:
        return []

    cycles: list[GraspCycle] = []
    start = first_open
    for close_index, release_index in transitions:
        end = min(len(gripper), release_index + 1 + post_release_frames)
        if start >= close_index:
            raise ValueError(
                "post_release_frames consumes the next approach/grasp interval; "
                f"segment start {start} is not before close {close_index}"
            )
        cycles.append(GraspCycle(start=start, close=close_index, release=release_index, end=end))
        start = end
    return cycles


def _copy_attrs(source: h5py.AttributeManager, destination: h5py.AttributeManager) -> None:
    for key, value in source.items():
        destination[key] = value


def _create_dataset(destination: h5py.Group, name: str, source: h5py.Dataset, data: Any) -> h5py.Dataset:
    array = np.asarray(data)
    kwargs: dict[str, Any] = {"dtype": source.dtype}
    if array.ndim > 0 and source.chunks is not None:
        kwargs["chunks"] = True
    if source.compression is not None:
        kwargs["compression"] = source.compression
        kwargs["compression_opts"] = source.compression_opts
    if source.shuffle:
        kwargs["shuffle"] = True
    if source.fletcher32:
        kwargs["fletcher32"] = True
    dataset = destination.create_dataset(name, data=data, **kwargs)
    _copy_attrs(source.attrs, dataset.attrs)
    return dataset


def _path_values(data: Any) -> set[Path]:
    values = np.asarray(data, dtype=object).reshape(-1)
    paths: set[Path] = set()
    for value in values:
        decoded = value.decode("utf-8") if isinstance(value, bytes) else value
        if isinstance(decoded, str) and decoded:
            paths.add(Path(decoded))
    return paths


def _write_hdf5_slice(
    source_path: Path,
    destination_path: Path,
    cycle: GraspCycle,
    source_length: int,
) -> tuple[set[Path], set[Path]]:
    referenced_paths: set[Path] = set()
    required_paths: set[Path] = set()
    with h5py.File(source_path, "r") as source, h5py.File(destination_path, "w") as destination:
        _copy_attrs(source.attrs, destination.attrs)

        def copy_item(name: str, item: h5py.Group | h5py.Dataset) -> None:
            if isinstance(item, h5py.Group):
                group = destination.require_group(name)
                _copy_attrs(item.attrs, group.attrs)
                return

            parent_name, dataset_name = name.rsplit("/", 1) if "/" in name else ("", name)
            parent = destination.require_group(parent_name) if parent_name else destination
            if name == SIZE_KEY and item.shape == ():
                data: Any = np.asarray(cycle.end - cycle.start, dtype=item.dtype)
            elif item.shape and item.shape[0] == source_length:
                data = item[cycle.start : cycle.end]
            else:
                data = item[()]
            _create_dataset(parent, dataset_name, item, data)
            if item.dtype.kind in {"O", "S", "U"}:
                item_paths = _path_values(data)
                referenced_paths.update(item_paths)
                if item.shape and item.shape[0] == source_length:
                    required_paths.update(item_paths)

        source.visititems(copy_item)
    return referenced_paths, required_paths


def _copy_referenced_files(
    source_episode: Path,
    destination_episode: Path,
    paths: set[Path],
    required_paths: set[Path],
) -> None:
    source_root = source_episode.resolve()
    for relative_path in sorted(paths):
        if relative_path.is_absolute():
            raise ValueError(f"HDF5 contains an absolute referenced path: {relative_path}")
        source_path = (source_episode / relative_path).resolve()
        if not source_path.is_relative_to(source_root):
            raise ValueError(f"HDF5 path escapes its episode directory: {relative_path}")
        if not source_path.is_file():
            if relative_path in required_paths:
                raise FileNotFoundError(source_path)
            print(f"Skipping missing optional referenced file: {source_path}")
            continue
        destination_path = destination_episode / relative_path
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, destination_path)

    instructions_json = source_episode / "instructions.json"
    if instructions_json.is_file():
        shutil.copy2(instructions_json, destination_episode / instructions_json.name)


def _plan_splits(episodes: list[Path], args: Args) -> list[tuple[Path, GraspCycle]]:
    plan: list[tuple[Path, GraspCycle]] = []
    for episode in episodes:
        with h5py.File(episode / "data.hdf5", "r") as file:
            if GRIPPER_KEY not in file:
                raise KeyError(f"{episode}: missing {GRIPPER_KEY}")
            gripper = np.asarray(file[GRIPPER_KEY][:], dtype=np.float64)
        cycles = detect_grasp_cycles(
            gripper,
            open_threshold=args.open_threshold,
            closed_threshold=args.closed_threshold,
            min_state_frames=args.min_state_frames,
            post_release_frames=args.post_release_frames,
        )
        if not cycles:
            raise ValueError(f"{episode}: no completed open -> closed -> open grasp cycle found")
        plan.extend((episode, cycle) for cycle in cycles)
    return plan


def split_dataset(args: Args) -> list[dict[str, int | str]]:
    data_dir = args.data_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not data_dir.is_dir():
        raise NotADirectoryError(data_dir)
    if output_dir == data_dir or output_dir.is_relative_to(data_dir):
        raise ValueError("--output-dir must not be the input directory or a directory inside it")

    episodes = _find_episode_dirs(data_dir)
    plan = _plan_splits(episodes, args)
    manifest = [
        {
            "output_episode": f"episode{output_index}",
            "source_episode": source_episode.name,
            "source_start": cycle.start,
            "source_close": cycle.close,
            "source_release": cycle.release,
            "source_end": cycle.end,
        }
        for output_index, (source_episode, cycle) in enumerate(plan)
    ]

    for entry in manifest:
        print(
            f"{entry['output_episode']}: {entry['source_episode']} "
            f"[{entry['source_start']}:{entry['source_end']}] "
            f"close={entry['source_close']} release={entry['source_release']}"
        )
    print(f"Planned {len(plan)} single-grasp episodes from {len(episodes)} source episodes")
    if args.dry_run:
        return manifest
    if output_dir.exists() and not args.overwrite:
        raise FileExistsError(f"Output directory already exists: {output_dir}; pass --overwrite to replace it")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary_dir = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}-", dir=output_dir.parent))
    try:
        for output_index, (source_episode, cycle) in enumerate(plan):
            destination_episode = temporary_dir / f"episode{output_index}"
            destination_episode.mkdir()
            with h5py.File(source_episode / "data.hdf5", "r") as source_file:
                source_length = len(source_file[GRIPPER_KEY])
            referenced_paths, required_paths = _write_hdf5_slice(
                source_episode / "data.hdf5",
                destination_episode / "data.hdf5",
                cycle,
                source_length,
            )
            _copy_referenced_files(source_episode, destination_episode, referenced_paths, required_paths)
            with (destination_episode / "split_info.json").open("w", encoding="utf-8") as file:
                json.dump(manifest[output_index], file, indent=2)
                file.write("\n")

        with (temporary_dir / "split_manifest.json").open("w", encoding="utf-8") as file:
            json.dump(manifest, file, indent=2)
            file.write("\n")
        if output_dir.exists():
            shutil.rmtree(output_dir)
        temporary_dir.replace(output_dir)
    finally:
        if temporary_dir.exists():
            shutil.rmtree(temporary_dir)

    print(f"Saved {len(plan)} single-grasp episodes to {output_dir}")
    return manifest


if __name__ == "__main__":
    split_dataset(tyro.cli(Args))

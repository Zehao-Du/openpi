"""Merge local LeRobot datasets by cloning the first and appending the rest.

Example:
uv run --project examples/realman_pika --no-sync python \
  examples/realman_pika/pull_stick/merge_lerobot_datasets.py \
  --inputs \
    Zehao123/pika_pull_stick_0827_37_gripper_224_224 \
    Zehao123/pika_pull_stick_0827_37_gripper_224_224_keypoint \
  --repo-id Zehao123/pika_pull_stick_0827_merged

Inputs may be repo IDs under ``HF_LEROBOT_HOME`` or absolute local paths. The
source datasets are read-only; the destination must be distinct from every
source. By default, the first dataset is copied without decoding or re-encoding
and later datasets are appended with LeRobot's official writer API. Use
``--no-reuse-first-dataset`` to replay every source, or ``--test-mode`` to
validate compatibility without writing data.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
import shutil
from typing import Any

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.constants import HF_LEROBOT_HOME
import numpy as np
import torch
from tqdm.auto import tqdm
import tyro

AUTO_FEATURES = {
    "timestamp",
    "frame_index",
    "episode_index",
    "index",
    "task_index",
}


@dataclasses.dataclass
class Args:
    inputs: tuple[str, ...] = ()
    repo_id: str = "Zehao123/pika_pull_stick_merged"
    output_root: Path | None = None
    robot_type: str | None = None
    max_episodes_per_dataset: int | None = None
    image_writer_threads: int = 10
    image_writer_processes: int = 5
    reuse_first_dataset: bool = True
    overwrite: bool = False
    test_mode: bool = False
    push_to_hub: bool = False


@dataclasses.dataclass(frozen=True)
class Source:
    label: str
    repo_id: str
    root: Path
    dataset: LeRobotDataset
    episodes: tuple[dict[str, Any], ...]


def _resolve_source(value: str) -> tuple[str, Path]:
    candidate = Path(value).expanduser()
    if candidate.is_absolute():
        root = candidate.resolve()
        repo_id = f"local/{root.name}"
    else:
        repo_id = value
        root = (HF_LEROBOT_HOME / value).resolve()
    if not (root / "meta/info.json").is_file():
        raise FileNotFoundError(f"Not a local LeRobot dataset: {root}")
    return repo_id, root


def _user_features(features: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {name: feature for name, feature in features.items() if name not in AUTO_FEATURES}


def _normalized(value: Any) -> Any:
    """Normalize tuples and numeric scalar types for stable schema comparison."""

    return json.loads(json.dumps(value))


def _load_sources(args: Args) -> list[Source]:
    if len(args.inputs) < 2:
        raise ValueError("--inputs requires at least two datasets")
    if args.max_episodes_per_dataset is not None and args.max_episodes_per_dataset < 1:
        raise ValueError("--max-episodes-per-dataset must be positive")

    sources = []
    for value in args.inputs:
        repo_id, root = _resolve_source(value)
        dataset = LeRobotDataset(repo_id=repo_id, root=root)
        episodes = tuple(sorted(dataset.meta.episodes.to_list(), key=lambda record: record["episode_index"]))
        if args.max_episodes_per_dataset is not None:
            episodes = episodes[: args.max_episodes_per_dataset]
        if not episodes:
            raise ValueError(f"{root}: dataset contains no selected episodes")
        sources.append(Source(value, repo_id, root, dataset, episodes))
    return sources


def _validate_sources(
    sources: list[Source], robot_type_override: str | None
) -> tuple[int, dict[str, dict[str, Any]], str | None]:
    reference = sources[0]
    fps = reference.dataset.meta.fps
    features = _user_features(reference.dataset.meta.features)
    normalized_features = _normalized(features)
    robot_type = reference.dataset.meta.robot_type

    for source in sources[1:]:
        if source.dataset.meta.fps != fps:
            raise ValueError(f"FPS mismatch: {reference.root} has {fps}, {source.root} has {source.dataset.meta.fps}")
        candidate = _user_features(source.dataset.meta.features)
        if _normalized(candidate) != normalized_features:
            reference_keys = set(features)
            candidate_keys = set(candidate)
            raise ValueError(
                f"Feature schema mismatch for {source.root}; "
                f"missing={sorted(reference_keys - candidate_keys)}, "
                f"extra={sorted(candidate_keys - reference_keys)}"
            )
        if source.dataset.meta.robot_type != robot_type and robot_type_override is None:
            raise ValueError(
                f"robot_type mismatch: {reference.root} has {robot_type!r}, "
                f"{source.root} has {source.dataset.meta.robot_type!r}. "
                "Pass --robot-type to explicitly choose the merged value."
            )
    return fps, features, robot_type


def _image_to_uint8_hwc(value: Any) -> np.ndarray:
    array = value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)
    if array.ndim != 3:
        raise ValueError(f"Expected a 3D image, got shape {array.shape}")
    if array.shape[0] in (1, 3, 4) and array.shape[-1] not in (1, 3, 4):
        array = np.moveaxis(array, 0, -1)
    if np.issubdtype(array.dtype, np.floating):
        if not np.isfinite(array).all():
            raise ValueError("Image contains NaN or Inf")
        if array.size and float(array.max()) <= 1.0 + 1e-6:
            array = array * 255.0
        array = np.rint(array)
    return np.clip(array, 0, 255).astype(np.uint8)


def _frame_for_writing(frame: dict[str, Any], features: dict[str, dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {"task": frame["task"]}
    for name, feature in features.items():
        value = frame[name]
        if feature["dtype"] in {"image", "video"}:
            output[name] = _image_to_uint8_hwc(value)
        elif isinstance(value, torch.Tensor):
            output[name] = value.detach().cpu().numpy()
        else:
            output[name] = value
    return output


def _destination(args: Args) -> Path:
    return (
        args.output_root.expanduser().resolve()
        if args.output_root is not None
        else (HF_LEROBOT_HOME / args.repo_id).resolve()
    )


def _episode_manifest_entry(
    output_episode: int,
    source: Source,
    episode: dict[str, Any],
) -> dict[str, Any]:
    start = int(episode["dataset_from_index"])
    end = int(episode["dataset_to_index"])
    return {
        "episode_index": output_episode,
        "source_dataset": source.label,
        "source_episode_index": int(episode["episode_index"]),
        "length": end - start,
    }


def _can_reuse_first_dataset(args: Args, first: Source) -> bool:
    """A byte-for-byte clone is valid only when the complete first source is selected."""

    if not args.reuse_first_dataset:
        return False
    if len(first.episodes) != first.dataset.meta.total_episodes:
        print(
            "Cannot reuse the first dataset because --max-episodes-per-dataset "
            "selected only part of it; replaying all selected sources instead."
        )
        return False
    if args.robot_type is not None and args.robot_type != first.dataset.meta.robot_type:
        print(
            "Cannot reuse the first dataset with a different --robot-type; "
            "replaying all selected sources instead."
        )
        return False
    return True


def main(args: Args) -> None:
    sources = _load_sources(args)
    fps, features, detected_robot_type = _validate_sources(sources, args.robot_type)
    destination = _destination(args)
    if any(destination == source.root for source in sources):
        raise ValueError("The destination must be different from every source dataset")

    total_episodes = sum(len(source.episodes) for source in sources)
    total_frames = sum(
        int(episode["dataset_to_index"]) - int(episode["dataset_from_index"])
        for source in sources
        for episode in source.episodes
    )
    for source in sources:
        frames = sum(
            int(episode["dataset_to_index"]) - int(episode["dataset_from_index"]) for episode in source.episodes
        )
        print(f"{source.label}: {len(source.episodes)} episodes, {frames} frames")
    print(f"Merged plan: {total_episodes} episodes, {total_frames} frames -> {destination}")
    if args.test_mode:
        return

    if destination.exists():
        if not args.overwrite:
            raise FileExistsError(f"Destination exists: {destination}; pass --overwrite to replace it")
        shutil.rmtree(destination)

    manifest: list[dict[str, Any]] = []
    reuse_first = _can_reuse_first_dataset(args, sources[0])
    if reuse_first:
        print(f"Copying first dataset without re-encoding: {sources[0].root} -> {destination}")
        shutil.copytree(sources[0].root, destination)
        output = LeRobotDataset.resume(
            repo_id=args.repo_id,
            root=destination,
            image_writer_threads=args.image_writer_threads,
            image_writer_processes=args.image_writer_processes,
        )
        for episode in sources[0].episodes:
            manifest.append(_episode_manifest_entry(len(manifest), sources[0], episode))
        sources_to_write = sources[1:]
        frames_to_write = total_frames - sum(entry["length"] for entry in manifest)
        print(
            f"Reused {len(manifest)} episodes; only {len(sources_to_write)} "
            f"remaining datasets ({frames_to_write} frames) require re-encoding"
        )
    else:
        output = LeRobotDataset.create(
            repo_id=args.repo_id,
            root=destination,
            robot_type=args.robot_type or detected_robot_type,
            fps=fps,
            features=features,
            use_videos=any(feature["dtype"] == "video" for feature in features.values()),
            image_writer_threads=args.image_writer_threads,
            image_writer_processes=args.image_writer_processes,
        )
        sources_to_write = sources
        frames_to_write = total_frames

    try:
        progress = tqdm(total=frames_to_write, desc="Appending LeRobot datasets", unit="frame")
        for source in sources_to_write:
            for episode in source.episodes:
                start = int(episode["dataset_from_index"])
                end = int(episode["dataset_to_index"])
                output_episode = len(manifest)
                for index in range(start, end):
                    output.add_frame(_frame_for_writing(source.dataset[index], features))
                    progress.update()
                output.save_episode()
                manifest.append(_episode_manifest_entry(output_episode, source, episode))
        progress.close()
    finally:
        if hasattr(output, "stop_image_writer"):
            output.stop_image_writer()
        else:
            output.finalize()

    with (destination / "merge_manifest.json").open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)
        file.write("\n")
    print(f"Saved {len(manifest)} episodes and {total_frames} frames to {destination}")
    if args.push_to_hub:
        output.push_to_hub(
            tags=["merged", "pika", "realman"],
            private=False,
            push_videos=True,
            license="apache-2.0",
        )


if __name__ == "__main__":
    main(tyro.cli(Args))

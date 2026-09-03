"""Merge ordered LeRobot v2.1 shards without decoding or re-encoding images.

Example:
uv run --project examples/realman_pika --no-sync python \
  examples/realman_pika/merge_lerobot_v21_shards.py \
  --inputs Zehao123/task_shard_0 Zehao123/task_shard_1 \
  --repo-id Zehao123/task --overwrite

The input order defines output episode order. Parquet image bytes are preserved;
only episode, frame, and global indices are rewritten.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
import shutil
import tempfile
from typing import Any

from lerobot.utils.constants import HF_LEROBOT_HOME
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm.auto import tqdm
import tyro

LEROBOT_VERSION = "v2.1"
MANIFEST_NAMES = ("pull_keypoint_manifest.json", "pull_recolor_manifest.json")


@dataclasses.dataclass
class Args:
    inputs: tuple[str, ...] = ()
    repo_id: str = ""
    output_root: Path | None = None
    overwrite: bool = False
    test_mode: bool = False


def _read_jsonlines(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def _write_jsonlines(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            json.dump(record, file, separators=(",", ":"))
            file.write("\n")


def _resolve_dataset(value: str) -> Path:
    candidate = Path(value).expanduser()
    root = candidate.resolve() if candidate.is_absolute() else (HF_LEROBOT_HOME / value).resolve()
    if not (root / "meta" / "info.json").is_file():
        raise FileNotFoundError(f"Not a local LeRobot dataset: {root}")
    return root


def _dataset_path(template: str, episode_index: int, chunks_size: int) -> Path:
    return Path(template.format(episode_chunk=episode_index // chunks_size, episode_index=episode_index))


def _replace_integer_column(table: pa.Table, name: str, values: list[int] | range) -> pa.Table:
    column_index = table.schema.get_field_index(name)
    if column_index < 0:
        raise ValueError(f"Episode Parquet is missing required column {name!r}")
    field = table.schema.field(column_index)
    return table.set_column(column_index, field, pa.array(values, type=field.type))


def _rewrite_indices(table: pa.Table, episode_index: int, global_index: int) -> pa.Table:
    length = len(table)
    table = _replace_integer_column(table, "episode_index", [episode_index] * length)
    table = _replace_integer_column(table, "frame_index", range(length))
    return _replace_integer_column(table, "index", range(global_index, global_index + length))


def _load_metadata(root: Path) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    with (root / "meta" / "info.json").open(encoding="utf-8") as file:
        info = json.load(file)
    if info.get("codebase_version") != LEROBOT_VERSION:
        raise ValueError(f"{root}: expected LeRobot {LEROBOT_VERSION}, got {info.get('codebase_version')!r}")
    if any(feature.get("dtype") == "video" for feature in info["features"].values()):
        raise ValueError(f"{root}: direct shard merge currently supports image datasets, not video features")
    episodes = sorted(_read_jsonlines(root / "meta" / "episodes.jsonl"), key=lambda item: item["episode_index"])
    stats = sorted(_read_jsonlines(root / "meta" / "episodes_stats.jsonl"), key=lambda item: item["episode_index"])
    if len(episodes) != len(stats):
        raise ValueError(f"{root}: episodes/stats length mismatch ({len(episodes)} != {len(stats)})")
    return info, episodes, stats


def main(args: Args) -> None:
    if len(args.inputs) < 2:
        raise ValueError("--inputs requires at least two ordered shards")
    if not args.repo_id.strip():
        raise ValueError("--repo-id must not be empty")
    roots = [_resolve_dataset(value) for value in args.inputs]
    destination = (
        args.output_root.expanduser().resolve()
        if args.output_root is not None
        else (HF_LEROBOT_HOME / args.repo_id).resolve()
    )
    if destination in roots:
        raise ValueError("The destination must differ from every input shard")

    metadata = [_load_metadata(root) for root in roots]
    reference_info = metadata[0][0]
    reference_tasks = _read_jsonlines(roots[0] / "meta" / "tasks.jsonl")
    manifest_names = {
        name for root in roots for name in MANIFEST_NAMES if (root / name).is_file()
    }
    if len(manifest_names) > 1:
        raise ValueError(f"Input shards use different manifest types: {sorted(manifest_names)}")
    manifest_name = next(iter(manifest_names), None)
    compatibility_keys = ("fps", "robot_type", "features", "data_path", "chunks_size")
    for root, (info, _episodes, _stats) in zip(roots[1:], metadata[1:], strict=True):
        mismatches = [key for key in compatibility_keys if info.get(key) != reference_info.get(key)]
        if mismatches:
            raise ValueError(f"{root}: incompatible metadata fields {mismatches}")
        if _read_jsonlines(root / "meta" / "tasks.jsonl") != reference_tasks:
            raise ValueError(f"{root}: task metadata differs from the first shard")

    total_episodes = sum(len(episodes) for _info, episodes, _stats in metadata)
    total_frames = sum(int(episode["length"]) for _info, episodes, _stats in metadata for episode in episodes)
    print(f"Merge plan: {len(roots)} shards, {total_episodes} episodes, {total_frames} frames -> {destination}")
    if args.test_mode:
        return
    if destination.exists() and not args.overwrite:
        raise FileExistsError(f"Destination exists: {destination}; pass --overwrite to replace it")

    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{destination.name}-merge-", dir=destination.parent) as temporary:
        staging = Path(temporary) / destination.name
        (staging / "meta").mkdir(parents=True)
        output_episodes: list[dict[str, Any]] = []
        output_stats: list[dict[str, Any]] = []
        output_manifest: list[dict[str, Any]] = []
        global_index = 0
        output_index = 0
        output_chunks_size = int(reference_info["chunks_size"])

        progress = tqdm(total=total_episodes, desc="Merging LeRobot v2.1 shards", unit="episode")
        for shard_index, (root, (info, episodes, stats)) in enumerate(zip(roots, metadata, strict=True)):
            stats_by_index = {int(item["episode_index"]): item for item in stats}
            manifest_path = root / manifest_name if manifest_name is not None else None
            source_manifest = None
            if manifest_path is not None and manifest_path.is_file():
                with manifest_path.open(encoding="utf-8") as file:
                    source_manifest = json.load(file)
            manifest_by_index = {int(item["episode_index"]): item for item in (source_manifest or [])}
            source_chunks_size = int(info["chunks_size"])
            for episode in episodes:
                source_index = int(episode["episode_index"])
                source_path = root / _dataset_path(info["data_path"], source_index, source_chunks_size)
                table = pq.read_table(source_path)
                length = int(episode["length"])
                if len(table) != length:
                    raise ValueError(f"{source_path}: {len(table)} rows, metadata says {length}")
                table = _rewrite_indices(table, output_index, global_index)
                output_path = staging / _dataset_path(reference_info["data_path"], output_index, output_chunks_size)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                pq.write_table(table, output_path)
                output_episodes.append({**episode, "episode_index": output_index})
                output_stats.append({**stats_by_index[source_index], "episode_index": output_index})
                source_entry = manifest_by_index.get(source_index, {})
                output_manifest.append(
                    {
                        **source_entry,
                        "episode_index": output_index,
                        "shard_index": shard_index,
                        "shard_episode_index": source_index,
                    }
                )
                global_index += length
                output_index += 1
                progress.update()
        progress.close()

        output_info = {
            **reference_info,
            "total_episodes": total_episodes,
            "total_frames": total_frames,
            "total_tasks": len(reference_tasks),
            "total_videos": 0,
            "total_chunks": (total_episodes + output_chunks_size - 1) // output_chunks_size,
            "splits": {"train": f"0:{total_episodes}"},
        }
        with (staging / "meta" / "info.json").open("w", encoding="utf-8") as file:
            json.dump(output_info, file, indent=4)
            file.write("\n")
        _write_jsonlines(staging / "meta" / "tasks.jsonl", reference_tasks)
        _write_jsonlines(staging / "meta" / "episodes.jsonl", output_episodes)
        _write_jsonlines(staging / "meta" / "episodes_stats.jsonl", output_stats)
        if manifest_name is not None:
            with (staging / manifest_name).open("w", encoding="utf-8") as file:
                json.dump(output_manifest, file, indent=2)
                file.write("\n")

        if destination.exists():
            shutil.rmtree(destination)
        shutil.move(staging, destination)
    print(f"Saved {total_episodes} episodes and {total_frames} frames to {destination}")


if __name__ == "__main__":
    main(tyro.cli(Args))

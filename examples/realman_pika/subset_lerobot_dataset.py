"""Create a v2.1 LeRobot dataset containing a selected number of episodes.

The source and destination both use the v2.1 one-Parquet-file-per-episode
layout. Episode and global frame indices are made contiguous in the subset.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
import random
import shutil
from typing import Any

from huggingface_hub import HfApi
from lerobot.utils.constants import HF_LEROBOT_HOME
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm.auto import tqdm
import tyro

LEROBOT_VERSION = "v2.1"


@dataclasses.dataclass
class Args:
    input: str
    repo_id: str
    num_episodes: int
    output_root: Path | None = None
    start_episode: int = 0
    random: bool = False
    seed: int = 0
    overwrite: bool = False
    dry_run: bool = False
    push_to_hub: bool = False


def _read_jsonlines(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def _write_jsonlines(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            json.dump(record, file, separators=(",", ":"))
            file.write("\n")


def _resolve_source(value: str) -> Path:
    candidate = Path(value).expanduser()
    root = candidate.resolve() if candidate.is_absolute() else (HF_LEROBOT_HOME / value).resolve()
    if not (root / "meta" / "info.json").is_file():
        raise FileNotFoundError(f"Not a local LeRobot dataset: {root}")
    return root


def _destination(args: Args) -> Path:
    if args.output_root is not None:
        return args.output_root.expanduser().resolve()
    return (HF_LEROBOT_HOME / args.repo_id).resolve()


def _select_episodes(
    episodes: list[dict[str, Any]],
    *,
    num_episodes: int,
    start_episode: int,
    use_random: bool,
    seed: int,
) -> list[dict[str, Any]]:
    if num_episodes < 1:
        raise ValueError("--num-episodes must be positive")
    if start_episode < 0:
        raise ValueError("--start-episode must be non-negative")
    ordered = sorted(episodes, key=lambda record: int(record["episode_index"]))
    candidates = ordered[start_episode:]
    if num_episodes > len(candidates):
        raise ValueError(
            f"Requested {num_episodes} episodes after start position {start_episode}, "
            f"but only {len(candidates)} are available"
        )
    if use_random:
        selected = random.Random(seed).sample(candidates, num_episodes)
        return sorted(selected, key=lambda record: int(record["episode_index"]))
    return candidates[:num_episodes]


def _episode_path(template: str, episode_index: int, chunks_size: int, **values: str) -> Path:
    return Path(
        template.format(
            episode_chunk=episode_index // chunks_size,
            episode_index=episode_index,
            **values,
        )
    )


def _replace_integer_column(table: pa.Table, name: str, values: range | list[int]) -> pa.Table:
    column_index = table.schema.get_field_index(name)
    if column_index < 0:
        raise ValueError(f"Episode Parquet is missing required column {name!r}")
    field = table.schema.field(column_index)
    return table.set_column(column_index, field, pa.array(values, type=field.type))


def _rewrite_episode(
    table: pa.Table,
    *,
    episode_index: int,
    global_from_index: int,
    video_paths: dict[str, str],
) -> pa.Table:
    length = len(table)
    table = _replace_integer_column(table, "episode_index", [episode_index] * length)
    table = _replace_integer_column(table, "frame_index", range(length))
    table = _replace_integer_column(table, "index", range(global_from_index, global_from_index + length))
    for video_key, relative_path in video_paths.items():
        column_index = table.schema.get_field_index(video_key)
        if column_index < 0:
            raise ValueError(f"Episode Parquet is missing video column {video_key!r}")
        field = table.schema.field(column_index)
        frames = table.column(column_index).to_pylist()
        for frame in frames:
            if frame is not None:
                frame["path"] = relative_path
        table = table.set_column(column_index, field, pa.array(frames, type=field.type))
    return table


def main(args: Args) -> None:
    source_root = _resolve_source(args.input)
    destination = _destination(args)
    if destination == source_root or destination in source_root.parents or source_root in destination.parents:
        raise ValueError("The source and destination dataset directories must not overlap")

    with (source_root / "meta" / "info.json").open(encoding="utf-8") as file:
        source_info = json.load(file)
    if source_info.get("codebase_version") != LEROBOT_VERSION:
        raise ValueError(
            f"Expected a LeRobot {LEROBOT_VERSION} input dataset, got {source_info.get('codebase_version')!r}"
        )

    episodes = _select_episodes(
        _read_jsonlines(source_root / "meta" / "episodes.jsonl"),
        num_episodes=args.num_episodes,
        start_episode=args.start_episode,
        use_random=args.random,
        seed=args.seed,
    )
    source_indices = [int(episode["episode_index"]) for episode in episodes]
    total_frames = sum(int(episode["length"]) for episode in episodes)
    print(f"Source: {source_root}")
    print(f"Selected source episodes: {source_indices}")
    print(f"Subset plan: {len(episodes)} episodes, {total_frames} frames -> {destination}")
    if args.dry_run:
        return

    if destination.exists():
        if not args.overwrite:
            raise FileExistsError(f"Destination exists: {destination}; pass --overwrite to replace it")
        shutil.rmtree(destination)
    (destination / "meta").mkdir(parents=True)

    chunks_size = int(source_info["chunks_size"])
    data_template = str(source_info["data_path"])
    video_template = str(source_info["video_path"])
    video_keys = [name for name, feature in source_info["features"].items() if feature["dtype"] == "video"]
    source_stats = {
        int(record["episode_index"]): record
        for record in _read_jsonlines(source_root / "meta" / "episodes_stats.jsonl")
    }
    output_episodes: list[dict[str, Any]] = []
    output_stats: list[dict[str, Any]] = []
    manifest: list[dict[str, int]] = []
    global_index = 0

    for output_index, episode in enumerate(tqdm(episodes, desc="Creating LeRobot v2.1 subset", unit="episode")):
        source_index = int(episode["episode_index"])
        length = int(episode["length"])
        source_data = source_root / _episode_path(data_template, source_index, chunks_size)
        output_data = destination / _episode_path(data_template, output_index, chunks_size)
        table = pq.read_table(source_data)
        if len(table) != length:
            raise ValueError(f"Episode {source_index} has {len(table)} rows, metadata says {length}")

        rewritten_video_paths: dict[str, str] = {}
        for video_key in video_keys:
            source_video = source_root / _episode_path(video_template, source_index, chunks_size, video_key=video_key)
            output_relative_video = _episode_path(video_template, output_index, chunks_size, video_key=video_key)
            output_video = destination / output_relative_video
            output_video.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_video, output_video)
            rewritten_video_paths[video_key] = output_relative_video.as_posix()

        table = _rewrite_episode(
            table,
            episode_index=output_index,
            global_from_index=global_index,
            video_paths=rewritten_video_paths,
        )
        output_data.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, output_data)
        output_episodes.append({**episode, "episode_index": output_index})
        stats = source_stats.get(source_index, {"stats": {}})
        output_stats.append({**stats, "episode_index": output_index})
        manifest.append({"episode_index": output_index, "source_episode_index": source_index, "length": length})
        global_index += length

    tasks = _read_jsonlines(source_root / "meta" / "tasks.jsonl")
    output_info = {
        **source_info,
        "codebase_version": LEROBOT_VERSION,
        "total_episodes": len(episodes),
        "total_frames": total_frames,
        "total_tasks": len(tasks),
        "total_videos": len(episodes) * len(video_keys),
        "total_chunks": (len(episodes) + chunks_size - 1) // chunks_size,
        "splits": {"train": f"0:{len(episodes)}"},
    }
    with (destination / "meta" / "info.json").open("w", encoding="utf-8") as file:
        json.dump(output_info, file, indent=4)
        file.write("\n")
    _write_jsonlines(destination / "meta" / "tasks.jsonl", tasks)
    _write_jsonlines(destination / "meta" / "episodes.jsonl", output_episodes)
    _write_jsonlines(destination / "meta" / "episodes_stats.jsonl", output_stats)
    with (destination / "subset_manifest.json").open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)
        file.write("\n")
    print(f"Saved {len(episodes)} episodes and {total_frames} frames to {destination} in v2.1 format")

    if args.push_to_hub:
        api = HfApi()
        api.create_repo(args.repo_id, repo_type="dataset", exist_ok=True)
        api.upload_folder(
            repo_id=args.repo_id,
            repo_type="dataset",
            folder_path=destination,
            commit_message=f"Upload {len(episodes)}-episode LeRobot v2.1 subset",
        )


if __name__ == "__main__":
    main(tyro.cli(Args))

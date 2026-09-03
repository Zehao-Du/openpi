"""Tests for the direct LeRobot v2.1 shard merger.

Run with:
uv run --project . pytest examples/realman_pika/merge_lerobot_v21_shards_test.py
"""

from __future__ import annotations

import json
from pathlib import Path

import merge_lerobot_v21_shards as merger
import pyarrow as pa
import pyarrow.parquet as pq


def _write_jsonlines(path: Path, records: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record) + "\n")


def _make_shard(
    root: Path,
    payload: bytes,
    planned_index: int,
    manifest_name: str = "pull_recolor_manifest.json",
) -> None:
    (root / "meta").mkdir(parents=True)
    data = root / "data/chunk-000/episode_000000.parquet"
    data.parent.mkdir(parents=True)
    pq.write_table(
        pa.table(
            {
                "episode_index": pa.array([0, 0], type=pa.int64()),
                "frame_index": pa.array([0, 1], type=pa.int64()),
                "index": pa.array([0, 1], type=pa.int64()),
                "image": [payload, payload],
            }
        ),
        data,
    )
    info = {
        "codebase_version": "v2.1",
        "fps": 10,
        "robot_type": "test",
        "total_episodes": 1,
        "total_frames": 2,
        "total_tasks": 1,
        "total_videos": 0,
        "total_chunks": 1,
        "chunks_size": 1000,
        "splits": {"train": "0:1"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {"image": {"dtype": "image", "shape": [1, 1, 3]}},
    }
    (root / "meta/info.json").write_text(json.dumps(info), encoding="utf-8")
    _write_jsonlines(root / "meta/tasks.jsonl", [{"task_index": 0, "task": "test"}])
    _write_jsonlines(root / "meta/episodes.jsonl", [{"episode_index": 0, "tasks": ["test"], "length": 2}])
    _write_jsonlines(root / "meta/episodes_stats.jsonl", [{"episode_index": 0, "stats": {}}])
    (root / manifest_name).write_text(
        json.dumps([{"episode_index": 0, "planned_episode_index": planned_index}]), encoding="utf-8"
    )


def test_merge_preserves_payload_and_makes_indices_contiguous(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    output = tmp_path / "merged"
    _make_shard(first, b"first-image-bytes", 10)
    _make_shard(second, b"second-image-bytes", 11)

    merger.main(
        merger.Args(
            inputs=(str(first), str(second)),
            repo_id="test/merged",
            output_root=output,
        )
    )

    first_table = pq.read_table(output / "data/chunk-000/episode_000000.parquet")
    second_table = pq.read_table(output / "data/chunk-000/episode_000001.parquet")
    assert first_table["image"].to_pylist() == [b"first-image-bytes"] * 2
    assert second_table["image"].to_pylist() == [b"second-image-bytes"] * 2
    assert second_table["episode_index"].to_pylist() == [1, 1]
    assert second_table["frame_index"].to_pylist() == [0, 1]
    assert second_table["index"].to_pylist() == [2, 3]
    manifest = json.loads((output / "pull_recolor_manifest.json").read_text(encoding="utf-8"))
    assert [entry["planned_episode_index"] for entry in manifest] == [10, 11]


def test_merge_preserves_keypoint_manifest_name(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    output = tmp_path / "merged"
    manifest_name = "pull_keypoint_manifest.json"
    _make_shard(first, b"first-image-bytes", 20, manifest_name)
    _make_shard(second, b"second-image-bytes", 21, manifest_name)

    merger.main(
        merger.Args(
            inputs=(str(first), str(second)),
            repo_id="test/merged",
            output_root=output,
        )
    )

    manifest = json.loads((output / manifest_name).read_text(encoding="utf-8"))
    assert [entry["planned_episode_index"] for entry in manifest] == [20, 21]
    assert not (output / "pull_recolor_manifest.json").exists()

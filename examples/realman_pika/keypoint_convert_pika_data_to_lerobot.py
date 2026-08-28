"""Split choumugun.mp4 into single-pull episodes in a LeRobot dataset.

The MP4 has no robot state or commanded actions, so this converter writes a
video-only dataset instead of inventing labels. LeRobot v3 may pack multiple
logical episodes into one physical video file; episode metadata preserves the
cuts. Default ranges were annotated for the supplied 3600-frame, 60 FPS video.
"""

from __future__ import annotations

from collections.abc import Sequence
import dataclasses
import json
from pathlib import Path
import shutil
import tempfile
from typing import Any

import cv2
from lerobot.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
from openpi_client import image_tools
from tqdm.auto import tqdm
import tyro


DEFAULT_VIDEO_PATH = Path(
    "/inspire/hdd/project/robot-dna/baojiachun-CZXS25130063/zehao/choumugun.mp4"
)
DEFAULT_OUTPUT_ROOT = Path("outputs/choumugun_keypoint_lerobot")
DEFAULT_REPO_ID = "Zehao123/choumugun_keypoint"
TASK_PROMPT = "pull one wooden stick from the pile"
IMAGE_SIZE = 224

# Complete approach -> grasp -> pull operations. Human reset frames are omitted.
DEFAULT_EPISODE_FRAME_RANGES: tuple[tuple[int, int], ...] = (
    (54, 276),
    (288, 492),
    (504, 762),
    (774, 1056),
    (1134, 1308),
    (1374, 1548),
    (1638, 1890),
    (1908, 2154),
    (2166, 2442),
    (2514, 2676),
    (2754, 2928),
    (3000, 3186),
    (3198, 3372),
    (3384, 3594),
)


@dataclasses.dataclass(frozen=True)
class EpisodeRange:
    start_frame: int
    end_frame: int

    @property
    def length(self) -> int:
        return self.end_frame - self.start_frame


@dataclasses.dataclass
class Args:
    video_path: Path = DEFAULT_VIDEO_PATH
    output_root: Path = DEFAULT_OUTPUT_ROOT
    repo_id: str = DEFAULT_REPO_ID
    keypoints_json: Path | None = None
    task_prompt: str = TASK_PROMPT
    image_size: int = IMAGE_SIZE
    max_episodes: int | None = None
    overwrite: bool = False
    dry_run: bool = False


def _open_video(path: Path) -> tuple[cv2.VideoCapture, float, int, int, int]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        raise ValueError(f"Could not open video: {path}")
    metadata = (
        float(capture.get(cv2.CAP_PROP_FPS)),
        int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
        int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )
    fps, count, width, height = metadata
    if not np.isfinite(fps) or fps <= 0 or min(count, width, height) <= 0:
        capture.release()
        raise ValueError(f"Invalid video metadata: fps={fps}, frames={count}, size={width}x{height}")
    return capture, *metadata


def _range_from_record(record: Any, fps: float, index: int) -> EpisodeRange:
    if not isinstance(record, dict):
        raise TypeError(f"Keypoint record {index} must be a JSON object")
    frame_keys = {"start_frame", "end_frame"}
    second_keys = {"start_seconds", "end_seconds"}
    has_frames = bool(frame_keys & record.keys())
    has_seconds = bool(second_keys & record.keys())
    if has_frames == has_seconds:
        raise ValueError(f"Keypoint record {index} must use exactly one coordinate type")
    keys = frame_keys if has_frames else second_keys
    if not keys <= record.keys():
        raise ValueError(f"Keypoint record {index} has an incomplete start/end pair")
    if has_frames:
        start, end = record["start_frame"], record["end_frame"]
        if any(isinstance(value, bool) or not isinstance(value, int) for value in (start, end)):
            raise TypeError(f"Keypoint record {index} frame values must be integers")
        return EpisodeRange(start, end)
    start, end = float(record["start_seconds"]), float(record["end_seconds"])
    if not np.isfinite([start, end]).all():
        raise ValueError(f"Keypoint record {index} seconds must be finite")
    return EpisodeRange(round(start * fps), round(end * fps))


def _load_ranges(path: Path | None, fps: float) -> list[EpisodeRange]:
    if path is None:
        return [EpisodeRange(*values) for values in DEFAULT_EPISODE_FRAME_RANGES]
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8") as file:
        records = json.load(file)
    if not isinstance(records, list) or not records:
        raise ValueError(f"{path} must contain a non-empty JSON list")
    return [_range_from_record(record, fps, index) for index, record in enumerate(records)]


def _validate_ranges(ranges: Sequence[EpisodeRange], frame_count: int) -> None:
    if not ranges:
        raise ValueError("At least one episode range is required")
    previous_end = 0
    for index, item in enumerate(ranges):
        if not 0 <= item.start_frame < item.end_frame <= frame_count:
            raise ValueError(
                f"Episode {index} range [{item.start_frame}, {item.end_frame}) "
                f"is outside a {frame_count}-frame video"
            )
        if index and item.start_frame < previous_end:
            raise ValueError(f"Episode {index} overlaps or is out of order")
        previous_end = item.end_frame


def _create_dataset(repo_id: str, root: Path, fps: int, image_size: int) -> LeRobotDataset:
    return LeRobotDataset.create(
        repo_id=repo_id,
        root=root,
        robot_type="rendered realman pika demonstration",
        fps=fps,
        features={
            "image": {
                "dtype": "video",
                "shape": (image_size, image_size, 3),
                "names": ["height", "width", "channel"],
            }
        },
        use_videos=True,
        image_writer_threads=8,
        image_writer_processes=0,
    )


def _manifest(path: Path, fps: float, ranges: Sequence[EpisodeRange], task: str) -> dict[str, Any]:
    return {
        "source_video": str(path),
        "source_fps": fps,
        "interval_convention": "half-open [start_frame, end_frame)",
        "episodes": [
            {
                "episode_index": index,
                "start_frame": item.start_frame,
                "end_frame": item.end_frame,
                "start_seconds": item.start_frame / fps,
                "end_seconds": item.end_frame / fps,
                "frame_count": item.length,
                "task": task,
            }
            for index, item in enumerate(ranges)
        ],
    }


def main(args: Args) -> None:
    video_path = args.video_path.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    if not video_path.is_file():
        raise FileNotFoundError(video_path)
    if not args.repo_id.strip() or not args.task_prompt.strip():
        raise ValueError("--repo-id and --task-prompt must not be empty")
    if args.image_size < 1 or (args.max_episodes is not None and args.max_episodes < 1):
        raise ValueError("--image-size and --max-episodes must be positive")

    capture, fps, frame_count, width, height = _open_video(video_path)
    capture.release()
    ranges = _load_ranges(args.keypoints_json, fps)
    _validate_ranges(ranges, frame_count)
    ranges = ranges[: args.max_episodes]
    output_fps = round(fps)
    if not np.isclose(fps, output_fps, atol=1e-6):
        raise ValueError(f"LeRobot output requires integral FPS, got {fps}")

    print(f"Source: {video_path} ({width}x{height}, {fps:g} FPS, {frame_count} frames)")
    for index, item in enumerate(ranges):
        print(
            f"episode{index}: [{item.start_frame}, {item.end_frame}) = "
            f"[{item.start_frame / fps:.3f}, {item.end_frame / fps:.3f}) s"
        )
    if args.dry_run:
        return
    if output_root.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists: {output_root}; pass --overwrite to replace it")
    output_root.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix=f".{output_root.name}-", dir=output_root.parent) as temp:
        staging = Path(temp) / "dataset"
        dataset = _create_dataset(args.repo_id, staging, output_fps, args.image_size)
        capture, *_ = _open_video(video_path)
        try:
            for episode_index, item in enumerate(tqdm(ranges, desc="Converting", unit="episode")):
                if not capture.set(cv2.CAP_PROP_POS_FRAMES, item.start_frame):
                    raise RuntimeError(f"Could not seek to frame {item.start_frame}")
                for frame_index in tqdm(range(item.start_frame, item.end_frame), leave=False):
                    ok, bgr = capture.read()
                    if not ok:
                        raise RuntimeError(f"Failed to decode frame {frame_index}")
                    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    image = image_tools.resize_with_pad(rgb, args.image_size, args.image_size)
                    dataset.add_frame({"image": np.asarray(image, dtype=np.uint8), "task": args.task_prompt})
                dataset.save_episode()
            dataset.finalize()
        finally:
            capture.release()

        with (staging / "keypoint_manifest.json").open("w", encoding="utf-8") as file:
            json.dump(_manifest(video_path, fps, ranges, args.task_prompt), file, indent=2)
            file.write("\n")
        if output_root.exists():
            shutil.rmtree(output_root)
        shutil.move(staging, output_root)
    print(f"Saved {len(ranges)} episodes to {output_root}")
    print(f"LeRobot video files: {output_root / 'videos'}")


if __name__ == "__main__":
    main(tyro.cli(Args))

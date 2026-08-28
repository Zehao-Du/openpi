"""Convert a LeRobot v2.1 image dataset into a video-backed dataset.

The source dataset is left untouched. By default, the output is written next
to it with ``_video`` appended to the directory name. For every episode and
camera this script:

1. encodes the embedded images as an MP4 file;
2. replaces the Parquet ``Image`` column with ``VideoFrame(path, timestamp)``;
3. updates ``meta/info.json`` to describe the video-backed dataset.

Example:

    uv run --project examples/realman_pika python \
        examples/realman_pika/convert_lerobot_images_to_videos.py \
        --dataset-root \
        /inspire/hdd/project/robot-dna/baojiachun-CZXS25130063/zehao/dataset/lerobot/Zehao123/pika_collect_blocks_224_224_visualprompt_new_0824

Resume an interrupted conversion:

    uv run --project examples/realman_pika python \
        examples/realman_pika/convert_lerobot_images_to_videos.py \
        --dataset-root /absolute/path/to/dataset --resume
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed
import dataclasses
from io import BytesIO
import json
import os
from pathlib import Path
import shutil
from typing import Any

import av
from PIL import Image
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm.auto import tqdm
import tyro

DEFAULT_DATASET_ROOT = Path(
    "/inspire/hdd/project/robot-dna/baojiachun-CZXS25130063/zehao/dataset/lerobot/"
    "Zehao123/pika_collect_blocks_224_224_visualprompt_new_0824"
)
CONVERSION_MARKER = ".image_to_video_conversion.json"


@dataclasses.dataclass
class Args:
    dataset_root: Path = DEFAULT_DATASET_ROOT
    output_root: Path | None = None
    camera_keys: tuple[str, ...] | None = None
    workers: int = 2
    codec: str = "libx264"
    pixel_format: str = "yuv420p"
    crf: int = 23
    preset: str = "medium"
    resume: bool = False
    overwrite: bool = False


def _resolve_source(path: Path) -> Path:
    source = path.expanduser().resolve()
    if not source.is_dir():
        raise NotADirectoryError(source)
    info_path = source / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(info_path)
    return source


def _default_output_root(source: Path) -> Path:
    return source.with_name(f"{source.name}_video")


def _load_info(root: Path) -> dict[str, Any]:
    with (root / "meta" / "info.json").open(encoding="utf-8") as file:
        return json.load(file)


def _select_camera_keys(info: dict[str, Any], requested: tuple[str, ...] | None) -> tuple[str, ...]:
    features = info.get("features", {})
    image_keys = tuple(key for key, feature in features.items() if feature.get("dtype") == "image")
    if requested is None:
        if not image_keys:
            raise ValueError("The source info.json has no features with dtype='image'")
        return image_keys

    if not requested:
        raise ValueError("--camera-keys must not be empty")
    missing = [key for key in requested if key not in features]
    if missing:
        raise KeyError(f"Unknown camera keys: {missing}")
    not_images = [key for key in requested if features[key].get("dtype") != "image"]
    if not_images:
        raise ValueError(f"Camera keys are not image features: {not_images}")
    return tuple(dict.fromkeys(requested))


def _episode_index(path: Path) -> int:
    prefix = "episode_"
    if not path.stem.startswith(prefix) or not path.stem[len(prefix) :].isdigit():
        raise ValueError(f"Unexpected episode Parquet filename: {path.name}")
    return int(path.stem[len(prefix) :])


def _relative_video_path(info: dict[str, Any], episode_index: int, camera_key: str) -> Path:
    chunk_size = int(info["chunks_size"])
    template = info["video_path"]
    return Path(
        template.format(
            episode_chunk=episode_index // chunk_size,
            episode_index=episode_index,
            video_key=camera_key,
        )
    )


def _image_from_record(record: dict[str, Any] | None, source_root: Path) -> Image.Image:
    if record is None:
        raise ValueError("Encountered a null image record")
    image_bytes = record.get("bytes")
    image_path = record.get("path")
    if image_bytes is not None:
        image = Image.open(BytesIO(image_bytes))
    elif image_path:
        path = Path(image_path)
        image = Image.open(path if path.is_absolute() else source_root / path)
    else:
        raise ValueError("Image record contains neither bytes nor path")
    return image.convert("RGB")


def _encode_video(
    records: list[dict[str, Any] | None],
    source_root: Path,
    output_path: Path,
    fps: int,
    *,
    codec: str,
    pixel_format: str,
    crf: int,
    preset: str,
) -> None:
    if not records:
        raise ValueError(f"Cannot encode an empty video: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.stem}.tmp{output_path.suffix}")
    if temporary_path.exists():
        temporary_path.unlink()

    first_image = _image_from_record(records[0], source_root)
    width, height = first_image.size
    if pixel_format == "yuv420p" and (width % 2 or height % 2):
        raise ValueError(f"yuv420p requires even dimensions, got {width}x{height}")

    try:
        with av.open(str(temporary_path), mode="w") as container:
            stream = container.add_stream(
                codec,
                rate=fps,
                options={"crf": str(crf), "preset": preset},
            )
            stream.width = width
            stream.height = height
            stream.pix_fmt = pixel_format
            for frame_index, record in enumerate(records):
                image = first_image if frame_index == 0 else _image_from_record(record, source_root)
                if image.size != (width, height):
                    raise ValueError(f"Frame {frame_index} has size {image.size}, expected {(width, height)}")
                frame = av.VideoFrame.from_image(image)
                for packet in stream.encode(frame):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
        if not temporary_path.is_file() or temporary_path.stat().st_size == 0:
            raise RuntimeError(f"Video encoder produced no output: {temporary_path}")
        os.replace(temporary_path, output_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _video_column(relative_path: Path, timestamps: pa.ChunkedArray) -> pa.StructArray:
    timestamp_values = timestamps.combine_chunks().cast(pa.float32())
    paths = pa.array([relative_path.as_posix()] * len(timestamp_values), type=pa.string())
    return pa.StructArray.from_arrays(
        (paths, timestamp_values),
        fields=(pa.field("path", pa.string()), pa.field("timestamp", pa.float32())),
    )


def _with_video_huggingface_metadata(table: pa.Table, camera_keys: tuple[str, ...]) -> pa.Table:
    metadata = dict(table.schema.metadata or {})
    serialized = metadata.get(b"huggingface")
    if serialized is None:
        raise ValueError("Parquet schema is missing Hugging Face feature metadata")
    huggingface_metadata = json.loads(serialized)
    feature_metadata = huggingface_metadata["info"]["features"]
    for camera_key in camera_keys:
        feature_metadata[camera_key] = {"_type": "VideoFrame"}
    metadata[b"huggingface"] = json.dumps(huggingface_metadata, separators=(",", ":")).encode()
    return table.replace_schema_metadata(metadata)


def _output_episode_is_complete(
    output_parquet: Path,
    output_root: Path,
    info: dict[str, Any],
    episode_index: int,
    camera_keys: tuple[str, ...],
) -> bool:
    if not output_parquet.is_file():
        return False
    schema = pq.ParquetFile(output_parquet).schema_arrow
    for camera_key in camera_keys:
        expected_type = pa.struct({"path": pa.string(), "timestamp": pa.float32()})
        if schema.field(camera_key).type != expected_type:
            return False
        if not (output_root / _relative_video_path(info, episode_index, camera_key)).is_file():
            return False
    return True


def _convert_episode(
    source_parquet: Path,
    source_root: Path,
    output_root: Path,
    info: dict[str, Any],
    camera_keys: tuple[str, ...],
    args: Args,
) -> tuple[int, int, bool]:
    episode_index = _episode_index(source_parquet)
    relative_parquet = source_parquet.relative_to(source_root)
    output_parquet = output_root / relative_parquet
    if args.resume and _output_episode_is_complete(output_parquet, output_root, info, episode_index, camera_keys):
        return episode_index, pq.ParquetFile(output_parquet).metadata.num_rows, True

    table = pq.read_table(source_parquet)
    if "timestamp" not in table.column_names:
        raise KeyError(f"{source_parquet}: missing timestamp column")
    frame_count = len(table)
    if frame_count == 0:
        raise ValueError(f"{source_parquet}: episode is empty")

    for camera_key in camera_keys:
        if camera_key not in table.column_names:
            raise KeyError(f"{source_parquet}: missing camera column {camera_key!r}")
        records = table[camera_key].combine_chunks().to_pylist()
        relative_video = _relative_video_path(info, episode_index, camera_key)
        video_path = output_root / relative_video
        if not (args.resume and video_path.is_file() and video_path.stat().st_size > 0):
            _encode_video(
                records,
                source_root,
                video_path,
                int(info["fps"]),
                codec=args.codec,
                pixel_format=args.pixel_format,
                crf=args.crf,
                preset=args.preset,
            )
        column_index = table.column_names.index(camera_key)
        table = table.set_column(
            column_index,
            camera_key,
            _video_column(relative_video, table["timestamp"]),
        )

    table = _with_video_huggingface_metadata(table, camera_keys)
    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    temporary_parquet = output_parquet.with_name(f".{output_parquet.name}.tmp")
    try:
        pq.write_table(table, temporary_parquet, compression="zstd")
        os.replace(temporary_parquet, output_parquet)
    finally:
        if temporary_parquet.exists():
            temporary_parquet.unlink()
    return episode_index, frame_count, False


def _prepare_output(
    source: Path,
    output: Path,
    camera_keys: tuple[str, ...],
    args: Args,
) -> None:
    if source == output:
        raise ValueError("--output-root must differ from --dataset-root; in-place conversion is not supported")
    marker = output / CONVERSION_MARKER
    expected_marker = {
        "source_root": str(source),
        "camera_keys": list(camera_keys),
    }

    if output.exists():
        if args.overwrite:
            shutil.rmtree(output)
        elif args.resume:
            if not marker.is_file():
                raise ValueError(f"Refusing to resume: conversion marker is missing from {output}")
            with marker.open(encoding="utf-8") as file:
                actual_marker = json.load(file)
            if actual_marker != expected_marker:
                raise ValueError(f"Conversion marker does not match this request: {actual_marker} != {expected_marker}")
            return
        else:
            raise FileExistsError(f"Output already exists: {output}. Use --resume or --overwrite.")

    shutil.copytree(source, output, ignore=shutil.ignore_patterns("data", "videos"))
    with marker.open("w", encoding="utf-8") as file:
        json.dump(expected_marker, file, indent=2)
        file.write("\n")


def _finalize_info(
    output_root: Path,
    info: dict[str, Any],
    camera_keys: tuple[str, ...],
    episode_count: int,
) -> None:
    output_info = json.loads(json.dumps(info))
    for camera_key in camera_keys:
        output_info["features"][camera_key]["dtype"] = "video"
    output_info["total_videos"] = episode_count * len(camera_keys)
    info_path = output_root / "meta" / "info.json"
    temporary_path = info_path.with_name(".info.json.tmp")
    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(output_info, file, indent=4)
        file.write("\n")
    os.replace(temporary_path, info_path)


def main(args: Args) -> None:
    if args.workers < 1:
        raise ValueError(f"--workers must be at least 1, got {args.workers}")
    if not 0 <= args.crf <= 51:
        raise ValueError(f"--crf must be between 0 and 51, got {args.crf}")
    if args.resume and args.overwrite:
        raise ValueError("--resume and --overwrite are mutually exclusive")

    source = _resolve_source(args.dataset_root)
    output = args.output_root.expanduser().resolve() if args.output_root is not None else _default_output_root(source)
    info = _load_info(source)
    if info.get("codebase_version") != "v2.1":
        raise ValueError(f"Expected a LeRobot v2.1 dataset, got {info.get('codebase_version')!r}")
    camera_keys = _select_camera_keys(info, args.camera_keys)
    source_parquets = sorted((source / "data").glob("chunk-*/episode_*.parquet"))
    if not source_parquets:
        raise FileNotFoundError(f"No data/chunk-*/episode_*.parquet files found under {source}")
    if len(source_parquets) != int(info["total_episodes"]):
        raise ValueError(f"Found {len(source_parquets)} episode files, info.json says {info['total_episodes']}")

    _prepare_output(source, output, camera_keys, args)
    print(f"Source: {source}")
    print(f"Output: {output}")
    print(f"Cameras: {', '.join(camera_keys)}; codec={args.codec}; fps={info['fps']}")

    total_frames = 0
    skipped = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                _convert_episode,
                parquet,
                source,
                output,
                info,
                camera_keys,
                args,
            ): parquet
            for parquet in source_parquets
        }
        with tqdm(total=len(futures), desc="Encoding episodes", unit="episode") as progress:
            for future in as_completed(futures):
                episode_index, frame_count, was_skipped = future.result()
                total_frames += frame_count
                skipped += int(was_skipped)
                progress.set_postfix(episode=episode_index, frames=total_frames, resumed=skipped)
                progress.update(1)

    if total_frames != int(info["total_frames"]):
        raise ValueError(f"Converted {total_frames} frames, info.json says {info['total_frames']}")
    _finalize_info(output, info, camera_keys, len(source_parquets))
    (output / CONVERSION_MARKER).unlink()
    print(
        f"Saved video-backed LeRobot dataset to {output}: "
        f"{len(source_parquets)} episodes, {total_frames} frames, "
        f"{len(source_parquets) * len(camera_keys)} videos"
    )


if __name__ == "__main__":
    main(tyro.cli(Args))

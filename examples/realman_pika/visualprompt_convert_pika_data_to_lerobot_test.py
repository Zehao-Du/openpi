from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from PIL import Image
import pytest
import visualprompt_convert_pika_data_to_lerobot as converter


class _FakePreprocessor:
    def __init__(self) -> None:
        self.calls: list[dict[str, np.ndarray]] = []

    def preprocess(self, images: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        self.calls.append(images)
        return {key: np.clip(image.astype(np.int16) + 1, 0, 255).astype(np.uint8) for key, image in images.items()}


class _BadShapePreprocessor:
    def preprocess(self, images: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        return {key: np.zeros((10, 10, 3), dtype=np.uint8) for key in images}


class _FakeSequentialPreprocessor(_FakePreprocessor):
    requires_sequential_frames = True


class _FakeBidirectionalPreprocessor(_FakeSequentialPreprocessor):
    def __init__(self) -> None:
        super().__init__()
        self.sequences: list[list[int]] = []

    def start_episode(self) -> None:
        self.sequences.append([])

    def has_active_trackers(self, camera_names: tuple[str, ...]) -> bool:
        return set(camera_names) == {"image", "wrist_image"}

    def preprocess(self, images: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        self.sequences[-1].append(int(images["image"][0, 0, 0]))
        return super().preprocess(images)


class _FakeRetryingBidirectionalPreprocessor(_FakeBidirectionalPreprocessor):
    def __init__(self, successful_source_frames: set[int]) -> None:
        super().__init__()
        self.successful_source_frames = successful_source_frames
        self._active = False

    def start_episode(self) -> None:
        super().start_episode()
        self._active = False

    def has_active_trackers(self, camera_names: tuple[str, ...]) -> bool:
        return super().has_active_trackers(camera_names) and self._active

    def preprocess(self, images: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        if not self.sequences[-1]:
            self._active = int(images["image"][0, 0, 0]) in self.successful_source_frames
        return super().preprocess(images)


class _FakeDataset:
    def __init__(self) -> None:
        self.frames: list[dict[str, Any]] = []

    def add_frame(self, frame: dict[str, Any]) -> None:
        self.frames.append(frame)


class _FakeVideoWriter:
    def __init__(self) -> None:
        self.frames: list[np.ndarray] = []

    def add_frame(self, image: np.ndarray) -> None:
        self.frames.append(image)


def _fake_file(frame_count: int) -> dict[str, list[int]]:
    return {
        converter.FISHEYE_KEY: list(range(frame_count)),
        converter.DEPTH_CAMERA_RGB_KEY: list(range(100, 100 + frame_count)),
    }


def _fake_read_rgb(episode_dir: Path, value: object) -> np.ndarray:
    del episode_dir
    return np.full((converter.IMAGE_SIZE, converter.IMAGE_SIZE, 3), int(value), dtype=np.uint8)


def test_episode_conversion_batches_both_cameras_and_keeps_frame_order(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(converter, "_read_rgb", _fake_read_rgb)
    state = np.arange(9 * 7, dtype=np.float32).reshape(9, 7)
    dataset = _FakeDataset()
    preprocessor = _FakePreprocessor()
    video_writer = _FakeVideoWriter()

    converter._write_episode_frames(  # noqa: SLF001
        dataset,
        Path("/unused/episode0"),
        _fake_file(20),  # type: ignore[arg-type]
        state,
        2,
        preprocessor,
        sam_batch_size=8,
        task_prompt="grasp the green block and place it into the drawer",
        output_name="episode4",
        preview_writer=video_writer,  # type: ignore[arg-type]
    )

    assert [len(call) for call in preprocessor.calls] == [16, 2]
    assert len(dataset.frames) == len(state)
    for frame_index, frame in enumerate(dataset.frames):
        assert (frame["image"] == frame_index + 3).all()
        assert (frame["wrist_image"] == 103 + frame_index).all()
        np.testing.assert_array_equal(frame["state"], state[frame_index])
        np.testing.assert_array_equal(frame["actions"], state[frame_index])
        assert frame["actions"] is not frame["state"]
        assert frame["task"] == "grasp the green block and place it into the drawer"

    assert len(video_writer.frames) == len(state)
    assert video_writer.frames[0].shape == (converter.IMAGE_SIZE * 2, converter.IMAGE_SIZE * 2, 3)
    assert (video_writer.frames[0][100, 100] == 2).all()
    assert (video_writer.frames[0][100, converter.IMAGE_SIZE + 100] == 3).all()
    assert (video_writer.frames[0][converter.IMAGE_SIZE + 100, 100] == 102).all()
    assert (video_writer.frames[0][converter.IMAGE_SIZE + 100, converter.IMAGE_SIZE + 100] == 103).all()


def test_grasp_detection_anchor_uses_lowest_z_before_close() -> None:
    state = np.zeros((5, 7), dtype=np.float32)
    state[:, 2] = (0.8, 0.2, 0.5, 0.1, 0.0)

    assert converter._grasp_detection_anchor_index(state, grasp_close_local_index=2) == 1  # noqa: SLF001


def test_detection_candidates_expand_from_preferred_anchor() -> None:
    assert converter._detection_candidate_local_indices(5, 2) == (2, 1, 3, 0, 4)  # noqa: SLF001


def test_episode_tracking_starts_at_anchor_and_tracks_both_directions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(converter, "_read_rgb", _fake_read_rgb)
    state = np.zeros((5, 7), dtype=np.float32)
    dataset = _FakeDataset()
    preprocessor = _FakeBidirectionalPreprocessor()

    written = converter._write_episode_frames(  # noqa: SLF001
        dataset,
        Path("/unused/episode0"),
        _fake_file(20),  # type: ignore[arg-type]
        state,
        source_start=2,
        image_preprocessor=preprocessor,
        sam_batch_size=8,
        task_prompt="grasp the blue block and place it into the drawer",
        output_name="episode0",
        detection_anchor_local_index=2,
    )

    # Source frame 4 is detected twice: once for forward tracking and once to
    # initialize the independent backward tracker.
    assert preprocessor.sequences == [[4, 5, 6], [4, 3, 2]]
    assert written is True
    assert len(dataset.frames) == 5
    assert [int(frame["image"][0, 0, 0]) for frame in dataset.frames] == [3, 4, 5, 6, 7]
    assert all(
        frame["task"] == "grasp the blue block and place it into the drawer"
        for frame in dataset.frames
    )


def test_episode_tracking_retries_an_adjacent_frame_when_anchor_detection_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(converter, "_read_rgb", _fake_read_rgb)
    state = np.zeros((5, 7), dtype=np.float32)
    dataset = _FakeDataset()
    preprocessor = _FakeRetryingBidirectionalPreprocessor(successful_source_frames={3})

    written = converter._write_episode_frames(  # noqa: SLF001
        dataset,
        Path("/unused/episode0"),
        _fake_file(20),  # type: ignore[arg-type]
        state,
        source_start=2,
        image_preprocessor=preprocessor,
        sam_batch_size=8,
        task_prompt="grasp the blue block and place it into the drawer",
        output_name="episode0",
        detection_anchor_local_index=2,
    )

    assert written is True
    assert preprocessor.sequences == [[4], [3, 4, 5, 6], [3, 2]]
    assert [int(frame["image"][0, 0, 0]) for frame in dataset.frames] == [3, 4, 5, 6, 7]


def test_episode_is_not_written_when_all_detection_candidates_fail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(converter, "_read_rgb", _fake_read_rgb)
    state = np.zeros((5, 7), dtype=np.float32)
    dataset = _FakeDataset()
    preprocessor = _FakeRetryingBidirectionalPreprocessor(successful_source_frames=set())

    written = converter._write_episode_frames(  # noqa: SLF001
        dataset,
        Path("/unused/episode0"),
        _fake_file(20),  # type: ignore[arg-type]
        state,
        source_start=2,
        image_preprocessor=preprocessor,
        sam_batch_size=8,
        task_prompt="grasp the blue block and place it into the drawer",
        output_name="episode0",
        detection_anchor_local_index=2,
    )

    assert written is False
    assert preprocessor.sequences == [[4], [3], [5], [2], [6]]
    assert dataset.frames == []


def test_preview_video_writer_encodes_mp4(tmp_path: Path) -> None:
    import av

    output_path = tmp_path / "preview.mp4"
    writer = converter._PreviewVideoWriter(output_path, converter.FPS)  # noqa: SLF001
    writer.add_frame(np.zeros((converter.IMAGE_SIZE * 2, converter.IMAGE_SIZE * 2, 3), dtype=np.uint8))
    writer.add_frame(np.full((converter.IMAGE_SIZE * 2, converter.IMAGE_SIZE * 2, 3), 255, dtype=np.uint8))
    writer.close()

    assert output_path.stat().st_size > 0
    with av.open(str(output_path)) as container:
        assert sum(1 for _ in container.decode(video=0)) == 2


@pytest.mark.parametrize(
    ("response", "expected"),
    [("y", True), ("YES", True), ("n", False), ("", False)],
)
def test_existing_preview_video_asks_before_overwrite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    response: str,
    expected: object,
) -> None:
    preview_path = tmp_path / "preview.mp4"
    preview_path.write_bytes(b"existing video")
    monkeypatch.setattr("builtins.input", lambda _: response)

    assert converter._confirm_preview_video_overwrite(preview_path) is expected  # noqa: SLF001
    assert preview_path.read_bytes() == b"existing video"


def test_preview_video_overwrite_rejects_directory(tmp_path: Path) -> None:
    preview_directory = tmp_path / "preview.mp4"
    preview_directory.mkdir()

    with pytest.raises(IsADirectoryError, match="not a file"):
        converter._confirm_preview_video_overwrite(preview_directory)  # noqa: SLF001


def test_preprocess_batch_rejects_invalid_image_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(converter, "_read_rgb", _fake_read_rgb)

    with pytest.raises(ValueError, match="expected .*uint8"):
        converter._preprocess_frame_batch(  # noqa: SLF001
            Path("/unused/episode0"),
            _fake_file(1),  # type: ignore[arg-type]
            range(1),
            _BadShapePreprocessor(),
        )


def test_tracking_preprocessor_receives_frames_sequentially(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(converter, "_read_rgb", _fake_read_rgb)
    preprocessor = _FakeSequentialPreprocessor()

    originals, processed = converter._preprocess_frame_batch(  # noqa: SLF001
        Path("/unused/episode0"),
        _fake_file(3),  # type: ignore[arg-type]
        range(3),
        preprocessor,
    )

    assert len(preprocessor.calls) == 3
    assert all(set(call) == {"image", "wrist_image"} for call in preprocessor.calls)
    assert set(processed) == set(originals)
    assert (processed["2:image"] == 3).all()
    assert (processed["2:wrist_image"] == 103).all()


@pytest.mark.parametrize(
    ("color", "rgb"),
    [
        ("red", (250, 20, 15)),
        ("green", (20, 245, 40)),
        ("blue", (10, 130, 250)),
        ("pink", (250, 20, 105)),
    ],
)
def test_classify_grasp_color_from_preclose_center_region(
    color: str,
    rgb: tuple[int, int, int],
) -> None:
    images = []
    for _ in range(3):
        image = np.full((480, 640, 3), 90, dtype=np.uint8)
        image[270:440, 250:440] = rgb
        image[10:100, 10:100] = (0, 0, 255)  # Distractor outside the grasp ROI.
        images.append(image)

    classification = converter._classify_grasp_color(  # noqa: SLF001
        images,
        (50, 55, 60),
        converter.ColorDetectionConfig(),
    )

    assert classification.color == color
    assert classification.confidence > 0.95
    assert classification.reference_frames == (50, 55, 60)


def test_classify_grasp_color_rejects_ambiguous_target() -> None:
    image = np.full((480, 640, 3), 90, dtype=np.uint8)
    image[260:440, 220:320] = (250, 20, 15)
    image[260:440, 320:420] = (10, 130, 250)

    with pytest.raises(ValueError, match="Ambiguous"):
        converter._classify_grasp_color(  # noqa: SLF001
            [image],
            [50],
            converter.ColorDetectionConfig(min_confidence=0.75),
        )


def test_color_prompt_templates_are_filled_per_episode() -> None:
    assert converter._format_color_prompts(  # noqa: SLF001
        ("{color} block", "{color} cube"),
        "green",
    ) == ("green block", "green cube")


@pytest.mark.parametrize(
    ("target_rgb", "expected_color"),
    [
        ((0, 0, 255), "blue"),
        ((255, 0, 0), "red"),
        ((255, 0, 128), "pink"),
        ((255, 255, 255), "white"),
    ],
)
def test_policy_color_name_is_derived_from_recolor_target(
    target_rgb: tuple[int, int, int], expected_color: str
) -> None:
    assert converter._target_rgb_color_name(target_rgb) == expected_color  # noqa: SLF001


def test_manifest_uses_recolor_target_for_policy_prompt(tmp_path: Path) -> None:
    episode_slice = converter.EpisodeSlice(
        output_index=0,
        source_episode_dir=tmp_path / "episode7",
        cycle=converter.GraspCycle(start=1, close=2, release=3, end=4),
    )
    classification = converter.GraspColorClassification(
        color="red",
        confidence=0.9,
        colored_fraction=0.1,
        reference_frames=(1,),
        scores={"red": 0.9},
    )

    manifest = converter._make_manifest(  # noqa: SLF001
        [episode_slice],
        {0: classification},
        {0: ("red block",)},
        target_rgb=(0, 0, 255),
        target_color="blue",
    )

    assert manifest[0]["color"] == "red"
    assert manifest[0]["sam3_prompts"] == ["red block"]
    assert manifest[0]["recolor_target_rgb"] == [0, 0, 255]
    assert manifest[0]["task_prompt"] == "grasp the blue block and place it into the drawer"


def test_offline_sam3_defaults_to_cuda() -> None:
    assert converter.Sam3Config().device == "cuda"
    assert converter.Sam3Config().fisheye_score_threshold == 0.4
    assert converter.Sam3Config().cross_camera_mapping == converter.DEFAULT_CROSS_CAMERA_MAPPING


def test_offline_preprocessor_uses_fisheye_specific_score_threshold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    class _FakeTracker:
        def __init__(self, checkpoint: Path, **kwargs: Any) -> None:
            captured["checkpoint"] = checkpoint
            captured.update(kwargs)

    monkeypatch.setattr(converter, "Sam3EpisodeTrackerPreprocessor", _FakeTracker)
    config = converter.Sam3Config(score_threshold=0.5, fisheye_score_threshold=0.4)

    converter._make_preprocessor(config, ("red block",))  # noqa: SLF001

    assert captured["score_threshold"] == 0.5
    assert captured["camera_score_thresholds"] == {"image": 0.4}
    assert captured["cross_camera_mapping"] == config.cross_camera_mapping
    assert captured["mapping_source_camera"] == "wrist_image"
    assert captured["mapping_destination_camera"] == "image"
    assert captured["redetect_area_ratio"] == config.redetect_area_ratio
    assert captured["redetect_reference_decay"] == config.redetect_reference_decay
    assert captured["redetect_cooldown_frames"] == config.redetect_cooldown_frames


def test_episode_color_uses_cycle_close_frame_and_realsense_image(tmp_path: Path) -> None:
    episode_dir = tmp_path / "episode0"
    image_dir = episode_dir / "camera/color/pikaDepthCamera"
    image_dir.mkdir(parents=True)
    string_dtype = h5py.string_dtype(encoding="utf-8")
    image_paths = []
    for frame_index in range(8):
        relative_path = Path("camera/color/pikaDepthCamera") / f"{frame_index}.png"
        rgb = (20, 245, 40) if frame_index == 5 else (250, 20, 15)
        Image.fromarray(np.full((100, 120, 3), rgb, dtype=np.uint8)).save(episode_dir / relative_path)
        image_paths.append(str(relative_path))
    with h5py.File(episode_dir / "data.hdf5", "w") as file:
        file.create_dataset(converter.DEPTH_CAMERA_RGB_KEY, data=image_paths, dtype=string_dtype)
    with h5py.File(episode_dir / "data.hdf5", "r") as file:
        classification = converter._classify_episode_grasp_color(  # noqa: SLF001
            episode_dir,
            file,
            converter.GraspCycle(start=0, close=6, release=7, end=8),
            converter.ColorDetectionConfig(reference_frame_offsets=(-1,)),
        )

    assert classification.color == "green"
    assert classification.reference_frames == (5,)


def _write_gripper_episode(path: Path, gripper: np.ndarray) -> Path:
    path.mkdir()
    with h5py.File(path / "data.hdf5", "w") as file:
        file.create_dataset(converter.GRIPPER_KEY, data=gripper)
    return path


def test_plan_episode_slices_flattens_source_episodes_with_contiguous_indices(tmp_path: Path) -> None:
    one_cycle = np.array([0.09] * 3 + [0.07] * 3 + [0.09] * 3)
    two_cycles = np.array([0.09] * 3 + [0.07] * 3 + [0.09] * 3 + [0.07] * 3 + [0.09] * 3)
    episodes = [
        _write_gripper_episode(tmp_path / "episode0", one_cycle),
        _write_gripper_episode(tmp_path / "episode1", two_cycles),
    ]

    slices = converter._plan_episode_slices(  # noqa: SLF001
        episodes,
        converter.SplitConfig(post_release_frames=0),
    )

    assert [item.output_index for item in slices] == [0, 1, 2]
    assert [item.source_episode_dir.name for item in slices] == ["episode0", "episode1", "episode1"]
    assert [(item.cycle.start, item.cycle.end) for item in slices] == [(0, 7), (0, 7), (7, 13)]


def test_test_mode_does_not_construct_sam3(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    episode_dir = tmp_path / "episode0"
    episode_dir.mkdir()
    gripper = np.array([0.09] * 3 + [0.07] * 3 + [0.09] * 3, dtype=np.float64)
    with h5py.File(episode_dir / "data.hdf5", "w") as file:
        file.create_dataset(converter.TCP_KEY, data=np.zeros((len(gripper), 6), dtype=np.float64))
        file.create_dataset(converter.GRIPPER_KEY, data=gripper)

    def fail_if_called(config: converter.Sam3Config, initial_prompts: tuple[str, ...]) -> None:
        del config, initial_prompts
        raise AssertionError("SAM 3 should not be constructed in test mode")

    monkeypatch.setattr(converter, "_make_preprocessor", fail_if_called)
    converter.main(converter.Args(data_dir=tmp_path, test_mode=True))


def test_convert_v3_staging_dataset_to_expected_v21_layout(tmp_path: Path) -> None:
    source_root = tmp_path / "v3"
    destination_root = tmp_path / "v2.1"
    source_meta = source_root / "meta"
    source_episodes = source_meta / "episodes/chunk-000"
    source_data = source_root / "data/chunk-000"
    source_episodes.mkdir(parents=True)
    source_data.mkdir(parents=True)

    features = {
        "state": {"dtype": "float32", "shape": [1], "names": ["state"]},
        "episode_index": {"dtype": "int64", "shape": [1], "names": None},
        "frame_index": {"dtype": "int64", "shape": [1], "names": None},
    }
    (source_meta / "info.json").write_text(
        json.dumps(
            {
                "codebase_version": "v3.0",
                "fps": converter.FPS,
                "features": features,
                "robot_type": "realman arm with pika gripper",
            }
        ),
        encoding="utf-8",
    )
    converter.pq.write_table(
        converter.pa.Table.from_pylist(
            [
                {"task_index": 0, "task": "grasp the red block"},
                {"task_index": 1, "task": "grasp the blue block"},
            ]
        ),
        source_meta / "tasks.parquet",
    )
    episode_records = [
        {
            "episode_index": episode_index,
            "tasks": [f"grasp the {color} block"],
            "length": length,
            "data/chunk_index": 0,
            "data/file_index": 0,
            "stats/state/min": [float(episode_index)],
            "stats/state/max": [float(episode_index + length - 1)],
            "stats/state/mean": [float(episode_index)],
            "stats/state/std": [0.0],
            "stats/state/count": [length],
        }
        for episode_index, color, length in ((0, "red", 2), (1, "blue", 1))
    ]
    converter.pq.write_table(
        converter.pa.Table.from_pylist(episode_records),
        source_episodes / "file-000.parquet",
    )
    source_table = converter.pa.table(
        {
            "state": [[0.0], [1.0], [2.0]],
            "episode_index": [0, 0, 1],
            "frame_index": [0, 1, 0],
        }
    )
    source_table = source_table.replace_schema_metadata(
        {
            b"huggingface": json.dumps(
                {
                    "info": {
                        "features": {
                            "state": {
                                "feature": {"dtype": "float32", "_type": "Value"},
                                "length": 1,
                                "_type": "List",
                            }
                        }
                    },
                    "fingerprint": "v3-only",
                }
            ).encode()
        }
    )
    converter.pq.write_table(source_table, source_data / "file-000.parquet")

    converter._convert_lerobot_v3_to_v21(source_root, destination_root)  # noqa: SLF001

    output_info = json.loads((destination_root / "meta/info.json").read_text())
    assert output_info["codebase_version"] == "v2.1"
    assert output_info["total_episodes"] == 2
    assert output_info["total_frames"] == 3
    assert output_info["data_path"] == ("data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet")
    assert sorted(path.name for path in (destination_root / "data/chunk-000").iterdir()) == [
        "episode_000000.parquet",
        "episode_000001.parquet",
    ]
    assert not (destination_root / "meta/tasks.parquet").exists()
    tasks = [json.loads(line) for line in (destination_root / "meta/tasks.jsonl").read_text().splitlines()]
    episodes = [json.loads(line) for line in (destination_root / "meta/episodes.jsonl").read_text().splitlines()]
    stats = [json.loads(line) for line in (destination_root / "meta/episodes_stats.jsonl").read_text().splitlines()]
    assert [task["task_index"] for task in tasks] == [0, 1]
    assert [episode["length"] for episode in episodes] == [2, 1]
    assert stats[0]["stats"]["state"]["count"] == [2]
    output_schema = converter.pq.read_schema(destination_root / "data/chunk-000/episode_000000.parquet")
    huggingface_metadata = json.loads(output_schema.metadata[b"huggingface"])
    assert huggingface_metadata["info"]["features"]["state"]["_type"] == "Sequence"
    assert "fingerprint" not in huggingface_metadata
    assert (
        converter.pq.read_table(destination_root / "data/chunk-000/episode_000001.parquet").to_pylist()
        == source_table.slice(2, 1).to_pylist()
    )

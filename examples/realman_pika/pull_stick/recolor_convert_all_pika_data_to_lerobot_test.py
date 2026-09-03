"""Tests for pull-stick two-color SAM3 recoloring.

Run with:
uv run --project examples/realman_pika --no-sync pytest \
  examples/realman_pika/pull_stick/recolor_convert_all_pika_data_to_lerobot_test.py
"""

from __future__ import annotations

# The tests intentionally exercise small internal helpers without loading SAM3.
# ruff: noqa: SLF001
import numpy as np
import pytest
import recolor_convert_all_pika_data_to_lerobot as converter


def test_apply_recolor_masks_gives_grabbed_stick_color1_and_other_sticks_color2() -> None:
    image = np.full((3, 4, 3), 100, dtype=np.uint8)
    all_sticks = np.zeros((3, 4), dtype=bool)
    all_sticks[1, 1:3] = True
    grabbed = np.zeros((3, 4), dtype=bool)
    grabbed[1, 1] = True

    output = converter._apply_recolor_masks(
        image,
        all_sticks,
        grabbed,
        converter.RecolorConfig(color1_rgb=(255, 0, 0), color2_rgb=(0, 0, 255), alpha=1.0),
    )

    np.testing.assert_array_equal(output[1, 1], (100, 0, 0))
    np.testing.assert_array_equal(output[1, 2], (0, 0, 100))
    np.testing.assert_array_equal(output[0, 0], image[0, 0])


def test_resolve_task_prompt_replaces_both_color_variables() -> None:
    prompt = converter._resolve_task_prompt(
        "pull ${color1}; others are ${color2}",
        (255, 0, 255),
        (0, 255, 255),
    )
    assert prompt == "pull magenta; others are cyan"


def test_default_task_prompt_uses_grabbed_stick_color() -> None:
    prompt = converter._resolve_task_prompt(
        converter.DEFAULT_TASK_PROMPT,
        (255, 0, 255),
        (0, 255, 255),
    )
    assert prompt == "pull the magenta stick and place it on the desk"


def test_validate_args_rejects_identical_colors() -> None:
    args = converter.Args(recolor=converter.RecolorConfig(color1_rgb=(0, 0, 255), color2_rgb=(0, 0, 255)))
    with pytest.raises(ValueError, match="must differ"):
        converter._validate_args(args)


def test_validate_args_rejects_out_of_range_shard_index() -> None:
    with pytest.raises(ValueError, match="shard-index"):
        converter._validate_args(converter.Args(num_shards=8, shard_index=8))


def test_select_grabbed_stick_prefers_instance_touching_gripper_cap() -> None:
    cap = np.zeros((20, 30), dtype=bool)
    cap[9:12, 13:16] = True
    near = np.zeros_like(cap)
    near[8:18, 14:17] = True
    far_high_confidence = np.zeros_like(cap)
    far_high_confidence[2:6, 2:10] = True

    selected = converter._select_grabbed_stick_mask(
        [far_high_confidence, near],
        [0.99, 0.5],
        cap,
        (14.0, 10.0),
        max_distance=0.2,
    )

    np.testing.assert_array_equal(selected, near)


def test_missing_white_cap_leaves_tracker_inactive_for_adjacent_frame_retry() -> None:
    class AllSticks:
        def _validate_images(self, images: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
            return images

        def track_masks(self, images: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
            return {name: np.ones(image.shape[:2], dtype=bool) for name, image in images.items()}

    class GrabbedStick:
        def track_masks(self, images: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
            del images
            raise converter._WhiteCapNotFoundError("wrist_image: no white cap candidate between the jaws")

    preprocessor = converter.Sam3TwoColorRecolorPreprocessor.__new__(converter.Sam3TwoColorRecolorPreprocessor)
    preprocessor.all_sticks = AllSticks()
    preprocessor.grabbed_stick = GrabbedStick()
    preprocessor.recolor = converter.RecolorConfig()
    images = {
        "image": np.full((4, 5, 3), 10, dtype=np.uint8),
        "wrist_image": np.full((4, 5, 3), 20, dtype=np.uint8),
    }

    output = preprocessor.preprocess(images)

    for camera_name, image in images.items():
        np.testing.assert_array_equal(output[camera_name], image)
        assert output[camera_name] is not image


def test_default_sam3_checkpoint_points_outside_openpi_repo() -> None:
    assert converter.DEFAULT_SAM3_CHECKPOINT.name == "SAM3"
    assert converter.DEFAULT_SAM3_CHECKPOINT.parent.name == "foundation_models"
    assert converter.DEFAULT_SAM3_CHECKPOINT.parents[1].name == "zehao"


def test_make_preview_frame_accepts_native_originals_and_resized_outputs() -> None:
    originals = {
        "7:image": np.full((480, 640, 3), (20, 40, 60), dtype=np.uint8),
        "7:wrist_image": np.full((360, 640, 3), (80, 100, 120), dtype=np.uint8),
    }
    processed = {
        "7:image": np.full((224, 224, 3), (255, 0, 255), dtype=np.uint8),
        "7:wrist_image": np.full((224, 224, 3), (0, 255, 255), dtype=np.uint8),
    }

    preview = converter._make_preview_frame(originals, processed, 7, "pull magenta; others cyan")

    assert preview.shape == (960, 1280, 3)
    np.testing.assert_array_equal(preview[200, 900], (255, 0, 255))
    np.testing.assert_array_equal(preview[700, 900], (0, 255, 255))


def test_dataset_images_are_resized_only_at_writer_boundary() -> None:
    class Dataset:
        frame: dict[str, object] | None = None

        def add_frame(self, frame: dict[str, object]) -> None:
            self.frame = frame

    dataset = Dataset()
    wrapper = converter._ResizeDatasetImages(dataset)
    wrapper.add_frame(
        {
            "image": np.zeros((480, 640, 3), dtype=np.uint8),
            "wrist_image": np.zeros((480, 640, 3), dtype=np.uint8),
            "state": np.zeros(7),
        }
    )

    assert dataset.frame is not None
    assert np.asarray(dataset.frame["image"]).shape == (224, 224, 3)
    assert np.asarray(dataset.frame["wrist_image"]).shape == (224, 224, 3)

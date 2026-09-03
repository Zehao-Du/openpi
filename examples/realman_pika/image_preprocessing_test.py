from __future__ import annotations

from contextlib import nullcontext
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal

import image_preprocessing
import numpy as np
import pytest
import torch


class _Batch(dict):
    def to(self, device: str) -> _Batch:
        return self


class _FakeModel:
    def __init__(self, *, error: Exception | None = None) -> None:
        self.error = error
        self.forward_count = 0

    def to(self, device: str) -> _FakeModel:
        return self

    def eval(self) -> _FakeModel:
        return self

    def __call__(self, **inputs: Any) -> object:
        self.forward_count += 1
        if self.error is not None:
            raise self.error
        return object()


class _FakeProcessor:
    def __init__(self, results: list[dict[str, Any]]) -> None:
        self.results = results
        self.calls: list[tuple[list[np.ndarray], list[str]]] = []
        self.sizes: list[dict[str, int]] = []
        self.post_process_kwargs: list[dict[str, Any]] = []

    def __call__(
        self,
        *,
        images: list[np.ndarray],
        text: list[str],
        return_tensors: str,
        size: dict[str, int],
    ) -> _Batch:
        self.calls.append((images, text))
        self.sizes.append(size)
        sizes = [[image.shape[0], image.shape[1]] for image in images]
        return _Batch(original_sizes=torch.tensor(sizes), pixel_values=torch.empty(len(images), 1))

    def post_process_instance_segmentation(self, outputs: object, **kwargs: Any) -> list[dict[str, Any]]:
        self.post_process_kwargs.append(kwargs)
        return self.results


class _FakeTorch:
    class cuda:  # noqa: N801
        @staticmethod
        def is_available() -> bool:
            return False

    @staticmethod
    def inference_mode() -> nullcontext:
        return nullcontext()


def _preprocessor(
    model: _FakeModel,
    processor: _FakeProcessor,
    *,
    prompts: tuple[str, ...] = ("pink block",),
    min_component_area: int = 1,
    error_policy: Literal["fallback", "raise"] = "fallback",
) -> image_preprocessing.Sam3RecolorPreprocessor:
    return image_preprocessing.Sam3RecolorPreprocessor(
        "unused",
        prompts=prompts,
        target_rgb=(0, 0, 255),
        device="cpu",
        alpha=1.0,
        min_component_area=min_component_area,
        error_policy=error_policy,
        model=model,
        processor=processor,
        torch_module=_FakeTorch,
    )


def test_recolor_changes_only_mask_and_preserves_value_and_dtype() -> None:
    image = np.full((12, 12, 3), (200, 100, 50), dtype=np.uint8)
    mask = np.zeros((12, 12), dtype=bool)
    mask[2:10, 3:11] = True

    output = image_preprocessing.recolor_masked_region(
        image,
        mask,
        target_rgb=(0, 0, 255),
        alpha=1.0,
    )

    assert output.dtype == np.uint8
    np.testing.assert_array_equal(output[~mask], image[~mask])
    np.testing.assert_array_equal(output[mask], np.tile((0, 0, 200), (mask.sum(), 1)))


def test_clean_mask_keeps_all_large_components_and_removes_noise() -> None:
    mask = np.zeros((32, 32), dtype=bool)
    mask[2:10, 2:10] = True
    mask[18:28, 20:28] = True
    mask[14, 14] = True

    cleaned = image_preprocessing.clean_mask(mask, min_component_area=64)

    assert cleaned[2:10, 2:10].all()
    assert cleaned[18:28, 20:28].all()
    assert not cleaned[14, 14]


def test_empty_mask_returns_unchanged_image() -> None:
    image = np.arange(12 * 12 * 3, dtype=np.uint8).reshape(12, 12, 3)
    output = image_preprocessing.recolor_masked_region(
        image,
        np.zeros((12, 12), dtype=bool),
        target_rgb=(0, 0, 255),
        alpha=0.9,
    )
    np.testing.assert_array_equal(output, image)


def test_multiple_cameras_and_prompts_use_one_model_forward() -> None:
    mask = np.zeros((10, 10), dtype=bool)
    mask[2:8, 2:8] = True
    results = [{"masks": np.asarray([mask])} for _ in range(4)]
    model = _FakeModel()
    processor = _FakeProcessor(results)
    preprocessor = _preprocessor(model, processor, prompts=("pink block", "pink cube"))
    images = {
        "fisheye": np.full((10, 10, 3), 100, dtype=np.uint8),
        "rgb": np.full((10, 10, 3), 200, dtype=np.uint8),
    }

    output = preprocessor.preprocess(images)

    assert model.forward_count == 1
    assert len(processor.calls) == 1
    assert processor.sizes == [{"height": 224, "width": 224}]
    batch_images, batch_prompts = processor.calls[0]
    assert len(batch_images) == 4
    assert batch_prompts == ["pink block", "pink cube", "pink block", "pink cube"]
    assert (output["fisheye"][mask] == (0, 0, 100)).all()
    assert (output["rgb"][mask] == (0, 0, 200)).all()


def test_prompts_can_change_between_offline_episodes_without_reloading_model() -> None:
    mask = np.ones((10, 10), dtype=bool)
    model = _FakeModel()
    processor = _FakeProcessor([{"masks": np.asarray([mask])}])
    preprocessor = _preprocessor(model, processor)

    preprocessor.set_prompts(("green block",))
    preprocessor.preprocess({"rgb": np.full((10, 10, 3), 100, dtype=np.uint8)})

    assert model.forward_count == 1
    assert processor.calls[0][1] == ["green block"]


def test_bad_result_for_one_camera_falls_back_only_for_that_camera() -> None:
    good_mask = np.ones((10, 10), dtype=bool)
    results = [
        {"masks": np.ones((1, 3, 3), dtype=bool)},
        {"masks": np.asarray([good_mask])},
    ]
    preprocessor = _preprocessor(_FakeModel(), _FakeProcessor(results))
    images = {
        "fisheye": np.full((10, 10, 3), 100, dtype=np.uint8),
        "rgb": np.full((10, 10, 3), 200, dtype=np.uint8),
    }

    output = preprocessor.preprocess(images)

    np.testing.assert_array_equal(output["fisheye"], images["fisheye"])
    assert (output["rgb"] == (0, 0, 200)).all()


def test_model_failure_returns_all_original_images() -> None:
    model = _FakeModel(error=RuntimeError("forward failed"))
    preprocessor = _preprocessor(model, _FakeProcessor([]))
    images = {
        "fisheye": np.full((10, 10, 3), 100, dtype=np.uint8),
        "rgb": np.full((10, 10, 3), 200, dtype=np.uint8),
    }

    output = preprocessor.preprocess(images)

    assert model.forward_count == 1
    np.testing.assert_array_equal(output["fisheye"], images["fisheye"])
    np.testing.assert_array_equal(output["rgb"], images["rgb"])


def test_raise_error_policy_propagates_model_failure() -> None:
    model = _FakeModel(error=RuntimeError("forward failed"))
    preprocessor = _preprocessor(model, _FakeProcessor([]), error_policy="raise")

    with pytest.raises(RuntimeError, match="forward failed"):
        preprocessor.preprocess({"fisheye": np.full((10, 10, 3), 100, dtype=np.uint8)})


def test_episode_tracker_applies_camera_specific_first_frame_score_threshold() -> None:
    mask = np.zeros((10, 10), dtype=bool)
    mask[2:8, 2:8] = True
    processor = _FakeProcessor(
        [
            {"scores": torch.tensor([0.41]), "masks": torch.tensor(mask[None])},
            {"scores": torch.tensor([0.41]), "masks": torch.tensor(mask[None])},
        ]
    )
    preprocessor = image_preprocessing.Sam3EpisodeTrackerPreprocessor.__new__(
        image_preprocessing.Sam3EpisodeTrackerPreprocessor
    )
    preprocessor.prompts = ("red block",)
    preprocessor.model_input_size = 224
    preprocessor.device = "cpu"
    preprocessor.score_threshold = 0.5
    preprocessor.camera_score_thresholds = {"image": 0.4}
    preprocessor.mask_threshold = 0.3
    preprocessor.min_component_area = 1
    preprocessor._processor = processor  # noqa: SLF001
    preprocessor._model = _FakeModel()  # noqa: SLF001
    preprocessor._torch = _FakeTorch  # noqa: SLF001

    masks = preprocessor._detect_first_frame(  # noqa: SLF001
        {
            "image": np.zeros((10, 10, 3), dtype=np.uint8),
            "wrist_image": np.zeros((10, 10, 3), dtype=np.uint8),
        }
    )

    assert processor.post_process_kwargs[0]["threshold"] == 0.4
    assert masks["image"].any()
    assert not masks["wrist_image"].any()


def test_episode_tracker_detects_first_frame_then_tracks_later_frames() -> None:
    preprocessor = image_preprocessing.Sam3EpisodeTrackerPreprocessor.__new__(
        image_preprocessing.Sam3EpisodeTrackerPreprocessor
    )
    preprocessor._frame_index = 0  # noqa: SLF001
    preprocessor._sessions = {}  # noqa: SLF001
    preprocessor.error_policy = "raise"
    preprocessor.target_rgb = (0, 0, 255)
    preprocessor.alpha = 0.9
    preprocessor.last_elapsed_ms = None
    calls: list[str] = []

    def detect(images: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        calls.append("detect")
        return {name: np.ones(image.shape[:2], dtype=bool) for name, image in images.items()}

    def initialize(images: dict[str, np.ndarray], masks: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        preprocessor._sessions = {name: object() for name in images}  # noqa: SLF001
        return masks

    def track(camera_name: str, image: np.ndarray) -> np.ndarray:
        calls.append(f"track:{camera_name}")
        return np.ones(image.shape[:2], dtype=bool)

    preprocessor._detect_first_frame = detect  # type: ignore[method-assign]  # noqa: SLF001
    preprocessor._initialize_first_frame_trackers = initialize  # type: ignore[method-assign]  # noqa: SLF001
    preprocessor._track_frame = track  # type: ignore[method-assign]  # noqa: SLF001
    preprocessor._maybe_redetect_shrunken_mask = lambda images, masks: masks  # type: ignore[method-assign]  # noqa: SLF001
    images = {
        "fisheye": np.full((10, 10, 3), 100, dtype=np.uint8),
        "rgb": np.full((10, 10, 3), 200, dtype=np.uint8),
    }

    preprocessor.preprocess(images)
    preprocessor.preprocess(images)

    assert calls == ["detect", "track:fisheye", "track:rgb"]


def _identity_camera_mapping() -> image_preprocessing.PolynomialCameraMapping:
    return image_preprocessing.PolynomialCameraMapping(
        source_size_wh=(640, 480),
        destination_size_wh=(640, 480),
        term_powers_xy=np.asarray(((0, 0), (1, 0), (0, 1))),
        destination_normalized_x=np.asarray((0.0, 1.0, 0.0)),
        destination_normalized_y=np.asarray((0.0, 0.0, 1.0)),
    )


def test_polynomial_camera_mapping_loads_json_and_maps_known_points(tmp_path: Path) -> None:
    mapping_path = tmp_path / "mapping.json"
    mapping_path.write_text(
        json.dumps(
            {
                "format_version": 1,
                "model": "normalized_2d_polynomial",
                "normalization": "pixel_to_minus_one_plus_one_using_(size-1)",
                "source_size_wh": [640, 480],
                "destination_size_wh": [640, 480],
                "term_powers_xy": [[0, 0], [1, 0], [0, 1]],
                "coefficients": {
                    "destination_normalized_x": [0.0, 1.0, 0.0],
                    "destination_normalized_y": [0.0, 0.0, 1.0],
                },
            }
        ),
        encoding="utf-8",
    )

    mapping = image_preprocessing.PolynomialCameraMapping.from_json(mapping_path)
    points = np.asarray(((0.0, 0.0), (319.5, 239.5), (639.0, 479.0)))

    np.testing.assert_allclose(mapping.map_points(points), points, atol=1e-10)


def test_resize_pad_point_transform_round_trip() -> None:
    native_points = np.asarray(((0.0, 0.0), (320.0, 240.0), (639.0, 479.0)))
    padded_points = image_preprocessing.resize_pad_points(
        native_points,
        native_size_wh=(640, 480),
        padded_size_wh=(224, 224),
    )

    np.testing.assert_allclose(padded_points[1], (112.0, 112.0))
    np.testing.assert_allclose(
        image_preprocessing.resize_pad_points(
            padded_points,
            native_size_wh=(640, 480),
            padded_size_wh=(224, 224),
            inverse=True,
        ),
        native_points,
    )


def test_source_mask_projects_to_destination_point_and_box() -> None:
    preprocessor = image_preprocessing.Sam3EpisodeTrackerPreprocessor.__new__(
        image_preprocessing.Sam3EpisodeTrackerPreprocessor
    )
    preprocessor.cross_camera_mapping = _identity_camera_mapping()
    preprocessor.spatial_prompt_box_padding = 4.0
    mask = np.zeros((224, 224), dtype=bool)
    mask[100:120, 90:110] = True

    point, box = preprocessor._cross_camera_spatial_prompt(  # noqa: SLF001
        mask,
        source_image_shape=(224, 224),
        destination_image_shape=(224, 224),
    )

    np.testing.assert_allclose(point, (99.5, 109.5))
    np.testing.assert_allclose(box, (86.0, 96.0, 113.0, 123.0))


def test_spatial_prompt_tracker_receives_positive_point_and_box() -> None:
    class _Session:
        def add_new_frame(self, pixel_values: Any) -> int:
            self.pixel_values = pixel_values
            return 0

    class _TrackerProcessor:
        def __init__(self) -> None:
            self.prompt_kwargs: dict[str, Any] | None = None

        def init_video_session(self, **kwargs: Any) -> _Session:
            return _Session()

        def __call__(self, **kwargs: Any) -> dict[str, Any]:
            return {
                "pixel_values": torch.zeros((1, 3, 10, 10)),
                "original_sizes": torch.tensor(((10, 10),)),
            }

        def process_new_points_or_boxes_for_video_frame(self, session: _Session, **kwargs: Any) -> None:
            self.prompt_kwargs = kwargs

        def post_process_masks(self, *args: Any, **kwargs: Any) -> list[torch.Tensor]:
            return [torch.ones((1, 1, 10, 10), dtype=torch.bool)]

    tracker_processor = _TrackerProcessor()
    preprocessor = image_preprocessing.Sam3EpisodeTrackerPreprocessor.__new__(
        image_preprocessing.Sam3EpisodeTrackerPreprocessor
    )
    preprocessor._tracker_processor = tracker_processor  # noqa: SLF001
    preprocessor._tracker_model = lambda **kwargs: SimpleNamespace(  # noqa: SLF001
        pred_masks=torch.ones((1, 1, 10, 10))
    )
    preprocessor._cache_tracker_vision_features = lambda session, frame_idx: None  # noqa: SLF001
    preprocessor._torch = torch  # noqa: SLF001
    preprocessor.device = "cpu"
    preprocessor.model_input_size = 224
    preprocessor.mask_threshold = 0.3
    preprocessor.min_component_area = 1
    preprocessor._sessions = {}  # noqa: SLF001

    mask = preprocessor._start_tracker_with_spatial_prompt(  # noqa: SLF001
        "image",
        np.zeros((10, 10, 3), dtype=np.uint8),
        (4.0, 5.0),
        (2.0, 3.0, 7.0, 8.0),
    )

    assert mask.all()
    assert tracker_processor.prompt_kwargs is not None
    assert tracker_processor.prompt_kwargs["input_points"] == [[[[4.0, 5.0]]]]
    assert tracker_processor.prompt_kwargs["input_labels"] == [[[1]]]
    assert tracker_processor.prompt_kwargs["input_boxes"] == [[[2.0, 3.0, 7.0, 8.0]]]
    assert tracker_processor.prompt_kwargs["original_size"] == (10, 10)


def test_cross_camera_prompt_failure_falls_back_to_detector_masks() -> None:
    preprocessor = image_preprocessing.Sam3EpisodeTrackerPreprocessor.__new__(
        image_preprocessing.Sam3EpisodeTrackerPreprocessor
    )
    preprocessor.cross_camera_mapping = _identity_camera_mapping()
    preprocessor.mapping_source_camera = "wrist_image"
    preprocessor.mapping_destination_camera = "image"
    preprocessor._cross_camera_spatial_prompt = lambda *args, **kwargs: (_ for _ in ()).throw(  # noqa: SLF001
        ValueError("mapping failed")
    )
    started: list[tuple[str, np.ndarray]] = []
    preprocessor._start_tracker = lambda name, image, mask: started.append((name, mask.copy()))  # noqa: SLF001
    source_mask = np.ones((10, 10), dtype=bool)
    destination_mask = np.zeros((10, 10), dtype=bool)
    destination_mask[2:5, 3:7] = True
    images = {
        "image": np.zeros((10, 10, 3), dtype=np.uint8),
        "wrist_image": np.zeros((10, 10, 3), dtype=np.uint8),
    }

    output_masks = preprocessor._initialize_first_frame_trackers(  # noqa: SLF001
        images,
        {"image": destination_mask.copy(), "wrist_image": source_mask.copy()},
    )

    assert [name for name, _ in started] == ["image", "wrist_image"]
    np.testing.assert_array_equal(output_masks["image"], destination_mask)


def test_significantly_shrunken_mask_triggers_one_redetect_and_restarts_reference() -> None:
    preprocessor = image_preprocessing.Sam3EpisodeTrackerPreprocessor.__new__(
        image_preprocessing.Sam3EpisodeTrackerPreprocessor
    )
    preprocessor.mapping_destination_camera = "image"
    preprocessor.redetect_area_ratio = 0.5
    preprocessor.redetect_reference_decay = 0.98
    preprocessor.redetect_cooldown_frames = 15
    preprocessor._mask_area_reference = 100.0  # noqa: SLF001
    preprocessor._last_redetect_frame = -15  # noqa: SLF001
    preprocessor._frame_index = 10  # noqa: SLF001
    recovered_mask = np.zeros((10, 10), dtype=bool)
    recovered_mask[1:9, 1:9] = True
    redetect_calls: list[dict[str, np.ndarray]] = []
    preprocessor._redetect_destination = lambda images: (  # noqa: SLF001
        redetect_calls.append(images) or recovered_mask
    )
    shrunken_mask = np.zeros((10, 10), dtype=bool)
    shrunken_mask[1:5, 1:5] = True
    images = {"image": np.zeros((10, 10, 3), dtype=np.uint8)}

    masks = preprocessor._maybe_redetect_shrunken_mask(  # noqa: SLF001
        images, {"image": shrunken_mask}
    )

    assert len(redetect_calls) == 1
    np.testing.assert_array_equal(masks["image"], recovered_mask)
    assert preprocessor._mask_area_reference == 64.0  # noqa: SLF001
    assert preprocessor._last_redetect_frame == 10  # noqa: SLF001


def test_normal_mask_area_does_not_redetect() -> None:
    preprocessor = image_preprocessing.Sam3EpisodeTrackerPreprocessor.__new__(
        image_preprocessing.Sam3EpisodeTrackerPreprocessor
    )
    preprocessor.mapping_destination_camera = "image"
    preprocessor.redetect_area_ratio = 0.5
    preprocessor.redetect_reference_decay = 0.98
    preprocessor.redetect_cooldown_frames = 15
    preprocessor._mask_area_reference = 100.0  # noqa: SLF001
    preprocessor._last_redetect_frame = -15  # noqa: SLF001
    preprocessor._frame_index = 10  # noqa: SLF001
    preprocessor._redetect_destination = lambda images: (_ for _ in ()).throw(  # noqa: SLF001
        AssertionError("normal area must not trigger detection")
    )
    mask = np.zeros((10, 10), dtype=bool)
    mask[:6, :] = True

    preprocessor._maybe_redetect_shrunken_mask(  # noqa: SLF001
        {"image": np.zeros((10, 10, 3), dtype=np.uint8)}, {"image": mask}
    )

    assert preprocessor._mask_area_reference == 98.0  # noqa: SLF001

"""Optional image preprocessing for the RealMan-Pika policy client."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
import logging
import pathlib
import time
from typing import Any, Literal, Protocol

import numpy as np
from scipy import ndimage

logger = logging.getLogger(__name__)


class PolynomialCameraMapping:
    """Map pixels between two native camera images with a calibrated polynomial."""

    def __init__(
        self,
        *,
        source_size_wh: tuple[int, int],
        destination_size_wh: tuple[int, int],
        term_powers_xy: np.ndarray,
        destination_normalized_x: np.ndarray,
        destination_normalized_y: np.ndarray,
    ) -> None:
        self.source_size_wh = self.validate_size(source_size_wh, "source_size_wh")
        self.destination_size_wh = self.validate_size(
            destination_size_wh, "destination_size_wh"
        )
        self.term_powers_xy = np.asarray(term_powers_xy, dtype=np.int64)
        self.destination_normalized_x = np.asarray(destination_normalized_x, dtype=np.float64)
        self.destination_normalized_y = np.asarray(destination_normalized_y, dtype=np.float64)
        if self.term_powers_xy.ndim != 2 or self.term_powers_xy.shape[1] != 2:
            raise ValueError(
                f"term_powers_xy must have shape (N, 2), got {self.term_powers_xy.shape}"
            )
        if np.any(self.term_powers_xy < 0):
            raise ValueError("term_powers_xy must contain non-negative powers")
        term_count = len(self.term_powers_xy)
        if self.destination_normalized_x.shape != (term_count,):
            raise ValueError("destination_normalized_x length does not match term_powers_xy")
        if self.destination_normalized_y.shape != (term_count,):
            raise ValueError("destination_normalized_y length does not match term_powers_xy")
        if not (
            np.isfinite(self.destination_normalized_x).all()
            and np.isfinite(self.destination_normalized_y).all()
        ):
            raise ValueError("Camera mapping coefficients must be finite")

    @staticmethod
    def validate_size(size_wh: tuple[int, int], name: str) -> tuple[int, int]:
        if len(size_wh) != 2 or any(int(value) < 2 for value in size_wh):
            raise ValueError(f"{name} must contain width and height >= 2, got {size_wh}")
        return int(size_wh[0]), int(size_wh[1])

    @classmethod
    def from_json(cls, path: pathlib.Path | str) -> PolynomialCameraMapping:
        path = pathlib.Path(path).expanduser()
        with path.open(encoding="utf-8") as file:
            data = json.load(file)
        if data.get("format_version") != 1:
            raise ValueError(f"Unsupported camera mapping format_version in {path}")
        if data.get("model") != "normalized_2d_polynomial":
            raise ValueError(f"Unsupported camera mapping model in {path}: {data.get('model')!r}")
        expected_normalization = "pixel_to_minus_one_plus_one_using_(size-1)"
        if data.get("normalization") != expected_normalization:
            raise ValueError(
                f"Unsupported camera mapping normalization in {path}: {data.get('normalization')!r}"
            )
        try:
            coefficients = data["coefficients"]
            return cls(
                source_size_wh=tuple(data["source_size_wh"]),
                destination_size_wh=tuple(data["destination_size_wh"]),
                term_powers_xy=np.asarray(data["term_powers_xy"]),
                destination_normalized_x=np.asarray(coefficients["destination_normalized_x"]),
                destination_normalized_y=np.asarray(coefficients["destination_normalized_y"]),
            )
        except (KeyError, TypeError) as error:
            raise ValueError(f"Malformed camera mapping JSON: {path}") from error

    def map_points(self, source_points_xy: np.ndarray) -> np.ndarray:
        """Map ``(..., x/y)`` native source pixels to native destination pixels."""
        points = np.asarray(source_points_xy, dtype=np.float64)
        if points.ndim < 1 or points.shape[-1] != 2:
            raise ValueError(f"source_points_xy must have shape (..., 2), got {points.shape}")
        source_width, source_height = self.source_size_wh
        normalized_x = 2.0 * points[..., 0] / (source_width - 1) - 1.0
        normalized_y = 2.0 * points[..., 1] / (source_height - 1) - 1.0
        terms = np.stack(
            [
                normalized_x**power_x * normalized_y**power_y
                for power_x, power_y in self.term_powers_xy
            ],
            axis=-1,
        )
        destination_x = terms @ self.destination_normalized_x
        destination_y = terms @ self.destination_normalized_y
        destination_width, destination_height = self.destination_size_wh
        return np.stack(
            [
                (destination_x + 1.0) * (destination_width - 1) / 2.0,
                (destination_y + 1.0) * (destination_height - 1) / 2.0,
            ],
            axis=-1,
        )


def resize_pad_points(
    points_xy: np.ndarray,
    *,
    native_size_wh: tuple[int, int],
    padded_size_wh: tuple[int, int],
    inverse: bool = False,
) -> np.ndarray:
    """Transform points exactly like ``openpi_client.image_tools.resize_with_pad``."""
    points = np.asarray(points_xy, dtype=np.float64)
    if points.ndim < 1 or points.shape[-1] != 2:
        raise ValueError(f"points_xy must have shape (..., 2), got {points.shape}")
    native_width, native_height = PolynomialCameraMapping.validate_size(
        native_size_wh, "native_size_wh"
    )
    padded_width, padded_height = PolynomialCameraMapping.validate_size(
        padded_size_wh, "padded_size_wh"
    )
    ratio = max(native_width / padded_width, native_height / padded_height)
    resized_width = int(native_width / ratio)
    resized_height = int(native_height / ratio)
    pad_x = max(0, int((padded_width - resized_width) / 2))
    pad_y = max(0, int((padded_height - resized_height) / 2))
    scale = np.asarray([resized_width / native_width, resized_height / native_height])
    offset = np.asarray([pad_x, pad_y], dtype=np.float64)
    return (points - offset) / scale if inverse else points * scale + offset


class ImagePreprocessor(Protocol):
    """Transforms a named batch of RGB images without changing its structure."""

    def preprocess(self, images: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]: ...


def clean_mask(mask: np.ndarray, min_component_area: int) -> np.ndarray:
    """Close small holes and retain every sufficiently large 8-connected region."""
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2:
        raise ValueError(f"Expected a 2D mask, got shape {mask.shape}")
    if min_component_area < 1:
        raise ValueError("min_component_area must be at least 1")
    if not mask.any():
        return np.zeros_like(mask)

    cleaned = ndimage.binary_closing(mask, structure=np.ones((3, 3), dtype=bool), border_value=1)
    labels, component_count = ndimage.label(cleaned, structure=np.ones((3, 3), dtype=np.uint8))
    if component_count == 0:
        return np.zeros_like(mask)
    areas = np.bincount(labels.ravel())
    keep = np.flatnonzero(areas >= min_component_area)
    keep = keep[keep != 0]
    return np.isin(labels, keep)


def recolor_masked_region(
    image_rgb: np.ndarray,
    mask: np.ndarray,
    *,
    target_rgb: tuple[int, int, int],
    alpha: float,
) -> np.ndarray:
    """Blend in a target chroma while retaining each pixel's HSV value."""
    image_rgb = np.asarray(image_rgb)
    mask = np.asarray(mask, dtype=bool)
    if image_rgb.ndim != 3 or image_rgb.shape[-1] != 3 or image_rgb.dtype != np.uint8:
        raise ValueError(f"Expected an HWC RGB uint8 image, got shape {image_rgb.shape}, dtype {image_rgb.dtype}")
    if mask.shape != image_rgb.shape[:2]:
        raise ValueError(f"Mask shape {mask.shape} does not match image shape {image_rgb.shape[:2]}")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be between 0 and 1")
    if not mask.any():
        return image_rgb.copy()

    target = np.asarray(target_rgb, dtype=np.float32)
    if target.shape != (3,) or np.any(target < 0) or np.any(target > 255):
        raise ValueError(f"target_rgb must contain three values in [0, 255], got {target_rgb}")
    target_value = float(target.max())
    target_chroma = target / target_value if target_value > 0 else target

    source = image_rgb.astype(np.float32)
    source_value = source.max(axis=-1, keepdims=True)
    recolored = source_value * target_chroma
    output = image_rgb.copy()
    blended = (1.0 - alpha) * source[mask] + alpha * recolored[mask]
    blended_value = blended.max(axis=-1, keepdims=True)
    original_value = source_value[mask]
    blended *= np.divide(original_value, blended_value, out=np.ones_like(blended_value), where=blended_value > 0)
    output[mask] = np.clip(np.rint(blended), 0, 255).astype(np.uint8)
    return output


class Sam3RecolorPreprocessor:
    """Batch SAM 3 segmentation followed by brightness-preserving recoloring."""

    def __init__(
        self,
        checkpoint: pathlib.Path | str,
        *,
        prompts: tuple[str, ...] = ("pink block",),
        target_rgb: tuple[int, int, int] = (0, 0, 255),
        device: str | None = None,
        score_threshold: float = 0.5,
        mask_threshold: float = 0.5,
        alpha: float = 0.9,
        min_component_area: int = 64,
        model_input_size: int = 224,
        error_policy: Literal["fallback", "raise"] = "fallback",
        model: Any | None = None,
        processor: Any | None = None,
        torch_module: Any | None = None,
    ) -> None:
        self._validate_prompts(prompts)
        if len(target_rgb) != 3 or any(channel < 0 or channel > 255 for channel in target_rgb):
            raise ValueError(f"target_rgb must contain three values in [0, 255], got {target_rgb}")
        if not 0.0 <= score_threshold <= 1.0:
            raise ValueError("score_threshold must be between 0 and 1")
        if not 0.0 <= mask_threshold <= 1.0:
            raise ValueError("mask_threshold must be between 0 and 1")
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be between 0 and 1")
        if min_component_area < 1:
            raise ValueError("min_component_area must be at least 1")
        if model_input_size < 1 or model_input_size % 14 != 0:
            raise ValueError("model_input_size must be a positive multiple of SAM 3's 14-pixel patch size")
        if error_policy not in {"fallback", "raise"}:
            raise ValueError(f"error_policy must be 'fallback' or 'raise', got {error_policy!r}")
        if (model is None) != (processor is None):
            raise ValueError("model and processor must either both be supplied or both be omitted")

        if torch_module is None:
            import torch

            torch_module = torch

        self._torch = torch_module
        self.device = device or ("cuda" if torch_module.cuda.is_available() else "cpu")
        self.checkpoint = pathlib.Path(checkpoint).expanduser()
        self.prompts = tuple(prompts)
        self.target_rgb = tuple(target_rgb)
        self.score_threshold = score_threshold
        self.mask_threshold = mask_threshold
        self.alpha = alpha
        self.min_component_area = min_component_area
        self.model_input_size = model_input_size
        self.error_policy = error_policy
        self.last_elapsed_ms: float | None = None

        if model is None:
            if not self.checkpoint.exists():
                raise FileNotFoundError(f"SAM 3 checkpoint directory does not exist: {self.checkpoint}")
            from transformers import AutoConfig
            from transformers import Sam3Model
            from transformers import Sam3Processor

            logger.info("Loading SAM 3 from %s on %s", self.checkpoint, self.device)
            # facebook/sam3 is configured for 1008x1008 inputs.  The ViT builds
            # its global rotary-position buffers from that value at construction
            # time, so overriding only the processor size leaves 5184 positions
            # for the 256 tokens produced by a 224x224 image.  Construct the
            # detector with the actual online input size as well.
            video_config = AutoConfig.from_pretrained(self.checkpoint, local_files_only=True)
            detector_config = video_config.detector_config
            detector_config.image_size = self.model_input_size
            model = Sam3Model.from_pretrained(
                self.checkpoint,
                config=detector_config,
                local_files_only=True,
            )
            processor = Sam3Processor.from_pretrained(self.checkpoint, local_files_only=True)
        self._model = model.to(self.device)
        self._model.eval()
        self._processor = processor

    @staticmethod
    def _validate_prompts(prompts: tuple[str, ...]) -> None:
        if not prompts or any(not prompt.strip() for prompt in prompts):
            raise ValueError("prompts must contain at least one non-empty prompt")

    def set_prompts(self, prompts: tuple[str, ...]) -> None:
        """Change text prompts without reloading the model.

        This is intended for sequential offline conversion, where each episode
        can target a different object while sharing one loaded SAM 3 model.
        """
        self._validate_prompts(prompts)
        self.prompts = tuple(prompts)

    @staticmethod
    def _validate_images(images: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
        validated: dict[str, np.ndarray] = {}
        for name, image in images.items():
            image_array = np.asarray(image)
            if image_array.ndim != 3 or image_array.shape[-1] != 3 or image_array.dtype != np.uint8:
                raise ValueError(
                    f"Camera {name!r} must be an HWC RGB uint8 image, "
                    f"got shape {image_array.shape}, dtype {image_array.dtype}"
                )
            validated[name] = image_array
        return validated

    @staticmethod
    def _masks_to_numpy(masks: Any, expected_shape: tuple[int, int]) -> np.ndarray:
        if hasattr(masks, "detach"):
            masks = masks.detach().cpu().numpy()
        masks = np.asarray(masks, dtype=bool)
        if masks.size == 0:
            return np.zeros(expected_shape, dtype=bool)
        if masks.ndim == 2:
            masks = masks[None, ...]
        if masks.ndim != 3 or masks.shape[1:] != expected_shape:
            raise ValueError(
                f"SAM 3 returned masks with shape {masks.shape}; expected (N, {expected_shape[0]}, {expected_shape[1]})"
            )
        return np.any(masks, axis=0)

    def preprocess(self, images: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
        validated = self._validate_images(images)
        originals = {name: image.copy() for name, image in validated.items()}
        if not validated:
            return originals

        started = time.perf_counter()
        camera_names = list(validated)
        batch_camera_names = [name for name in camera_names for _ in self.prompts]
        batch_images = [validated[name] for name in camera_names for _ in self.prompts]
        batch_prompts = [prompt for _ in camera_names for prompt in self.prompts]

        try:
            inputs = self._processor(
                images=batch_images,
                text=batch_prompts,
                return_tensors="pt",
                size={"height": self.model_input_size, "width": self.model_input_size},
            )
            inputs = inputs.to(self.device)
            target_sizes = inputs["original_sizes"].detach().cpu().tolist()
            with self._torch.inference_mode():
                model_outputs = self._model(**inputs)
            results = self._processor.post_process_instance_segmentation(
                model_outputs,
                threshold=self.score_threshold,
                mask_threshold=self.mask_threshold,
                target_sizes=target_sizes,
            )
            if len(results) != len(batch_camera_names):
                raise ValueError(f"SAM 3 returned {len(results)} batch results for {len(batch_camera_names)} inputs")
        except Exception:
            self.last_elapsed_ms = (time.perf_counter() - started) * 1000.0
            if self.error_policy == "raise":
                raise
            logger.exception("SAM 3 inference failed; using the original camera images")
            return originals

        union_masks = {name: np.zeros(validated[name].shape[:2], dtype=bool) for name in camera_names}
        failed_cameras: set[str] = set()
        for camera_name, result in zip(batch_camera_names, results, strict=True):
            if camera_name in failed_cameras:
                continue
            try:
                masks = result.get("masks")
                if masks is not None:
                    union_masks[camera_name] |= self._masks_to_numpy(masks, validated[camera_name].shape[:2])
            except Exception as error:
                if self.error_policy == "raise":
                    raise RuntimeError(f"Failed to process the SAM 3 result for camera {camera_name}") from error
                logger.exception(
                    "Failed to process the SAM 3 result for camera %s; using its original image", camera_name
                )
                failed_cameras.add(camera_name)

        output: dict[str, np.ndarray] = {}
        for camera_name, image in validated.items():
            if camera_name in failed_cameras:
                output[camera_name] = originals[camera_name]
                continue
            mask = clean_mask(union_masks[camera_name], self.min_component_area)
            output[camera_name] = recolor_masked_region(
                image,
                mask,
                target_rgb=self.target_rgb,
                alpha=self.alpha,
            )

        self.last_elapsed_ms = (time.perf_counter() - started) * 1000.0
        logger.debug("SAM 3 preprocessed %d camera(s) in %.1f ms", len(images), self.last_elapsed_ms)
        return output


class Sam3EpisodeTrackerPreprocessor(Sam3RecolorPreprocessor):
    """Detect on an episode's first frame, then recolor tracker masks.

    One independent tracker session is maintained per image key, since camera
    viewpoints cannot share temporal memory. ``start_episode`` must be called
    before the first frame of every episode.
    """

    requires_sequential_frames = True

    def __init__(
        self,
        checkpoint: pathlib.Path | str,
        *,
        prompts: tuple[str, ...] = ("pink block",),
        target_rgb: tuple[int, int, int] = (0, 0, 255),
        device: str = "cuda",
        score_threshold: float = 0.5,
        camera_score_thresholds: Mapping[str, float] | None = None,
        mask_threshold: float = 0.5,
        alpha: float = 0.9,
        min_component_area: int = 64,
        model_input_size: int = 224,
        error_policy: Literal["fallback", "raise"] = "raise",
        video_model: Any | None = None,
        detector_processor: Any | None = None,
        tracker_processor: Any | None = None,
        torch_module: Any | None = None,
        cross_camera_mapping: pathlib.Path | str | PolynomialCameraMapping | None = None,
        mapping_source_camera: str = "wrist_image",
        mapping_destination_camera: str = "image",
        spatial_prompt_box_padding: float = 4.0,
        redetect_area_ratio: float = 0.5,
        redetect_reference_decay: float = 0.98,
        redetect_cooldown_frames: int = 15,
    ) -> None:
        camera_score_thresholds = dict(camera_score_thresholds or {})
        for camera_name, threshold in camera_score_thresholds.items():
            if not camera_name:
                raise ValueError("camera_score_thresholds keys must be non-empty")
            if not 0.0 <= threshold <= 1.0:
                raise ValueError(
                    f"camera_score_thresholds[{camera_name!r}] must be between 0 and 1"
                )
        if video_model is None:
            if torch_module is None:
                import torch

                torch_module = torch
            checkpoint_path = pathlib.Path(checkpoint).expanduser()
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"SAM 3 checkpoint directory does not exist: {checkpoint_path}")

            from transformers import AutoConfig
            from transformers import Sam3Processor
            from transformers import Sam3TrackerVideoProcessor
            from transformers import Sam3VideoModel

            video_config = AutoConfig.from_pretrained(checkpoint_path, local_files_only=True)
            self._set_video_image_size(video_config, model_input_size)
            logger.info("Loading SAM 3 detector and tracker from %s on %s", checkpoint_path, device)
            video_model = Sam3VideoModel.from_pretrained(
                checkpoint_path,
                config=video_config,
                local_files_only=True,
            )
            detector_processor = Sam3Processor.from_pretrained(checkpoint_path, local_files_only=True)
            tracker_processor = Sam3TrackerVideoProcessor.from_pretrained(
                checkpoint_path,
                local_files_only=True,
                target_size=model_input_size,
            )
        elif detector_processor is None or tracker_processor is None or torch_module is None:
            raise ValueError(
                "video_model, detector_processor, tracker_processor, and torch_module must be supplied together"
            )
        if not mapping_source_camera or not mapping_destination_camera:
            raise ValueError("Cross-camera mapping camera names must be non-empty")
        if mapping_source_camera == mapping_destination_camera:
            raise ValueError("Cross-camera mapping source and destination cameras must differ")
        if spatial_prompt_box_padding < 0:
            raise ValueError("spatial_prompt_box_padding must be non-negative")
        if not 0.0 < redetect_area_ratio < 1.0:
            raise ValueError("redetect_area_ratio must be between 0 and 1")
        if not 0.0 < redetect_reference_decay <= 1.0:
            raise ValueError("redetect_reference_decay must be between 0 and 1")
        if redetect_cooldown_frames < 1:
            raise ValueError("redetect_cooldown_frames must be at least 1")

        # Reuse validation and recoloring helpers from the image preprocessor,
        # while avoiding a second checkpoint load.
        super().__init__(
            checkpoint,
            prompts=prompts,
            target_rgb=target_rgb,
            device=device,
            score_threshold=score_threshold,
            mask_threshold=mask_threshold,
            alpha=alpha,
            min_component_area=min_component_area,
            model_input_size=model_input_size,
            error_policy=error_policy,
            model=video_model.detector_model,
            processor=detector_processor,
            torch_module=torch_module,
        )
        self._video_model = video_model.to(self.device)
        self._video_model.eval()
        self._tracker_model = self._video_model.tracker_model
        self._tracker_processor = tracker_processor
        self.camera_score_thresholds = camera_score_thresholds
        if cross_camera_mapping is None or isinstance(cross_camera_mapping, PolynomialCameraMapping):
            self.cross_camera_mapping = cross_camera_mapping
        else:
            self.cross_camera_mapping = PolynomialCameraMapping.from_json(cross_camera_mapping)
        self.mapping_source_camera = mapping_source_camera
        self.mapping_destination_camera = mapping_destination_camera
        self.spatial_prompt_box_padding = float(spatial_prompt_box_padding)
        self.redetect_area_ratio = float(redetect_area_ratio)
        self.redetect_reference_decay = float(redetect_reference_decay)
        self.redetect_cooldown_frames = int(redetect_cooldown_frames)
        self._sessions: dict[str, Any | None] = {}
        self._mask_area_reference: float | None = None
        self._last_redetect_frame = -self.redetect_cooldown_frames
        self._frame_index = 0

    @staticmethod
    def _set_video_image_size(video_config: Any, image_size: int) -> None:
        if image_size < 1 or image_size % 14 != 0:
            raise ValueError("model_input_size must be a positive multiple of SAM 3's 14-pixel patch size")
        feature_size = image_size // 14
        video_config.detector_config.image_size = image_size
        tracker_config = video_config.tracker_config
        tracker_config.image_size = image_size
        tracker_config.prompt_encoder_config.image_size = image_size
        tracker_config.vision_config.backbone_config.image_size = image_size
        tracker_config.vision_config.backbone_feature_sizes = [
            [feature_size * 4, feature_size * 4],
            [feature_size * 2, feature_size * 2],
            [feature_size, feature_size],
        ]
        tracker_config.memory_attention_rope_feat_sizes = [feature_size, feature_size]

    def start_episode(self) -> None:
        """Discard all temporal state before processing a new episode."""
        self._sessions = {}
        self._mask_area_reference = None
        self._last_redetect_frame = -self.redetect_cooldown_frames
        self._frame_index = 0

    def has_active_trackers(self, camera_names: Sequence[str]) -> bool:
        """Return whether every requested camera has an initialized tracker."""
        return all(self._sessions.get(camera_name) is not None for camera_name in camera_names)

    def _detect_first_frame(self, validated: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        camera_names = list(validated)
        batch_camera_names = [name for name in camera_names for _ in self.prompts]
        inputs = self._processor(
            images=[validated[name] for name in camera_names for _ in self.prompts],
            text=[prompt for _ in camera_names for prompt in self.prompts],
            return_tensors="pt",
            size={"height": self.model_input_size, "width": self.model_input_size},
        ).to(self.device)
        target_sizes = inputs["original_sizes"].detach().cpu().tolist()
        with self._torch.inference_mode():
            outputs = self._model(**inputs)
        postprocess_threshold = min(
            (self.score_threshold, *self.camera_score_thresholds.values())
        )
        results = self._processor.post_process_instance_segmentation(
            outputs,
            threshold=postprocess_threshold,
            mask_threshold=self.mask_threshold,
            target_sizes=target_sizes,
        )
        if len(results) != len(batch_camera_names):
            raise ValueError(f"SAM 3 returned {len(results)} results for {len(batch_camera_names)} inputs")

        masks = {name: np.zeros(validated[name].shape[:2], dtype=bool) for name in camera_names}
        for camera_name, result in zip(batch_camera_names, results, strict=True):
            result_masks = result.get("masks")
            if result_masks is not None:
                camera_threshold = self.camera_score_thresholds.get(
                    camera_name, self.score_threshold
                )
                if camera_threshold > postprocess_threshold:
                    result_scores = result.get("scores")
                    if result_scores is None:
                        raise ValueError(
                            "SAM 3 result has no scores for camera-specific thresholding"
                        )
                    result_masks = result_masks[result_scores >= camera_threshold]
                masks[camera_name] |= self._masks_to_numpy(result_masks, validated[camera_name].shape[:2])
        return {name: clean_mask(mask, self.min_component_area) for name, mask in masks.items()}

    def _start_tracker(self, camera_name: str, image: np.ndarray, mask: np.ndarray) -> None:
        if not mask.any():
            logger.warning("SAM 3 found no first-frame mask for %s; tracker initialization failed", camera_name)
            self._sessions[camera_name] = None
            return
        session = self._tracker_processor.init_video_session(
            video=None,
            inference_device=self.device,
            inference_state_device=self.device,
            video_storage_device="cpu",
            dtype=self._torch.float32,
        )
        inputs = self._tracker_processor(
            images=image,
            return_tensors="pt",
            size={"height": self.model_input_size, "width": self.model_input_size},
        )
        session.video_height, session.video_width = image.shape[:2]
        session.add_new_frame(inputs["pixel_values"])
        self._tracker_processor.process_new_mask_for_video_frame(
            session,
            frame_idx=0,
            obj_ids=[0],
            input_masks=[mask],
        )
        self._cache_tracker_vision_features(session, 0)
        with self._torch.inference_mode():
            self._tracker_model(inference_session=session, frame_idx=0)
        self._sessions[camera_name] = session

    def _cross_camera_spatial_prompt(
        self,
        source_mask: np.ndarray,
        *,
        source_image_shape: tuple[int, int],
        destination_image_shape: tuple[int, int],
    ) -> tuple[tuple[float, float], tuple[float, float, float, float]]:
        """Project a source mask centroid and outline into a destination point and box."""
        mapping = self.cross_camera_mapping
        if mapping is None:
            raise ValueError("No cross-camera mapping is configured")
        source_mask = np.asarray(source_mask, dtype=bool)
        if source_mask.shape != source_image_shape:
            raise ValueError(
                f"Source mask shape {source_mask.shape} does not match image {source_image_shape}"
            )
        mask_yx = np.argwhere(source_mask)
        if len(mask_yx) == 0:
            raise ValueError("Cannot project an empty source mask")

        centroid_xy = mask_yx[:, ::-1].mean(axis=0)
        boundary = source_mask & ~ndimage.binary_erosion(source_mask)
        boundary_xy = np.argwhere(boundary)[:, ::-1]
        if len(boundary_xy) > 128:
            sample_indices = np.linspace(0, len(boundary_xy) - 1, 128, dtype=np.int64)
            boundary_xy = boundary_xy[sample_indices]
        source_points = np.concatenate([centroid_xy[None], boundary_xy], axis=0)
        source_native = resize_pad_points(
            source_points,
            native_size_wh=mapping.source_size_wh,
            padded_size_wh=(source_image_shape[1], source_image_shape[0]),
            inverse=True,
        )
        destination_native = mapping.map_points(source_native)
        destination_points = resize_pad_points(
            destination_native,
            native_size_wh=mapping.destination_size_wh,
            padded_size_wh=(destination_image_shape[1], destination_image_shape[0]),
        )

        destination_height, destination_width = destination_image_shape
        valid = (
            np.isfinite(destination_points).all(axis=1)
            & (destination_points[:, 0] >= 0)
            & (destination_points[:, 0] <= destination_width - 1)
            & (destination_points[:, 1] >= 0)
            & (destination_points[:, 1] <= destination_height - 1)
        )
        if not valid[0]:
            raise ValueError("Projected source-mask centroid falls outside the destination image")
        projected_boundary = destination_points[1:][valid[1:]]
        if len(projected_boundary) < 2:
            raise ValueError("Too few projected boundary points to construct a destination box")

        box_min = projected_boundary.min(axis=0) - self.spatial_prompt_box_padding
        box_max = projected_boundary.max(axis=0) + self.spatial_prompt_box_padding
        box_min = np.maximum(box_min, (0.0, 0.0))
        box_max = np.minimum(box_max, (destination_width - 1.0, destination_height - 1.0))
        if np.any(box_max <= box_min):
            raise ValueError(f"Projected destination box is degenerate: {box_min}, {box_max}")
        point = tuple(float(value) for value in destination_points[0])
        box = tuple(float(value) for value in np.concatenate([box_min, box_max]))
        return point, box

    def _start_tracker_with_spatial_prompt(
        self,
        camera_name: str,
        image: np.ndarray,
        point_xy: tuple[float, float],
        box_xyxy: tuple[float, float, float, float],
    ) -> np.ndarray:
        """Start a tracker from a positive point and box and return its first-frame mask."""
        session = self._tracker_processor.init_video_session(
            video=None,
            inference_device=self.device,
            inference_state_device=self.device,
            video_storage_device="cpu",
            dtype=self._torch.float32,
        )
        inputs = self._tracker_processor(
            images=image,
            return_tensors="pt",
            size={"height": self.model_input_size, "width": self.model_input_size},
        )
        session.add_new_frame(inputs["pixel_values"])
        self._tracker_processor.process_new_points_or_boxes_for_video_frame(
            session,
            frame_idx=0,
            obj_ids=[0],
            input_points=[[[[point_xy[0], point_xy[1]]]]],
            input_labels=[[[1]]],
            input_boxes=[[[*box_xyxy]]],
            original_size=image.shape[:2],
        )
        self._cache_tracker_vision_features(session, 0)
        with self._torch.inference_mode():
            outputs = self._tracker_model(inference_session=session, frame_idx=0)
        output_masks = self._tracker_processor.post_process_masks(
            outputs.pred_masks.unsqueeze(0),
            inputs["original_sizes"],
            mask_threshold=0.0,
        )
        mask = clean_mask(
            self._masks_to_numpy(output_masks[0].squeeze(1), image.shape[:2]),
            self.min_component_area,
        )
        self._sessions[camera_name] = session
        return mask

    def _initialize_first_frame_trackers(
        self, validated: dict[str, np.ndarray], masks: dict[str, np.ndarray]
    ) -> dict[str, np.ndarray]:
        """Initialize each camera, using a projected prompt for the configured destination."""
        destination_prompt: tuple[
            tuple[float, float], tuple[float, float, float, float]
        ] | None = None
        mapping = self.cross_camera_mapping
        if mapping is not None:
            source_name = self.mapping_source_camera
            destination_name = self.mapping_destination_camera
            if source_name not in validated or destination_name not in validated:
                logger.warning(
                    "Cross-camera mapping needs %s and %s, but received %s; using detector masks",
                    source_name,
                    destination_name,
                    sorted(validated),
                )
            elif not masks[source_name].any():
                logger.warning(
                    "SAM 3 found no first-frame mask for mapping source %s; using the %s detector mask",
                    source_name,
                    destination_name,
                )
            else:
                try:
                    destination_prompt = self._cross_camera_spatial_prompt(
                        masks[source_name],
                        source_image_shape=validated[source_name].shape[:2],
                        destination_image_shape=validated[destination_name].shape[:2],
                    )
                except Exception:
                    logger.exception(
                        "Failed to project %s mask into %s; using the destination detector mask",
                        source_name,
                        destination_name,
                    )

        for camera_name, image in validated.items():
            if camera_name == self.mapping_destination_camera and destination_prompt is not None:
                try:
                    prompted_mask = self._start_tracker_with_spatial_prompt(
                        camera_name, image, *destination_prompt
                    )
                    if prompted_mask.any():
                        masks[camera_name] = prompted_mask
                    else:
                        logger.warning(
                            "The projected spatial prompt produced no %s mask; using its detector mask",
                            camera_name,
                        )
                        self._start_tracker(camera_name, image, masks[camera_name])
                except Exception:
                    logger.exception(
                        "SAM 3 spatial prompt failed for %s; using its detector mask", camera_name
                    )
                    self._start_tracker(camera_name, image, masks[camera_name])
            else:
                self._start_tracker(camera_name, image, masks[camera_name])
        return masks

    def _cache_tracker_vision_features(self, session: Any, frame_idx: int) -> None:
        """Cache shared SAM3 backbone features without running text detection."""
        pixel_values = session.get_frame(frame_idx).unsqueeze(0).to(self.device)
        with self._torch.inference_mode():
            vision_embeds = self._model.get_vision_features(pixel_values=pixel_values)
            vision_feats, vision_pos_embeds = self._video_model.get_vision_features_for_tracker(
                vision_embeds=vision_embeds
            )
        session.cache.cache_vision_features(
            frame_idx,
            {"vision_feats": vision_feats, "vision_pos_embeds": vision_pos_embeds},
        )

    def _track_frame(self, camera_name: str, image: np.ndarray) -> np.ndarray:
        session = self._sessions[camera_name]
        if session is None:
            return np.zeros(image.shape[:2], dtype=bool)
        inputs = self._tracker_processor(
            images=image,
            return_tensors="pt",
            size={"height": self.model_input_size, "width": self.model_input_size},
        )
        pixel_values = inputs["pixel_values"].to(self.device)
        frame_idx = session.add_new_frame(pixel_values)
        self._cache_tracker_vision_features(session, frame_idx)
        with self._torch.inference_mode():
            outputs = self._tracker_model(inference_session=session, frame_idx=frame_idx)
        masks = self._tracker_processor.post_process_masks(
            outputs.pred_masks.unsqueeze(0),
            inputs["original_sizes"],
            mask_threshold=0.0,
        )
        return clean_mask(
            self._masks_to_numpy(masks[0].squeeze(1), image.shape[:2]),
            self.min_component_area,
        )

    def _redetect_destination(self, validated: dict[str, np.ndarray]) -> np.ndarray | None:
        """Run text detection once and restart only the destination tracker."""
        destination_name = self.mapping_destination_camera
        if destination_name not in validated:
            return None
        detected_masks = self._detect_first_frame(validated)
        old_session = self._sessions[destination_name]
        mapping = self.cross_camera_mapping
        source_name = self.mapping_source_camera
        if mapping is not None and source_name in validated and detected_masks[source_name].any():
            try:
                prompt = self._cross_camera_spatial_prompt(
                    detected_masks[source_name],
                    source_image_shape=validated[source_name].shape[:2],
                    destination_image_shape=validated[destination_name].shape[:2],
                )
                mask = self._start_tracker_with_spatial_prompt(
                    destination_name, validated[destination_name], *prompt
                )
                if mask.any():
                    return mask
            except Exception:
                logger.exception(
                    "Mapped SAM 3 re-detection failed for %s on frame %d",
                    destination_name,
                    self._frame_index,
                )
            self._sessions[destination_name] = old_session

        destination_mask = detected_masks[destination_name]
        if destination_mask.any():
            self._start_tracker(destination_name, validated[destination_name], destination_mask)
            return destination_mask
        self._sessions[destination_name] = old_session
        logger.warning(
            "SAM 3 re-detection found no %s mask on frame %d; keeping the existing tracker",
            destination_name,
            self._frame_index,
        )
        return None

    def _maybe_redetect_shrunken_mask(
        self, validated: dict[str, np.ndarray], masks: dict[str, np.ndarray]
    ) -> dict[str, np.ndarray]:
        """Re-detect only after a substantial destination-mask area decrease."""
        destination_name = self.mapping_destination_camera
        if destination_name not in masks:
            return masks
        current_area = float(np.count_nonzero(masks[destination_name]))
        reference = self._mask_area_reference
        if reference is None:
            self._mask_area_reference = current_area
            return masks

        cooldown_elapsed = self._frame_index - self._last_redetect_frame
        significantly_smaller = current_area < reference * self.redetect_area_ratio
        if significantly_smaller and cooldown_elapsed >= self.redetect_cooldown_frames:
            logger.info(
                "SAM 3 %s mask shrank from reference %.0f to %.0f pixels on frame %d; re-detecting",
                destination_name,
                reference,
                current_area,
                self._frame_index,
            )
            recovered_mask = self._redetect_destination(validated)
            self._last_redetect_frame = self._frame_index
            if recovered_mask is not None:
                masks[destination_name] = recovered_mask
                current_area = float(np.count_nonzero(recovered_mask))
                reference = current_area

        if current_area >= reference * self.redetect_area_ratio:
            self._mask_area_reference = max(
                current_area, reference * self.redetect_reference_decay
            )
        return masks

    def preprocess(self, images: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
        validated = self._validate_images(images)
        originals = {name: image.copy() for name, image in validated.items()}
        if not validated:
            return originals
        if self._frame_index > 0 and set(validated) != set(self._sessions):
            raise ValueError(
                f"Camera keys changed within an episode: {set(self._sessions)} -> {set(validated)}"
            )

        started = time.perf_counter()
        try:
            if self._frame_index == 0:
                masks = self._detect_first_frame(validated)
                masks = self._initialize_first_frame_trackers(validated, masks)
            else:
                masks = {
                    camera_name: self._track_frame(camera_name, image)
                    for camera_name, image in validated.items()
                }
            masks = self._maybe_redetect_shrunken_mask(validated, masks)
        except Exception:
            self.last_elapsed_ms = (time.perf_counter() - started) * 1000.0
            if self.error_policy == "raise":
                raise
            logger.exception("SAM 3 episode tracking failed; using the original images")
            self._frame_index += 1
            return originals

        output = {
            camera_name: recolor_masked_region(
                image,
                masks[camera_name],
                target_rgb=self.target_rgb,
                alpha=self.alpha,
            )
            for camera_name, image in validated.items()
        }
        self._frame_index += 1
        self.last_elapsed_ms = (time.perf_counter() - started) * 1000.0
        return output

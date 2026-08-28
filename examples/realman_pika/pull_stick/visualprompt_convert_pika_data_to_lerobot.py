"""Convert pull-stick Pika recordings to keypoint visual-prompt LeRobot data.

The source recordings are split by ``convert_pika_data_to_lerobot.py``. SAM 3
identifies the white cap held by the gripper at a native-resolution grasp frame.
CoTracker then tracks several points on that cap both forward and backward in
the fisheye and wrist RGB streams. It draws one
synthetic magenta keypoint on the white end cap of each tracked stick. The exact
RGB value (255, 0, 255) is deliberately chosen as a
high-visibility value that natural camera frames do not contain.

Example:
uv run --project examples/realman_pika --no-sync python \
  examples/realman_pika/pull_stick/visualprompt_convert_pika_data_to_lerobot.py \
  --debug-video-dir /tmp/pull_stick_keypoints --debug-overwrite
"""

from __future__ import annotations

# Reuse the reference converters internal helpers intentionally.
# ruff: noqa: SLF001

from collections.abc import Mapping
import dataclasses
import gc
import importlib.util
import json
import logging
from pathlib import Path
import shutil
import sys
import time
from typing import Any

import h5py
import numpy as np
from scipy import ndimage
from PIL import Image
from PIL import ImageDraw
import torch
from tqdm.auto import tqdm
import tyro

from cotracker.predictor import CoTrackerPredictor

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent
sys.path.insert(0, str(PARENT))


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


pull = _load_module("pull_stick_converter", HERE / "convert_pika_data_to_lerobot.py")
visual = _load_module("pika_visualprompt_converter", PARENT / "visualprompt_convert_pika_data_to_lerobot.py")
# The shared writer now feeds native-resolution frames to this module's SAM3 preprocessor.
visual._read_rgb = pull.base._read_native_rgb

from image_preprocessing import Sam3EpisodeTrackerPreprocessor  # noqa: E402

LOGGER = logging.getLogger(__name__)
DEFAULT_REPO_ID = "Zehao123/pika_pull_stick_224_224_keypoint"
DEFAULT_CHECKPOINT = HERE.parents[3] / "foundation_models" / "SAM3"
DEFAULT_COTRACKER_CHECKPOINT = (
    HERE.parents[3] / "foundation_models" / "CoTracker3" / "scaled_offline.pth"
)
DEFAULT_PROMPTS = ("unused: spatial prompts only",)


@dataclasses.dataclass
class Sam3Config:
    checkpoint: Path = DEFAULT_CHECKPOINT
    prompts: tuple[str, ...] = DEFAULT_PROMPTS
    device: str = "cuda"
    score_threshold: float = 0.25
    fisheye_score_threshold: float = 0.20
    mask_threshold: float = 0.30
    min_component_area: int = 20
    model_input_size: int = 644
    redetect_area_ratio: float = 0.35
    redetect_reference_decay: float = 0.98
    redetect_cooldown_frames: int = 15
    fisheye_gripper_center_xy: tuple[float, float] = (0.50, 0.71)
    wrist_gripper_center_xy: tuple[float, float] = (0.50, 0.74)
    search_half_width: float = 0.12
    search_half_height: float = 0.15
    prompt_box_half_width: float = 0.065
    prompt_box_above: float = 0.055
    prompt_box_below: float = 0.28
    min_white_component_area: int = 35
    max_white_component_area: int = 2500
    max_prompt_distance: float = 0.13
    min_mask_cap_overlap: float = 0.20
    max_tip_motion_pixels: float = 40.0
    max_tip_misses: int = 3


@dataclasses.dataclass
class CoTrackerConfig:
    checkpoint: Path = DEFAULT_COTRACKER_CHECKPOINT
    device: str = "cuda"
    query_points: int = 9
    min_visible_points: int = 3
    visibility_threshold: float = 0.5
    max_point_deviation_pixels: float = 32.0
    max_frame_motion_pixels: float = 80.0
    anchor_search_radius: int = 8
    max_consecutive_misses: int = 3


@dataclasses.dataclass
class KeypointConfig:
    rgb: tuple[int, int, int] = (255, 0, 255)
    radius: int = 5
    outline_rgb: tuple[int, int, int] = (255, 255, 255)
    outline_width: int = 2
    white_min_value: int = 170
    white_max_chroma: int = 55
    white_min_area: int = 3
    tip_search_dilation: int = 4


@dataclasses.dataclass
class Args:
    data_dir: Path = pull.DEFAULT_DATA_DIR
    repo_id: str = DEFAULT_REPO_ID
    task_prompt: str = pull.TASK_PROMPT
    rewrite_existing_prompt: bool = False
    rewrite_task_index: int = 0
    max_recordings: int | None = None
    start_episode: int = 0
    max_episodes: int | None = None
    sam_batch_size: int = 8
    push_to_hub: bool = False
    test_mode: bool = False
    debug_video_dir: Path | None = None
    debug_overwrite: bool = False
    strict_splitting: bool = False
    split: pull.SplitConfig = dataclasses.field(default_factory=pull.SplitConfig)
    sam3: Sam3Config = dataclasses.field(default_factory=Sam3Config)
    cotracker: CoTrackerConfig = dataclasses.field(default_factory=CoTrackerConfig)
    keypoint: KeypointConfig = dataclasses.field(default_factory=KeypointConfig)


def _white_tip_keypoint(
    image: np.ndarray,
    mask: np.ndarray,
    config: KeypointConfig,
) -> tuple[int, int] | None:
    """Locate the white end cap nearest an endpoint of the stick mask."""
    mask = np.asarray(mask, dtype=bool)
    mask_points_yx = np.argwhere(mask)
    if len(mask_points_yx) == 0:
        return None

    rgb = np.asarray(image, dtype=np.int16)
    white = (rgb.min(axis=-1) >= config.white_min_value) & (
        np.ptp(rgb, axis=-1) <= config.white_max_chroma
    )
    search_mask = ndimage.binary_dilation(
        mask,
        iterations=config.tip_search_dilation,
    )
    labels, component_count = ndimage.label(white & search_mask)
    if component_count == 0:
        return None

    mask_centroid = mask_points_yx.mean(axis=0)
    centered = mask_points_yx - mask_centroid
    if len(mask_points_yx) > 1:
        covariance = centered.T @ centered
        _, eigenvectors = np.linalg.eigh(covariance)
        major_axis = eigenvectors[:, -1]
        axis_extent = max(float(np.max(np.abs(centered @ major_axis))), 1.0)
    else:
        major_axis = np.array((1.0, 0.0))
        axis_extent = 1.0

    best_score = None
    best_points_yx = None
    for label_index in range(1, component_count + 1):
        points_yx = np.argwhere(labels == label_index)
        if len(points_yx) < config.white_min_area:
            continue
        component_centroid = points_yx.mean(axis=0)
        terminality = abs(float((component_centroid - mask_centroid) @ major_axis)) / axis_extent
        score = (terminality, len(points_yx))
        if best_score is None or score > best_score:
            best_score = score
            best_points_yx = points_yx

    if best_points_yx is None:
        return None
    centroid = best_points_yx.mean(axis=0)
    point_yx = best_points_yx[
        np.argmin(np.square(best_points_yx - centroid).sum(axis=1))
    ]
    return int(point_yx[1]), int(point_yx[0])


def _sample_component_points(component: np.ndarray, count: int) -> np.ndarray:
    """Sample well-spread native-resolution xy points from one white cap."""
    points_yx = np.argwhere(np.asarray(component, dtype=bool))
    if count < 1 or len(points_yx) < count:
        raise ValueError(f"Cannot sample {count} points from {len(points_yx)} pixels")
    centroid = points_yx.mean(axis=0)
    chosen = [int(np.argmin(np.square(points_yx - centroid).sum(axis=1)))]
    minimum_distance = np.square(points_yx - points_yx[chosen[0]]).sum(axis=1)
    for _ in range(1, count):
        next_index = int(np.argmax(minimum_distance))
        chosen.append(next_index)
        distance = np.square(points_yx - points_yx[next_index]).sum(axis=1)
        minimum_distance = np.minimum(minimum_distance, distance)
    return points_yx[chosen, ::-1].astype(np.float32)


def _validate_rgb(value: tuple[int, int, int], name: str) -> None:
    if len(value) != 3 or any(channel < 0 or channel > 255 for channel in value):
        raise ValueError(f"{name} must contain three integer values in [0, 255], got {value}")


def _draw_keypoint_at(
    image: np.ndarray,
    point: tuple[int, int] | None,
    config: KeypointConfig,
) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[-1] != 3 or image.dtype != np.uint8:
        raise ValueError(f"Expected HWC uint8 RGB image, got {image.shape}/{image.dtype}")
    target_rgb = np.asarray(config.rgb, dtype=np.uint8)
    if np.any(np.all(image == target_rgb, axis=-1)):
        raise ValueError(
            f"Selected keypoint RGB {config.rgb} already occurs in a source image; "
            "choose another --keypoint.rgb"
        )
    if point is None:
        return image.copy()

    rendered = Image.fromarray(image)
    draw = ImageDraw.Draw(rendered)
    x, y = point
    outer_radius = config.radius + config.outline_width
    if config.outline_width:
        draw.ellipse(
            (x - outer_radius, y - outer_radius, x + outer_radius, y + outer_radius),
            fill=config.outline_rgb,
        )
    draw.ellipse(
        (x - config.radius, y - config.radius, x + config.radius, y + config.radius),
        fill=config.rgb,
    )
    return np.asarray(rendered, dtype=np.uint8)


def _draw_keypoint(image: np.ndarray, mask: np.ndarray, config: KeypointConfig) -> np.ndarray:
    if np.asarray(mask).shape != np.asarray(image).shape[:2]:
        raise ValueError(f"Mask shape {np.asarray(mask).shape} does not match image {np.asarray(image).shape[:2]}")
    return _draw_keypoint_at(image, _white_tip_keypoint(image, mask, config), config)


def _resize_point_with_pad(
    point: tuple[int, int],
    source_shape: tuple[int, int],
    output_size: int,
) -> tuple[int, int]:
    source_height, source_width = source_shape
    ratio = max(source_width / output_size, source_height / output_size)
    resized_height = int(source_height / ratio)
    resized_width = int(source_width / ratio)
    pad_y = (output_size - resized_height) // 2
    pad_x = (output_size - resized_width) // 2
    x, y = point
    return int(round(x / ratio)) + pad_x, int(round(y / ratio)) + pad_y


def _render_keypoint_output(
    image: np.ndarray,
    mask: np.ndarray,
    config: KeypointConfig,
) -> np.ndarray:
    """Find the white tip natively, then resize and render the 224px output."""
    image = np.asarray(image)
    point = _white_tip_keypoint(image, mask, config)
    output_size = pull.base.IMAGE_SIZE
    resized_image = pull.base.image_tools.resize_with_pad(image, output_size, output_size)
    resized_point = None if point is None else _resize_point_with_pad(
        point,
        image.shape[:2],
        output_size,
    )
    return _draw_keypoint_at(resized_image, resized_point, config)


class Sam3KeypointPreprocessor(Sam3EpisodeTrackerPreprocessor):
    """SAM3 tracker initialized from the white cap between the gripper jaws."""

    def __init__(
        self, *args: Any, keypoint: KeypointConfig, spatial: Sam3Config, **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)
        self.keypoint = keypoint
        self.spatial = spatial
        self._tip_points: dict[str, tuple[float, float]] = {}
        self._tip_misses: dict[str, int] = {}
        self._lost_cameras: set[str] = set()

    def start_episode(self) -> None:
        super().start_episode()
        self._tip_points = {}
        self._tip_misses = {}
        self._lost_cameras = set()

    def _gripper_spatial_prompt(
        self, camera_name: str, image: np.ndarray
    ) -> tuple[tuple[float, float], tuple[float, float, float, float], np.ndarray]:
        """Select the white cap between the jaws and construct a narrow stick box."""
        h, w = image.shape[:2]
        center = {
            "image": self.spatial.fisheye_gripper_center_xy,
            "wrist_image": self.spatial.wrist_gripper_center_xy,
        }[camera_name]
        expected = np.array((center[0] * w, center[1] * h))
        rgb = image.astype(np.int16)
        white = (rgb.min(axis=-1) >= self.keypoint.white_min_value) & (
            np.ptp(rgb, axis=-1) <= self.keypoint.white_max_chroma
        )
        roi = np.zeros((h, w), dtype=bool)
        x0 = max(0, int((center[0] - self.spatial.search_half_width) * w))
        x1 = min(w, int((center[0] + self.spatial.search_half_width) * w))
        y0 = max(0, int((center[1] - self.spatial.search_half_height) * h))
        y1 = min(h, int((center[1] + self.spatial.search_half_height) * h))
        roi[y0:y1, x0:x1] = True
        # Separate touching round caps without broadening the gripper ROI.
        separated_white = ndimage.binary_erosion(white & roi, iterations=2)
        labels, count = ndimage.label(separated_white)
        candidates = []
        diagonal = float(np.hypot(w, h))
        for label_index in range(1, count + 1):
            component = labels == label_index
            points = np.argwhere(component)
            area = len(points)
            if not self.spatial.min_white_component_area <= area <= self.spatial.max_white_component_area:
                continue
            span = np.ptp(points, axis=0)
            if np.any(span < 4) or np.any(span > 75):
                continue
            point = points[:, ::-1].mean(axis=0)
            distance = float(np.linalg.norm(point - expected) / diagonal)
            candidates.append((distance - 0.012 * np.log(area), distance, point, component))
        if not candidates:
            raise RuntimeError(f"{camera_name}: no white cap candidate between the jaws")
        _score, distance, point, component = min(candidates, key=lambda item: item[0])
        if distance > self.spatial.max_prompt_distance:
            raise RuntimeError(
                f"{camera_name}: white cap too far from gripper "
                f"({distance:.3f} > {self.spatial.max_prompt_distance:.3f})"
            )
        x, y = float(point[0]), float(point[1])
        box = (
            max(0.0, x - self.spatial.prompt_box_half_width * w),
            max(0.0, y - self.spatial.prompt_box_above * h),
            min(w - 1.0, x + self.spatial.prompt_box_half_width * w),
            min(h - 1.0, y + self.spatial.prompt_box_below * h),
        )
        return (x, y), box, component

    def _detect_first_frame(self, validated: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Use native-resolution point/box prompts and completely bypass text detection."""
        masks = {}
        for camera_name, image in validated.items():
            point, box, cap = self._gripper_spatial_prompt(camera_name, image)
            mask = self._start_tracker_with_spatial_prompt(camera_name, image, point, box)
            overlap = float(np.count_nonzero(mask & cap) / np.count_nonzero(cap))
            if not mask.any() or overlap < self.spatial.min_mask_cap_overlap:
                self._sessions[camera_name] = None
                raise RuntimeError(
                    f"{camera_name}: SAM3 rejected spatial cap (overlap={overlap:.3f})"
                )
            LOGGER.info(
                "%s spatial SAM3 point=(%.1f, %.1f), box=%s, overlap=%.3f",
                camera_name, point[0], point[1], box, overlap,
            )
            self._tip_points[camera_name] = point
            masks[camera_name] = mask
        return masks

    def detect_anchor_queries(
        self,
        images: Mapping[str, np.ndarray],
        query_count: int,
    ) -> dict[str, np.ndarray]:
        """Confirm the selected cap with SAM3 and sample CoTracker queries on it."""
        validated = self._validate_images(images)
        self.start_episode()
        masks = self._detect_first_frame(validated)
        queries = {}
        for camera_name, image in validated.items():
            _point, _box, cap = self._gripper_spatial_prompt(camera_name, image)
            confirmed_cap = cap & ndimage.binary_dilation(masks[camera_name], iterations=1)
            if np.count_nonzero(confirmed_cap) < query_count:
                raise RuntimeError(
                    f"{camera_name}: SAM3-confirmed white cap has only "
                    f"{np.count_nonzero(confirmed_cap)} pixels"
                )
            queries[camera_name] = _sample_component_points(confirmed_cap, query_count)
        return queries

    def _maybe_redetect_shrunken_mask(
        self, validated: dict[str, np.ndarray], masks: dict[str, np.ndarray]
    ) -> dict[str, np.ndarray]:
        # Text re-detection may jump to another identical stick.
        del validated
        return masks

    def _tracked_white_cap(
        self,
        camera_name: str,
        image: np.ndarray,
        mask: np.ndarray,
    ) -> tuple[int, int] | None:
        previous = self._tip_points.get(camera_name)
        if previous is None or camera_name in self._lost_cameras:
            return None
        rgb = image.astype(np.int16)
        white = (rgb.min(axis=-1) >= self.keypoint.white_min_value) & (
            np.ptp(rgb, axis=-1) <= self.keypoint.white_max_chroma
        )
        search = ndimage.binary_dilation(mask, iterations=self.keypoint.tip_search_dilation)
        labels, count = ndimage.label(white & search)
        candidates = []
        for label_index in range(1, count + 1):
            points = np.argwhere(labels == label_index)
            if len(points) < self.keypoint.white_min_area:
                continue
            point = points[:, ::-1].mean(axis=0)
            distance = float(np.linalg.norm(point - np.asarray(previous)))
            candidates.append((distance, point))
        distance, point = min(candidates, key=lambda item: item[0]) if candidates else (float("inf"), None)
        if distance > self.spatial.max_tip_motion_pixels:
            misses = self._tip_misses.get(camera_name, 0) + 1
            self._tip_misses[camera_name] = misses
            if misses >= self.spatial.max_tip_misses:
                self._lost_cameras.add(camera_name)
                LOGGER.warning("%s white cap lost after %d rejected frames", camera_name, misses)
            return None
        self._tip_misses[camera_name] = 0
        tracked = (float(point[0]), float(point[1]))
        self._tip_points[camera_name] = tracked
        return int(round(tracked[0])), int(round(tracked[1]))

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
            LOGGER.exception("SAM3 keypoint tracking failed; using original images")
            self._frame_index += 1
            return originals

        output = {}
        for camera_name, image in validated.items():
            point = self._tracked_white_cap(camera_name, image, masks[camera_name])
            resized = pull.base.image_tools.resize_with_pad(
                image, pull.base.IMAGE_SIZE, pull.base.IMAGE_SIZE
            )
            resized_point = (
                None
                if point is None
                else _resize_point_with_pad(point, image.shape[:2], pull.base.IMAGE_SIZE)
            )
            output[camera_name] = _draw_keypoint_at(resized, resized_point, self.keypoint)
        self._frame_index += 1
        self.last_elapsed_ms = (time.perf_counter() - started) * 1000.0
        return output


class _DatasetWithDebugVideo:
    """Forward frames to LeRobot while previewing the processed camera images."""

    def __init__(self, dataset: Any, writer: Any | None, task_prompt: str) -> None:
        self.dataset = dataset
        self.writer = writer
        self.task_prompt = task_prompt

    def add_frame(self, frame: dict[str, Any]) -> None:
        self.dataset.add_frame(frame)
        if self.writer is not None:
            self.writer.add_frame(
                pull.base._make_debug_frame(frame["image"], frame["wrist_image"], self.task_prompt)
            )


def _make_preprocessor(config: Sam3Config, keypoint: KeypointConfig) -> Sam3KeypointPreprocessor:
    return Sam3KeypointPreprocessor(
        config.checkpoint,
        prompts=config.prompts,
        target_rgb=keypoint.rgb,
        device=config.device,
        score_threshold=config.score_threshold,
        camera_score_thresholds={"image": config.fisheye_score_threshold},
        mask_threshold=config.mask_threshold,
        min_component_area=config.min_component_area,
        model_input_size=config.model_input_size,
        error_policy="raise",
        redetect_area_ratio=config.redetect_area_ratio,
        redetect_reference_decay=config.redetect_reference_decay,
        redetect_cooldown_frames=config.redetect_cooldown_frames,
        keypoint=keypoint,
        spatial=config,
    )


class CoTrackerKeypointTracker:
    """Track a fixed set of native-resolution points without re-detection."""

    def __init__(self, config: CoTrackerConfig) -> None:
        self.config = config
        if not config.checkpoint.is_file():
            raise FileNotFoundError(
                f"CoTracker checkpoint not found: {config.checkpoint}. "
                "Pass --cotracker.checkpoint PATH."
            )
        self.device = torch.device(config.device)
        self.model = CoTrackerPredictor(
            checkpoint=str(config.checkpoint),
            offline=True,
            v2=False,
            window_len=60,
        ).to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def track(
        self,
        frames: list[np.ndarray],
        anchor_local_index: int,
        query_points_xy: np.ndarray,
    ) -> list[tuple[int, int] | None]:
        if not frames:
            return []
        shape = frames[0].shape
        if any(frame.shape != shape for frame in frames):
            raise ValueError("Camera resolution changed within an episode")
        video = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2)[None]
        video = video.to(device=self.device, dtype=torch.float32, non_blocking=True)
        xy = torch.as_tensor(query_points_xy, device=self.device, dtype=torch.float32)
        times = torch.full(
            (len(xy), 1),
            float(anchor_local_index),
            device=self.device,
            dtype=torch.float32,
        )
        queries = torch.cat((times, xy), dim=1)[None]
        tracks, visibility = self.model(
            video,
            queries=queries,
            backward_tracking=True,
        )
        tracks_np = tracks[0].detach().float().cpu().numpy()
        visibility_np = visibility[0].detach().float().cpu().numpy()
        del video, queries, tracks, visibility
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        result: list[tuple[int, int] | None] = []
        previous_point: np.ndarray | None = None
        previous_index: int | None = None
        height, width = shape[:2]
        for frame_index, (frame_tracks, frame_visibility) in enumerate(
            zip(tracks_np, visibility_np, strict=True)
        ):
            visible = frame_visibility >= self.config.visibility_threshold
            candidates = frame_tracks[visible]
            if len(candidates) < self.config.min_visible_points:
                result.append(None)
                continue
            median = np.median(candidates, axis=0)
            coherent = candidates[
                np.linalg.norm(candidates - median, axis=1)
                <= self.config.max_point_deviation_pixels
            ]
            if len(coherent) < self.config.min_visible_points:
                result.append(None)
                continue
            point = np.median(coherent, axis=0)
            if not (0 <= point[0] < width and 0 <= point[1] < height):
                result.append(None)
                continue
            if previous_point is not None and previous_index is not None:
                allowed_motion = self.config.max_frame_motion_pixels * max(
                    1, frame_index - previous_index
                )
                if np.linalg.norm(point - previous_point) > allowed_motion:
                    result.append(None)
                    continue
            previous_point = point
            previous_index = frame_index
            result.append((int(round(point[0])), int(round(point[1]))))
        self._suppress_reacquisition(result, anchor_local_index)
        return result

    def _suppress_reacquisition(
        self,
        points: list[tuple[int, int] | None],
        anchor_local_index: int,
    ) -> None:
        """Permanently stop a direction after a sustained tracking loss."""
        directions = (
            range(anchor_local_index + 1, len(points)),
            range(anchor_local_index - 1, -1, -1),
        )
        for indices in directions:
            misses = 0
            lost = False
            for frame_index in indices:
                if lost:
                    points[frame_index] = None
                    continue
                if points[frame_index] is None:
                    misses += 1
                    if misses >= self.config.max_consecutive_misses:
                        lost = True
                else:
                    misses = 0


def _read_native_camera_frames(
    episode_dir: Path,
    file: h5py.File,
    key: str,
    start: int,
    end: int,
) -> list[np.ndarray]:
    return [
        pull.base._read_native_rgb(episode_dir, file[key][source_index])
        for source_index in range(start, end)
    ]


def _detect_anchor_queries(
    initializer: Sam3KeypointPreprocessor,
    episode_dir: Path,
    file: h5py.File,
    source_start: int,
    episode_length: int,
    preferred_local_index: int,
    config: CoTrackerConfig,
) -> tuple[int, dict[str, np.ndarray]] | None:
    low = max(0, preferred_local_index - config.anchor_search_radius)
    high = min(episode_length, preferred_local_index + config.anchor_search_radius + 1)
    candidates = sorted(
        range(low, high),
        key=lambda index: (abs(index - preferred_local_index), index),
    )
    for local_index in candidates:
        source_index = source_start + local_index
        images = {
            "image": pull.base._read_native_rgb(
                episode_dir, file[pull.base.FISHEYE_KEY][source_index]
            ),
            "wrist_image": pull.base._read_native_rgb(
                episode_dir, file[pull.base.DEPTH_CAMERA_RGB_KEY][source_index]
            ),
        }
        try:
            queries = initializer.detect_anchor_queries(
                images, config.query_points
            )
        except Exception as error:
            LOGGER.warning(
                "%s source frame %d rejected as SAM3 anchor: %s",
                episode_dir.name,
                source_index,
                error,
            )
            continue
        return local_index, queries
    return None


def _write_cotracker_episode_frames(
    dataset: Any,
    episode_dir: Path,
    file: h5py.File,
    state: np.ndarray,
    source_start: int,
    anchor_local_index: int,
    queries_by_camera: dict[str, np.ndarray],
    tracker: CoTrackerKeypointTracker,
    keypoint: KeypointConfig,
    task_prompt: str,
    output_name: str,
) -> bool:
    camera_keys = {
        "image": pull.base.FISHEYE_KEY,
        "wrist_image": pull.base.DEPTH_CAMERA_RGB_KEY,
    }
    processed: dict[str, list[np.ndarray]] = {}
    episode_length = len(state)
    for camera_name, hdf5_key in camera_keys.items():
        frames = _read_native_camera_frames(
            episode_dir,
            file,
            hdf5_key,
            source_start,
            source_start + episode_length,
        )
        points = tracker.track(
            frames,
            anchor_local_index,
            queries_by_camera[camera_name],
        )
        visible_count = sum(point is not None for point in points)
        tqdm.write(
            f"{output_name}: CoTracker {camera_name} visible on "
            f"{visible_count}/{episode_length} frames"
        )
        processed[camera_name] = []
        for frame, point in zip(frames, points, strict=True):
            resized = pull.base.image_tools.resize_with_pad(
                frame, pull.base.IMAGE_SIZE, pull.base.IMAGE_SIZE
            )
            resized_point = (
                None
                if point is None
                else _resize_point_with_pad(
                    point, frame.shape[:2], pull.base.IMAGE_SIZE
                )
            )
            processed[camera_name].append(
                _draw_keypoint_at(resized, resized_point, keypoint)
            )
        del frames, points
        gc.collect()

    for local_index in tqdm(
        range(episode_length),
        desc=output_name,
        unit="frame",
        leave=False,
    ):
        dataset.add_frame(
            {
                "image": processed["image"][local_index],
                "wrist_image": processed["wrist_image"][local_index],
                "state": state[local_index],
                "actions": state[local_index].copy(),
                "task": task_prompt,
            }
        )
    return True


def _prepare_debug_directory(path: Path | None, overwrite: bool) -> Path | None:
    if path is None:
        return None
    output = path.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    existing = list(output.glob("*.mp4"))
    if existing and not overwrite:
        raise FileExistsError(
            f"{len(existing)} MP4 files already exist in {output}; pass --debug-overwrite"
        )
    if overwrite:
        for video in existing:
            video.unlink()
    return output


def _validate_args(args: Args) -> None:
    if not args.repo_id.strip():
        raise ValueError("--repo-id must not be empty")
    if not args.task_prompt.strip():
        raise ValueError("--task-prompt must not be empty")
    if args.rewrite_task_index < 0:
        raise ValueError("--rewrite-task-index must be non-negative")
    if args.max_recordings is not None and args.max_recordings < 1:
        raise ValueError("--max-recordings must be positive")
    if args.start_episode < 0:
        raise ValueError("--start-episode must be non-negative")
    if args.max_episodes is not None and args.max_episodes < 1:
        raise ValueError("--max-episodes must be positive")
    if args.sam_batch_size < 1:
        raise ValueError("--sam-batch-size must be positive")
    for center in (
        args.sam3.fisheye_gripper_center_xy,
        args.sam3.wrist_gripper_center_xy,
    ):
        if len(center) != 2 or any(not 0.0 <= value <= 1.0 for value in center):
            raise ValueError("SAM3 gripper centers must be normalized xy pairs")
    if args.sam3.max_tip_motion_pixels <= 0 or args.sam3.max_tip_misses < 1:
        raise ValueError("SAM3 tip motion and miss limits must be positive")
    if args.cotracker.query_points < 3:
        raise ValueError("cotracker.query_points must be at least 3")
    if not 1 <= args.cotracker.min_visible_points <= args.cotracker.query_points:
        raise ValueError(
            "cotracker.min_visible_points must be between 1 and query_points"
        )
    if not 0.0 <= args.cotracker.visibility_threshold <= 1.0:
        raise ValueError("cotracker.visibility_threshold must be in [0, 1]")
    if (
        args.cotracker.max_point_deviation_pixels <= 0
        or args.cotracker.max_frame_motion_pixels <= 0
        or args.cotracker.anchor_search_radius < 0
        or args.cotracker.max_consecutive_misses < 1
    ):
        raise ValueError("CoTracker geometry limits must be positive")
    if args.keypoint.radius < 1 or args.keypoint.outline_width < 0:
        raise ValueError("Keypoint radius must be positive and outline width non-negative")
    if not 0 <= args.keypoint.white_min_value <= 255:
        raise ValueError("keypoint.white_min_value must be in [0, 255]")
    if not 0 <= args.keypoint.white_max_chroma <= 255:
        raise ValueError("keypoint.white_max_chroma must be in [0, 255]")
    if args.keypoint.white_min_area < 1 or args.keypoint.tip_search_dilation < 0:
        raise ValueError("White-tip area must be positive and dilation non-negative")
    _validate_rgb(args.keypoint.rgb, "keypoint.rgb")
    _validate_rgb(args.keypoint.outline_rgb, "keypoint.outline_rgb")


def _rewrite_existing_prompt(repo_id: str, task_index: int, prompt: str) -> None:
    """Atomically replace one task prompt in an existing LeRobot dataset."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    dataset_root = pull.HF_LEROBOT_HOME / repo_id
    tasks_path = dataset_root / "meta" / "tasks.parquet"
    if not tasks_path.is_file():
        raise FileNotFoundError(f"LeRobot task metadata not found: {tasks_path}")
    table = pq.read_table(tasks_path)
    required = {"task_index", "task"}
    if not required.issubset(table.column_names):
        raise ValueError(
            f"{tasks_path} must contain {sorted(required)}, got {table.column_names}"
        )
    task_indices = table["task_index"].to_pylist()
    matches = [
        row_index
        for row_index, value in enumerate(task_indices)
        if value == task_index
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one task_index={task_index} in {tasks_path}, "
            f"found {len(matches)}"
        )
    row_index = matches[0]
    tasks = table["task"].to_pylist()
    old_prompt = tasks[row_index]
    new_prompt = prompt.strip()
    tasks[row_index] = new_prompt
    task_column_index = table.schema.get_field_index("task")
    task_field = table.schema.field(task_column_index)
    table = table.set_column(
        task_column_index,
        task_field,
        pa.array(tasks, type=task_field.type),
    )
    temporary_path = tasks_path.with_name(f".{tasks_path.name}.tmp")
    try:
        pq.write_table(table, temporary_path)
        temporary_path.replace(tasks_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    print(
        f"Updated task_index={task_index} in {tasks_path}: "
        f"{old_prompt!r} -> {new_prompt!r}"
    )


def _validate_append_dataset(dataset: Any, output: Path) -> None:
    expected = {
        "image": (("image", "video"), (pull.base.IMAGE_SIZE, pull.base.IMAGE_SIZE, 3)),
        "wrist_image": (("image", "video"), (pull.base.IMAGE_SIZE, pull.base.IMAGE_SIZE, 3)),
        "state": (("float32",), (7,)),
        "actions": (("float32",), (7,)),
    }
    automatic = {"timestamp", "frame_index", "episode_index", "index", "task_index"}
    user_features = {
        name: feature
        for name, feature in dataset.meta.features.items()
        if name not in automatic
    }
    if set(user_features) != set(expected):
        raise ValueError(
            f"{output}: incompatible user features; expected {sorted(expected)}, "
            f"got {sorted(user_features)}"
        )
    if dataset.meta.fps != pull.base.FPS:
        raise ValueError(f"{output}: expected {pull.base.FPS} FPS, got {dataset.meta.fps}")
    for name, (allowed_dtypes, expected_shape) in expected.items():
        feature = user_features[name]
        if feature["dtype"] not in allowed_dtypes or tuple(feature["shape"]) != expected_shape:
            raise ValueError(
                f"{output}: incompatible feature {name!r}: {feature}; "
                f"expected dtype in {allowed_dtypes}, shape {expected_shape}"
            )


def _load_existing_manifest(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise ValueError(f"Expected a list of records in {path}")
    return value


def _tensor_to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _image_to_uint8_hwc(value: Any) -> np.ndarray:
    image = _tensor_to_numpy(value)
    if image.ndim != 3:
        raise ValueError(f"Expected a 3D image, got shape {image.shape}")
    if image.shape[0] in (1, 3, 4) and image.shape[-1] not in (1, 3, 4):
        image = np.moveaxis(image, 0, -1)
    if np.issubdtype(image.dtype, np.floating):
        if not np.isfinite(image).all():
            raise ValueError("Image contains NaN or Inf")
        if image.size and float(image.max()) <= 1.0 + 1e-6:
            image = image * 255.0
        image = np.rint(image)
    return np.clip(image, 0, 255).astype(np.uint8)


def _user_features(dataset: Any) -> dict[str, Any]:
    automatic = {"timestamp", "frame_index", "episode_index", "index", "task_index"}
    return {
        name: feature
        for name, feature in dataset.meta.features.items()
        if name not in automatic
    }


def _append_lerobot_dataset(target: Any, source: Any, features: dict[str, Any]) -> int:
    episodes = sorted(
        source.meta.episodes.to_list(),
        key=lambda record: record["episode_index"],
    )
    total_frames = sum(
        int(episode["dataset_to_index"]) - int(episode["dataset_from_index"])
        for episode in episodes
    )
    progress = tqdm(total=total_frames, desc="Appending LeRobot B", unit="frame")
    try:
        for episode in episodes:
            start = int(episode["dataset_from_index"])
            end = int(episode["dataset_to_index"])
            for frame_index in range(start, end):
                frame = source[frame_index]
                output = {"task": frame["task"]}
                for name, feature in features.items():
                    value = frame[name]
                    if feature["dtype"] in {"image", "video"}:
                        output[name] = _image_to_uint8_hwc(value)
                    elif hasattr(value, "detach"):
                        output[name] = value.detach().cpu().numpy()
                    else:
                        output[name] = value
                target.add_frame(output)
                progress.update()
            target.save_episode()
    finally:
        progress.close()
    return len(episodes)


def main(args: Args) -> None:
    _validate_args(args)
    if args.rewrite_existing_prompt:
        _rewrite_existing_prompt(
            args.repo_id,
            args.rewrite_task_index,
            args.task_prompt,
        )
        return
    root = args.data_dir.expanduser()
    if not root.is_absolute():
        raise ValueError("--data-dir must be absolute")
    root = root.resolve()
    dirs = pull.base._find_episode_dirs(root)
    if args.max_recordings is not None:
        dirs = dirs[: args.max_recordings]
    slices, warnings = pull._plan(dirs, args.split, args.strict_splitting)
    slices = slices[args.start_episode :]
    if args.max_episodes is not None:
        slices = slices[: args.max_episodes]
    for index, item in enumerate(slices):
        path, start, end, closed_start, closed_end, _sa, _center, _zmin = item
        print(
            f"episode {index}: {path.relative_to(root)} [{start}:{end}] "
            f"closed=[{closed_start}:{closed_end}]"
        )
    for path, start, end, reason in warnings:
        print(f"Warning {path.relative_to(root)} [{start}:{end}]: {reason}")
    print(f"Planned {len(slices)} keypoint episodes from {len(dirs)} recordings")
    if args.test_mode:
        return
    if not slices:
        raise ValueError("No valid pull slices were found")

    output = pull.HF_LEROBOT_HOME / args.repo_id
    append_from_repo_id = getattr(args, "append_from_repo_id", None)
    append_lerobot_repo_id = getattr(args, "append_lerobot_repo_id", None)
    append_source = None
    base_dataset = None
    if append_from_repo_id is not None:
        append_from_repo_id = append_from_repo_id.strip()
        if not append_from_repo_id:
            raise ValueError("--append-from-repo-id must not be empty")
        append_source = (pull.HF_LEROBOT_HOME / append_from_repo_id).resolve()
        if append_source == output.resolve():
            raise ValueError("--append-from-repo-id and --repo-id must be different")
        if not append_source.is_dir():
            raise FileNotFoundError(f"Append source dataset does not exist: {append_source}")
        base_dataset = pull.base.LeRobotDataset(
            repo_id=append_from_repo_id,
            root=append_source,
        )
        _validate_append_dataset(base_dataset, append_source)
        print(
            f"Copy source A validated: {base_dataset.meta.total_episodes} episodes, "
            f"{base_dataset.meta.total_frames} frames"
        )

    lerobot_append_dataset = None
    if append_lerobot_repo_id is not None:
        if append_source is None:
            raise ValueError(
                "--append-lerobot-repo-id requires --append-from-repo-id "
                "so A can be copied before B is appended"
            )
        append_lerobot_repo_id = append_lerobot_repo_id.strip()
        if not append_lerobot_repo_id:
            raise ValueError("--append-lerobot-repo-id must not be empty")
        lerobot_append_source = (
            pull.HF_LEROBOT_HOME / append_lerobot_repo_id
        ).resolve()
        if lerobot_append_source in {append_source, output.resolve()}:
            raise ValueError("LeRobot A, B, and target D must be different datasets")
        if not lerobot_append_source.is_dir():
            raise FileNotFoundError(
                f"LeRobot append dataset B does not exist: {lerobot_append_source}"
            )
        lerobot_append_dataset = pull.base.LeRobotDataset(
            repo_id=append_lerobot_repo_id,
            root=lerobot_append_source,
        )
        _validate_append_dataset(lerobot_append_dataset, lerobot_append_source)
        if lerobot_append_dataset.meta.robot_type != base_dataset.meta.robot_type:
            raise ValueError(
                "robot_type mismatch between LeRobot A and B: "
                f"{base_dataset.meta.robot_type!r} != "
                f"{lerobot_append_dataset.meta.robot_type!r}"
            )
        print(
            f"Append source B validated: "
            f"{lerobot_append_dataset.meta.total_episodes} episodes, "
            f"{lerobot_append_dataset.meta.total_frames} frames"
        )

    overwrite_output = False
    if output.exists():
        try:
            answer = input(f"Output exists: {output}\nOverwrite it? [y/N]: ")
        except EOFError:
            return
        if answer.strip().lower() not in {"y", "yes"}:
            return
        overwrite_output = True

    # First identify every anchor with SAM3. CoTracker is loaded only after
    # SAM3 is released so the two large models do not occupy GPU memory together.
    initializer = _make_preprocessor(args.sam3, args.keypoint)
    anchor_plans: list[tuple[int, dict[str, np.ndarray]] | None] = []
    for planned_index, item in enumerate(tqdm(slices, desc="Finding SAM3 anchors")):
        path, start, end, closed_start, _closed_end, *_rest = item
        with h5py.File(path / "data.hdf5") as file:
            required = (
                pull.base.TCP_KEY,
                pull.base.GRIPPER_KEY,
                pull.base.FISHEYE_KEY,
                pull.base.DEPTH_CAMERA_RGB_KEY,
            )
            if any(key not in file for key in required) or any(
                len(file[key]) != len(file[pull.base.TCP_KEY])
                for key in required[1:]
            ):
                raise ValueError(f"{path}: missing or misaligned data")
            state = pull.base._read_state(file, start, end)
            preferred_anchor = visual._grasp_detection_anchor_index(
                state,
                closed_start - start,
            )
            anchor_plan = _detect_anchor_queries(
                initializer,
                path,
                file,
                start,
                len(state),
                preferred_anchor,
                args.cotracker,
            )
        anchor_plans.append(anchor_plan)
        if anchor_plan is None:
            tqdm.write(
                f"episode{planned_index}: skipped because SAM3 could not confirm "
                "the held white cap in both cameras"
            )
        else:
            anchor_local_index, _queries = anchor_plan
            tqdm.write(
                f"episode{planned_index}: SAM3-confirmed CoTracker anchor source "
                f"frame {start + anchor_local_index}"
            )

    del initializer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if not any(plan is not None for plan in anchor_plans):
        raise RuntimeError("SAM3 could not initialize both cameras for any pull slice")

    # Load CoTracker before deleting an existing output dataset.
    tracker = CoTrackerKeypointTracker(args.cotracker)
    debug_dir = _prepare_debug_directory(
        args.debug_video_dir, args.debug_overwrite
    )
    if overwrite_output:
        shutil.rmtree(output)
    manifest_path = output / "pull_keypoint_manifest.json"
    if append_source is not None:
        print(f"Copying append source without re-encoding: {append_source} -> {output}")
        shutil.copytree(append_source, output)
        dataset = pull.base.LeRobotDataset.resume(
            repo_id=args.repo_id,
            root=output,
            image_writer_threads=10,
            image_writer_processes=5,
        )
        starting_episode_index = dataset.meta.total_episodes
        saved_manifest = _load_existing_manifest(manifest_path)
        print(f"Appending after {starting_episode_index} copied episodes in {output}")
    else:
        dataset = pull.base._create_dataset(args.repo_id)
        starting_episode_index = 0
        saved_manifest: list[dict[str, Any]] = []
    if lerobot_append_dataset is not None:
        appended_episodes = _append_lerobot_dataset(
            dataset,
            lerobot_append_dataset,
            _user_features(dataset),
        )
        starting_episode_index = dataset.meta.total_episodes
        print(
            f"Appended {appended_episodes} episodes from LeRobot B; "
            f"target now has {starting_episode_index} episodes"
        )
    new_episode_count = 0
    try:
        for planned_index, (item, anchor_plan) in enumerate(
            tqdm(
                zip(slices, anchor_plans, strict=True),
                total=len(slices),
                desc="Converting keypoint pulls",
            )
        ):
            if anchor_plan is None:
                continue
            (
                path,
                start,
                end,
                closed_start,
                closed_end,
                _start_anchor,
                _center,
                z_min,
            ) = item
            anchor_local_index, queries_by_camera = anchor_plan
            anchor_source_index = start + anchor_local_index
            debug_path = None
            writer = None
            if debug_dir is not None:
                debug_path = (
                    debug_dir
                    / f"episode_{planned_index:04d}_{path.parent.name}_{path.name}_{start}_{end}.mp4"
                )
                writer = pull.base._DebugVideoWriter(debug_path, pull.base.FPS)
            try:
                with h5py.File(path / "data.hdf5") as file:
                    state = pull.base._read_state(file, start, end)
                    sink = _DatasetWithDebugVideo(
                        dataset, writer, args.task_prompt
                    )
                    written = _write_cotracker_episode_frames(
                        sink,
                        path,
                        file,
                        state,
                        start,
                        anchor_local_index,
                        queries_by_camera,
                        tracker,
                        args.keypoint,
                        args.task_prompt,
                        f"episode{planned_index}",
                    )
            finally:
                if writer is not None:
                    writer.close()
            if not written:
                if debug_path is not None and debug_path.exists():
                    debug_path.unlink()
                continue
            dataset.save_episode()
            saved_manifest.append(
                {
                    "episode_index": starting_episode_index + new_episode_count,
                    "planned_episode_index": args.start_episode + planned_index,
                    "source_episode": str(path.relative_to(root)),
                    "start": start,
                    "end": end,
                    "closed_start": closed_start,
                    "closed_end": closed_end,
                    "sam3_anchor": anchor_source_index,
                    "sam3_initialization": "gripper-white-cap point+box",
                    "sam3_native_resolution": True,
                    "sam3_model_input_size": args.sam3.model_input_size,
                    "tracker": "cotracker3_offline",
                    "cotracker_checkpoint": str(args.cotracker.checkpoint),
                    "cotracker_query_points": args.cotracker.query_points,
                    "cotracker_min_visible_points": args.cotracker.min_visible_points,
                    "cotracker_visibility_threshold": args.cotracker.visibility_threshold,
                    "cotracker_max_consecutive_misses": args.cotracker.max_consecutive_misses,
                    "tracking_native_resolution": True,
                    "keypoint_rgb": list(args.keypoint.rgb),
                    "keypoint_radius": args.keypoint.radius,
                    "white_min_value": args.keypoint.white_min_value,
                    "white_max_chroma": args.keypoint.white_max_chroma,
                    "z_min": z_min,
                }
            )
            new_episode_count += 1
    finally:
        if hasattr(dataset, "stop_image_writer"):
            dataset.stop_image_writer()
        else:
            dataset.finalize()

    if not new_episode_count:
        raise RuntimeError("No SAM3-initialized CoTracker episode was saved")
    with manifest_path.open("w", encoding="utf-8") as file:
        json.dump(saved_manifest, file, indent=2)
        file.write("\n")
    print(
        f"Saved {new_episode_count} new keypoint episodes to {output}; "
        f"dataset now has {starting_episode_index + new_episode_count} episodes"
    )
    if debug_dir is not None:
        print(f"Saved keypoint debug videos to {debug_dir}")
    if args.push_to_hub:
        dataset.push_to_hub(
            tags=["pika", "realman", "pull-stick", "keypoint-visual-prompt"],
            private=False,
            push_videos=True,
            license="apache-2.0",
        )


if __name__ == "__main__":
    main(tyro.cli(Args))

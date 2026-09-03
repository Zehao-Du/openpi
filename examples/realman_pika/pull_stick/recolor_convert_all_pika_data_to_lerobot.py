"""Convert all pull-stick recordings to a two-color SAM3 visual-prompt dataset.

The stick held at the pull anchor is recolored to ``color1``. Every stick found
by the SAM3 text detector is recolored to ``color2`` first, so the target mask
can be painted over it with ``color1``. Both masks are propagated through the
episode by independent SAM3 tracker sessions that share one loaded model.

Example:
uv run --project examples/realman_pika --no-sync python \
  examples/realman_pika/pull_stick/recolor_convert_all_pika_data_to_lerobot.py \
  --repo-id Zehao123/pi05_recolor_pika_pull_stick \
  --recolor.color1-rgb 255 0 255 \
  --recolor.color2-rgb 0 255 255 \
  --overwrite

Validate splitting without loading SAM3 or writing a dataset:
uv run --project examples/realman_pika --no-sync python \
  examples/realman_pika/pull_stick/recolor_convert_all_pika_data_to_lerobot.py \
  --test-mode --max-recordings 1
"""

from __future__ import annotations

# Reuse the established pull-stick and SAM3 converter extension points.
# ruff: noqa: SLF001
from collections.abc import Mapping
import colorsys
import dataclasses
import importlib.util
import json
from pathlib import Path
import re
import shutil
import sys
import tempfile
import time
from typing import Any

import convert_double_close_pika_data_to_lerobot as double_close
import h5py
from image_preprocessing import Sam3EpisodeTrackerPreprocessor
from image_preprocessing import clean_mask
from image_preprocessing import recolor_masked_region
import numpy as np
from PIL import Image
from PIL import ImageDraw
from scipy import ndimage
from tqdm.auto import tqdm
import tyro

pull = double_close.pull
base = pull.base
HERE = Path(__file__).resolve().parent

DEFAULT_DATA_DIR = Path("/inspire/hdd/project/robot-dna/baojiachun-CZXS25130063/zehao/dataset/pika/pull_stick")
DEFAULT_REPO_ID = "Zehao123/pi05_recolor_pika_pull_stick_v21"
DEFAULT_SAM3_CHECKPOINT = Path(__file__).resolve().parents[4] / "foundation_models" / "SAM3"
DEFAULT_TASK_PROMPT = "pull the {color} stick and place it on the desk"


def _load_shared_converter() -> Any:
    """Load the v3 writer/v2.1 converter only in the newer examples environment."""
    module_name = "pika_recolor_shared_converter"
    if module_name in sys.modules:
        return sys.modules[module_name]
    path = HERE.parent / "collect_block" / "visualprompt_convert_pika_data_to_lerobot.py"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


@dataclasses.dataclass
class RecolorConfig:
    color1_rgb: tuple[int, int, int] = (255, 0, 255)
    color2_rgb: tuple[int, int, int] = (0, 255, 255)
    alpha: float = 0.9


@dataclasses.dataclass
class SelectionConfig:
    white_min_value: int = 170
    white_max_chroma: int = 55


@dataclasses.dataclass
class Sam3Config:
    checkpoint: Path = DEFAULT_SAM3_CHECKPOINT
    prompts: tuple[str, ...] = ("wooden stick", "stick")
    device: str = "cuda"
    score_threshold: float = 0.25
    fisheye_score_threshold: float = 0.2
    mask_threshold: float = 0.3
    min_component_area: int = 20
    # Use SAM3's pretrained resolution. Frames remain native 640x480 through
    # decoding, mask post-processing, and recoloring; only the final result is
    # resized to the 224x224 shape required by pi0.5.
    model_input_size: int = 1008
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


@dataclasses.dataclass
class SplitConfig:
    standard: pull.SplitConfig = dataclasses.field(default_factory=pull.SplitConfig)
    gripper: double_close.SplitConfig = dataclasses.field(default_factory=double_close.SplitConfig)


@dataclasses.dataclass
class Args:
    data_dir: Path = DEFAULT_DATA_DIR
    repo_id: str = DEFAULT_REPO_ID
    task_prompt: str = DEFAULT_TASK_PROMPT
    overwrite: bool = False
    max_recordings: int | None = None
    start_episode: int = 0
    max_episodes: int | None = None
    num_shards: int = 1
    shard_index: int = 0
    sam_batch_size: int = 8
    preview_video: Path | None = None
    push_to_hub: bool = False
    test_mode: bool = False
    strict_splitting: bool = False
    split: SplitConfig = dataclasses.field(default_factory=SplitConfig)
    sam3: Sam3Config = dataclasses.field(default_factory=Sam3Config)
    selection: SelectionConfig = dataclasses.field(default_factory=SelectionConfig)
    recolor: RecolorConfig = dataclasses.field(default_factory=RecolorConfig)


class _MaskTrackingMixin:
    """Expose SAM3 tracker masks without applying the base class rendering."""

    def track_masks(
        self,
        images: Mapping[str, np.ndarray],
        prepared_frames: Mapping[str, tuple[Any, Any, dict[str, Any]]] | None = None,
    ) -> dict[str, np.ndarray]:
        validated = self._validate_images(images)
        if not validated:
            return {}
        if self._frame_index > 0 and set(validated) != set(self._sessions):
            raise ValueError(f"Camera keys changed within an episode: {set(self._sessions)} -> {set(validated)}")

        started = time.perf_counter()
        if self._frame_index == 0:
            masks = self._detect_first_frame(validated)
            masks = self._initialize_first_frame_trackers(validated, masks)
        else:
            masks = {
                camera_name: self._track_frame(
                    camera_name,
                    image,
                    None if prepared_frames is None else prepared_frames[camera_name],
                )
                for camera_name, image in validated.items()
            }
        masks = self._maybe_redetect_shrunken_mask(validated, masks)
        self._frame_index += 1
        self.last_elapsed_ms = (time.perf_counter() - started) * 1000.0
        return masks


class _AllStickMaskTracker(_MaskTrackingMixin, Sam3EpisodeTrackerPreprocessor):
    """Track the union of all SAM3 text-detected wooden sticks."""


class _WhiteCapNotFoundError(RuntimeError):
    """Signal that the current frame cannot initialize grabbed-stick tracking."""


def _select_grabbed_stick_mask(
    masks: list[np.ndarray],
    scores: list[float],
    cap: np.ndarray,
    point_xy: tuple[float, float],
    max_distance: float,
) -> np.ndarray:
    """Choose the detected stick instance touching, or closest to, the gripper cap."""
    if len(masks) != len(scores):
        raise ValueError("SAM3 candidate masks and scores must have equal lengths")
    if not masks:
        return np.zeros_like(cap)
    height, width = cap.shape
    point_x = int(np.clip(round(point_xy[0]), 0, width - 1))
    point_y = int(np.clip(round(point_xy[1]), 0, height - 1))
    diagonal = float(np.hypot(width, height))
    expanded_cap = ndimage.binary_dilation(cap, iterations=5)
    candidates = []
    for mask, detector_score in zip(masks, scores, strict=True):
        if mask.shape != cap.shape:
            raise ValueError(f"SAM3 mask shape {mask.shape} does not match cap shape {cap.shape}")
        overlap = float(np.count_nonzero(mask & expanded_cap) / max(1, np.count_nonzero(expanded_cap)))
        distance = float(ndimage.distance_transform_edt(~mask)[point_y, point_x] / diagonal)
        # Prefer an instance intersecting the cap; detector confidence only breaks
        # ties after spatial proximity, because every stick has the same semantics.
        candidates.append((overlap > 0.0, overlap, -distance, detector_score, mask, distance))
    _touches, _overlap, _negative_distance, _score, selected, distance = max(candidates, key=lambda item: item[:4])
    if distance > max_distance:
        return np.zeros_like(cap)
    return selected


class _GrabbedStickMaskTracker(_MaskTrackingMixin, Sam3EpisodeTrackerPreprocessor):
    """Track the stick selected by the gripper/white-cap spatial prompt."""

    def __init__(
        self,
        *args: Any,
        selection: SelectionConfig,
        spatial: Sam3Config,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.selection = selection
        self.spatial = spatial

    def _gripper_spatial_prompt(
        self, camera_name: str, image: np.ndarray
    ) -> tuple[tuple[float, float], tuple[float, float, float, float], np.ndarray]:
        """Find the white end cap at the jaws and form a narrow stick box."""
        height, width = image.shape[:2]
        center = {
            "image": self.spatial.fisheye_gripper_center_xy,
            "wrist_image": self.spatial.wrist_gripper_center_xy,
        }[camera_name]
        expected = np.array((center[0] * width, center[1] * height))
        rgb = image.astype(np.int16)
        white = (rgb.min(axis=-1) >= self.selection.white_min_value) & (
            np.ptp(rgb, axis=-1) <= self.selection.white_max_chroma
        )
        roi = np.zeros((height, width), dtype=bool)
        x0 = max(0, int((center[0] - self.spatial.search_half_width) * width))
        x1 = min(width, int((center[0] + self.spatial.search_half_width) * width))
        y0 = max(0, int((center[1] - self.spatial.search_half_height) * height))
        y1 = min(height, int((center[1] + self.spatial.search_half_height) * height))
        roi[y0:y1, x0:x1] = True
        labels, count = ndimage.label(ndimage.binary_erosion(white & roi, iterations=2))
        candidates = []
        diagonal = float(np.hypot(width, height))
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
            raise _WhiteCapNotFoundError(f"{camera_name}: no white cap candidate between the jaws")
        _score, distance, point, component = min(candidates, key=lambda item: item[0])
        if distance > self.spatial.max_prompt_distance:
            raise _WhiteCapNotFoundError(
                f"{camera_name}: white cap too far from gripper "
                f"({distance:.3f} > {self.spatial.max_prompt_distance:.3f})"
            )
        x, y = float(point[0]), float(point[1])
        box = (
            max(0.0, x - self.spatial.prompt_box_half_width * width),
            max(0.0, y - self.spatial.prompt_box_above * height),
            min(width - 1.0, x + self.spatial.prompt_box_half_width * width),
            min(height - 1.0, y + self.spatial.prompt_box_below * height),
        )
        return (x, y), box, component

    def _detect_candidate_instances(
        self, validated: dict[str, np.ndarray]
    ) -> tuple[dict[str, list[np.ndarray]], dict[str, list[float]]]:
        """Detect raw stick instances once for union-mask and grabbed-stick selection."""
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
        postprocess_threshold = min((self.score_threshold, *self.camera_score_thresholds.values()))
        results = self._processor.post_process_instance_segmentation(
            outputs,
            threshold=postprocess_threshold,
            mask_threshold=self.mask_threshold,
            target_sizes=target_sizes,
        )
        masks_by_camera: dict[str, list[np.ndarray]] = {name: [] for name in camera_names}
        scores_by_camera: dict[str, list[float]] = {name: [] for name in camera_names}
        for camera_name, result in zip(batch_camera_names, results, strict=True):
            result_masks = result.get("masks")
            result_scores = result.get("scores")
            if result_masks is None or result_scores is None:
                continue
            camera_threshold = self.camera_score_thresholds.get(camera_name, self.score_threshold)
            for candidate_mask, candidate_score in zip(result_masks, result_scores, strict=True):
                score = float(candidate_score.detach().cpu() if hasattr(candidate_score, "detach") else candidate_score)
                if score < camera_threshold:
                    continue
                mask = self._masks_to_numpy(candidate_mask, validated[camera_name].shape[:2])
                if mask.any():
                    masks_by_camera[camera_name].append(mask)
                    scores_by_camera[camera_name].append(score)

        return masks_by_camera, scores_by_camera

    def _select_candidate_instances(
        self,
        validated: dict[str, np.ndarray],
        masks_by_camera: dict[str, list[np.ndarray]],
        scores_by_camera: dict[str, list[float]],
    ) -> dict[str, np.ndarray]:
        """Select the detected instance touching the white cap at the gripper."""

        selected = {}
        for camera_name, image in validated.items():
            point, _box, cap = self._gripper_spatial_prompt(camera_name, image)
            cleaned_masks = [clean_mask(mask, self.min_component_area) for mask in masks_by_camera[camera_name]]
            selected[camera_name] = _select_grabbed_stick_mask(
                cleaned_masks,
                scores_by_camera[camera_name],
                cap,
                point,
                self.spatial.max_prompt_distance,
            )
        return selected

    def _detect_first_frame(self, validated: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Select one SAM3 text-detected stick instance using the cap at the jaws."""
        masks_by_camera, scores_by_camera = self._detect_candidate_instances(validated)
        return self._select_candidate_instances(validated, masks_by_camera, scores_by_camera)

    def _maybe_redetect_shrunken_mask(
        self, validated: dict[str, np.ndarray], masks: dict[str, np.ndarray]
    ) -> dict[str, np.ndarray]:
        # Text re-detection can jump from the grabbed stick to another identical stick.
        del validated
        return masks


def _validate_rgb(rgb: tuple[int, int, int], name: str) -> None:
    if len(rgb) != 3 or any(channel < 0 or channel > 255 for channel in rgb):
        raise ValueError(f"{name} must contain three values in [0, 255], got {rgb}")


def _color_name(rgb: tuple[int, int, int]) -> str:
    _validate_rgb(rgb, "RGB color")
    if tuple(rgb) == (255, 0, 255):
        return "magenta"
    red, green, blue = (channel / 255.0 for channel in rgb)
    hue, saturation, value = colorsys.rgb_to_hsv(red, green, blue)
    if value < 0.15:
        return "black"
    if saturation < 0.15:
        return "white" if value >= 0.85 else "gray"
    hue_degrees = hue * 360.0
    sectors = (
        (15.0, "red"),
        (45.0, "orange"),
        (70.0, "yellow"),
        (165.0, "green"),
        (195.0, "cyan"),
        (255.0, "blue"),
        (285.0, "purple"),
        (345.0, "pink"),
        (360.0, "red"),
    )
    return next(name for upper_bound, name in sectors if hue_degrees < upper_bound)


def _resolve_task_prompt(template: str, color1_rgb: tuple[int, int, int], color2_rgb: tuple[int, int, int]) -> str:
    prompt = template
    replacements = {
        "color": _color_name(color1_rgb),
        "color1": _color_name(color1_rgb),
        "color2": _color_name(color2_rgb),
    }
    for key, value in replacements.items():
        prompt = prompt.replace(f"${{{key}}}", value).replace(f"{{{key}}}", value)
    prompt = prompt.strip()
    if not prompt:
        raise ValueError("--task-prompt must not be empty")
    return prompt


def _apply_recolor_masks(
    image: np.ndarray,
    all_sticks_mask: np.ndarray,
    grabbed_stick_mask: np.ndarray,
    config: RecolorConfig,
) -> np.ndarray:
    """Apply color2 to all sticks, then make the grabbed stick color1."""
    output = recolor_masked_region(
        image,
        all_sticks_mask,
        target_rgb=config.color2_rgb,
        alpha=config.alpha,
    )
    return recolor_masked_region(
        output,
        grabbed_stick_mask,
        target_rgb=config.color1_rgb,
        alpha=config.alpha,
    )


def _make_preview_frame(
    originals: dict[str, np.ndarray],
    processed: dict[str, np.ndarray],
    frame_index: int,
    task_prompt: str,
) -> np.ndarray:
    """Compose a 2x2 comparison without reducing the native 640x480 panels."""
    panels = (
        (originals[f"{frame_index}:image"], "fisheye original"),
        (processed[f"{frame_index}:image"], "fisheye SAM3 recolor"),
        (originals[f"{frame_index}:wrist_image"], "RealSense original"),
        (processed[f"{frame_index}:wrist_image"], "RealSense SAM3 recolor"),
    )
    panel_height = max(panel.shape[0] for panel, _label in panels)
    panel_width = max(panel.shape[1] for panel, _label in panels)
    canvas = np.zeros((panel_height * 2, panel_width * 2, 3), dtype=np.uint8)
    for panel_index, (panel, _label) in enumerate(panels):
        row, column = divmod(panel_index, 2)
        y, x = row * panel_height, column * panel_width
        resized = base.image_tools.resize_with_pad(panel, panel_height, panel_width)
        canvas[y : y + panel_height, x : x + panel_width] = resized

    preview = Image.fromarray(canvas)
    draw = ImageDraw.Draw(preview)
    for panel_index, (_, label) in enumerate(panels):
        row, column = divmod(panel_index, 2)
        y, x = row * panel_height, column * panel_width
        draw.rectangle((x, y, x + 145, y + 15), fill=(0, 0, 0))
        draw.text((x + 3, y + 2), label, fill=(255, 255, 255))
    draw.rectangle((0, panel_height * 2 - 17, panel_width * 2, panel_height * 2), fill=(0, 0, 0))
    draw.text((3, panel_height * 2 - 15), task_prompt, fill=(255, 255, 255))
    return np.asarray(preview, dtype=np.uint8)


class _ResizeDatasetImages:
    """Resize native recolored images only at the LeRobot writer boundary."""

    def __init__(self, dataset: Any) -> None:
        self.dataset = dataset

    def add_frame(self, frame: dict[str, Any]) -> None:
        resized = dict(frame)
        for key in ("image", "wrist_image"):
            resized[key] = base.image_tools.resize_with_pad(frame[key], base.IMAGE_SIZE, base.IMAGE_SIZE)
        self.dataset.add_frame(resized)


class _NativePreviewVideoWriter:
    """Encode the native 2x2 comparison without the shared 448px constraint."""

    def __init__(self, path: Path, fps: int, width: int = 1280, height: int = 960) -> None:
        import av

        path.parent.mkdir(parents=True, exist_ok=True)
        self._container = av.open(str(path), mode="w")
        self._stream = self._container.add_stream("libx264", rate=fps)
        self._stream.width = width
        self._stream.height = height
        self._stream.pix_fmt = "yuv420p"
        self._av = av

    def add_frame(self, image: np.ndarray) -> None:
        expected = (self._stream.height, self._stream.width, 3)
        if image.shape != expected:
            raise ValueError(f"Native preview frame has shape {image.shape}; expected {expected}")
        frame = self._av.VideoFrame.from_ndarray(image, format="rgb24")
        for packet in self._stream.encode(frame):
            self._container.mux(packet)

    def close(self) -> None:
        for packet in self._stream.encode():
            self._container.mux(packet)
        self._container.close()


class Sam3TwoColorRecolorPreprocessor:
    requires_sequential_frames = True
    output_native_resolution = True

    def __init__(
        self,
        all_sticks: _AllStickMaskTracker,
        grabbed_stick: _GrabbedStickMaskTracker,
        recolor: RecolorConfig,
    ) -> None:
        self.all_sticks = all_sticks
        self.grabbed_stick = grabbed_stick
        self.recolor = recolor

    def start_episode(self) -> None:
        self.all_sticks.start_episode()
        self.grabbed_stick.start_episode()

    def has_active_trackers(self, camera_names: tuple[str, ...]) -> bool:
        return self.all_sticks.has_active_trackers(camera_names) and self.grabbed_stick.has_active_trackers(
            camera_names
        )

    def preprocess(self, images: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
        validated = self.all_sticks._validate_images(images)
        all_frame_index = getattr(self.all_sticks, "_frame_index", None)
        grabbed_frame_index = getattr(self.grabbed_stick, "_frame_index", None)
        if all_frame_index is None or grabbed_frame_index is None:
            all_masks = self.all_sticks.track_masks(validated)
            try:
                grabbed_masks = self.grabbed_stick.track_masks(validated)
            except _WhiteCapNotFoundError:
                return {camera_name: image.copy() for camera_name, image in validated.items()}
        elif all_frame_index == 0 and grabbed_frame_index == 0:
            try:
                candidate_masks, candidate_scores = self.grabbed_stick._detect_candidate_instances(validated)
                grabbed_masks = self.grabbed_stick._select_candidate_instances(
                    validated, candidate_masks, candidate_scores
                )
            except _WhiteCapNotFoundError:
                # Leave both trackers inactive. The shared writer retries adjacent
                # frames and skips the episode if no frame can initialize them.
                return {camera_name: image.copy() for camera_name, image in validated.items()}
            all_masks = {}
            for camera_name, image in validated.items():
                masks = candidate_masks[camera_name]
                union_mask = np.logical_or.reduce(masks) if masks else np.zeros(image.shape[:2], dtype=bool)
                all_masks[camera_name] = clean_mask(union_mask, self.all_sticks.min_component_area)
            all_masks = self.all_sticks._initialize_first_frame_trackers(validated, all_masks)
            grabbed_masks = self.grabbed_stick._initialize_first_frame_trackers(validated, grabbed_masks)
            all_masks = self.all_sticks._maybe_redetect_shrunken_mask(validated, all_masks)
            grabbed_masks = self.grabbed_stick._maybe_redetect_shrunken_mask(validated, grabbed_masks)
            self.all_sticks._frame_index += 1
            self.grabbed_stick._frame_index += 1
        else:
            prepared_frames = {
                camera_name: self.all_sticks.prepare_tracker_frame(image) for camera_name, image in validated.items()
            }
            all_masks = self.all_sticks.track_masks(validated, prepared_frames)
            grabbed_masks = self.grabbed_stick.track_masks(validated, prepared_frames)
        return {
            camera_name: _apply_recolor_masks(image, all_masks[camera_name], grabbed_masks[camera_name], self.recolor)
            for camera_name, image in validated.items()
        }


def _make_preprocessor(args: Args) -> Sam3TwoColorRecolorPreprocessor:
    common_kwargs = {
        "prompts": args.sam3.prompts,
        "device": args.sam3.device,
        "score_threshold": args.sam3.score_threshold,
        "camera_score_thresholds": {"image": args.sam3.fisheye_score_threshold},
        "mask_threshold": args.sam3.mask_threshold,
        "min_component_area": args.sam3.min_component_area,
        "model_input_size": args.sam3.model_input_size,
        "error_policy": "raise",
        "redetect_area_ratio": args.sam3.redetect_area_ratio,
        "redetect_reference_decay": args.sam3.redetect_reference_decay,
        "redetect_cooldown_frames": args.sam3.redetect_cooldown_frames,
    }
    grabbed = _GrabbedStickMaskTracker(
        args.sam3.checkpoint,
        target_rgb=args.recolor.color1_rgb,
        selection=args.selection,
        spatial=args.sam3,
        **common_kwargs,
    )
    all_sticks = _AllStickMaskTracker(
        args.sam3.checkpoint,
        target_rgb=args.recolor.color2_rgb,
        video_model=grabbed._video_model,
        detector_processor=grabbed._processor,
        tracker_processor=grabbed._tracker_processor,
        torch_module=grabbed._torch,
        **common_kwargs,
    )
    return Sam3TwoColorRecolorPreprocessor(all_sticks, grabbed, args.recolor)


def _uses_double_close(episode_dir: Path) -> bool:
    return episode_dir.parent.name.endswith("gripper")


def _double_warning_tuple(warning: str, episode_dirs: list[Path]) -> tuple[Path, int, int, str]:
    source = next((path for path in episode_dirs if warning.startswith(f"{path}:")), episode_dirs[0])
    reason = warning.removeprefix(f"{source}: ")
    match = re.search(r"\[(\d+):(\d+)\]", reason)
    if match:
        start, end = (int(value) for value in match.groups())
    else:
        with h5py.File(source / "data.hdf5") as file:
            start, end = 0, len(file[base.GRIPPER_KEY])
    return source, start, end, reason


def _adapt_double_close_plan(
    episode_dirs: list[Path],
    config: double_close.SplitConfig,
    strict: bool,  # noqa: FBT001 - matches the reused splitter interface.
) -> tuple[list[tuple[Any, ...]], list[tuple[Path, int, int, str]]]:
    planned, warnings = double_close._plan(episode_dirs, config, strict=strict)
    z_min_by_source: dict[Path, float] = {}
    compatible = []
    for item in planned:
        if item.source_dir not in z_min_by_source:
            with h5py.File(item.source_dir / "data.hdf5") as file:
                pose = np.asarray(file[base.TCP_KEY], dtype=float)
            z_min_by_source[item.source_dir] = float(pose[:, 2].min())
        closed_start, closed_end = max(item.sustained_closes, key=lambda run: run[1] - run[0])
        center = (closed_start + closed_end - 1) // 2
        anchor = (center, center + 1)
        compatible.append(
            (
                item.source_dir,
                item.start,
                item.end,
                closed_start,
                closed_end,
                anchor,
                anchor,
                z_min_by_source[item.source_dir],
            )
        )
    return compatible, [_double_warning_tuple(warning, episode_dirs) for warning in warnings]


def _plan(
    episode_dirs: list[Path],
    config: SplitConfig,
    strict: bool,  # noqa: FBT001 - called positionally by the converter.
) -> tuple[list[tuple[Any, ...]], list[tuple[Path, int, int, str]]]:
    standard_dirs = [path for path in episode_dirs if not _uses_double_close(path)]
    gripper_dirs = [path for path in episode_dirs if _uses_double_close(path)]
    planned: list[tuple[Any, ...]] = []
    warnings: list[tuple[Path, int, int, str]] = []
    if standard_dirs:
        standard_planned, standard_warnings = pull._plan(standard_dirs, config.standard, strict)
        planned.extend(standard_planned)
        warnings.extend(standard_warnings)
    if gripper_dirs:
        gripper_planned, gripper_warnings = _adapt_double_close_plan(gripper_dirs, config.gripper, strict)
        planned.extend(gripper_planned)
        warnings.extend(gripper_warnings)

    source_order = {path: index for index, path in enumerate(episode_dirs)}
    planned.sort(key=lambda item: (source_order[item[0]], item[1], item[2]))
    warnings.sort(key=lambda item: (source_order[item[0]], item[1], item[2]))
    print(
        f"Splitter routing: {len(standard_dirs)} standard recordings, "
        f"{len(gripper_dirs)} gripper/double-close recordings"
    )
    return planned, warnings


def _validate_args(args: Args) -> str:
    if not args.repo_id.strip():
        raise ValueError("--repo-id must not be empty")
    if args.max_recordings is not None and args.max_recordings < 1:
        raise ValueError("--max-recordings must be positive")
    if args.start_episode < 0:
        raise ValueError("--start-episode must be non-negative")
    if args.max_episodes is not None and args.max_episodes < 1:
        raise ValueError("--max-episodes must be positive")
    if args.num_shards < 1:
        raise ValueError("--num-shards must be positive")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("--shard-index must be in [0, num_shards)")
    if args.sam_batch_size < 1:
        raise ValueError("--sam-batch-size must be positive")
    if args.preview_video is not None and args.preview_video.exists() and not args.overwrite:
        raise FileExistsError(f"Preview video exists: {args.preview_video}. Pass --overwrite to replace it.")
    if not 0.0 <= args.recolor.alpha <= 1.0:
        raise ValueError("recolor.alpha must be between 0 and 1")
    _validate_rgb(args.recolor.color1_rgb, "recolor.color1_rgb")
    _validate_rgb(args.recolor.color2_rgb, "recolor.color2_rgb")
    if args.recolor.color1_rgb == args.recolor.color2_rgb:
        raise ValueError("recolor.color1_rgb and recolor.color2_rgb must differ")
    return _resolve_task_prompt(args.task_prompt, args.recolor.color1_rgb, args.recolor.color2_rgb)


def _validate_episode(file: h5py.File, episode_dir: Path) -> None:
    required = (base.TCP_KEY, base.GRIPPER_KEY, base.FISHEYE_KEY, base.DEPTH_CAMERA_RGB_KEY)
    missing = [key for key in required if key not in file]
    if missing:
        raise KeyError(f"{episode_dir}: missing HDF5 keys {missing}")
    length = len(file[base.TCP_KEY])
    if any(len(file[key]) != length for key in required[1:]):
        raise ValueError(f"{episode_dir}: camera/state length mismatch")


def main(args: Args) -> None:
    task_prompt = _validate_args(args)
    root = args.data_dir.expanduser()
    if not root.is_absolute():
        raise ValueError("--data-dir must be absolute")
    root = root.resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)

    episode_dirs = base._find_episode_dirs(root)
    if args.max_recordings is not None:
        episode_dirs = episode_dirs[: args.max_recordings]
    slices, warnings = _plan(episode_dirs, args.split, args.strict_splitting)
    slice_offset = args.start_episode
    slices = slices[args.start_episode :]
    if args.max_episodes is not None:
        slices = slices[: args.max_episodes]
    if args.num_shards > 1:
        shard_start = len(slices) * args.shard_index // args.num_shards
        shard_end = len(slices) * (args.shard_index + 1) // args.num_shards
        slice_offset += shard_start
        slices = slices[shard_start:shard_end]
    for path, start, end, reason in warnings:
        print(f"Warning {path.relative_to(root)} [{start}:{end}]: {reason}")
    print(
        f"Planned {len(slices)} recolor episodes from {len(episode_dirs)} recordings "
        f"for shard {args.shard_index + 1}/{args.num_shards}"
    )
    print(f"Task prompt: {task_prompt}")
    if args.test_mode:
        for path, start, end, *_ in slices:
            with h5py.File(path / "data.hdf5") as file:
                _validate_episode(file, path)
                base._read_state(file, start, end)
        return
    if not slices:
        raise ValueError("No valid pull slices were found")

    output = pull.HF_LEROBOT_HOME / args.repo_id
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists: {output}. Pass --overwrite to replace it.")

    # Load and validate SAM3 before touching an existing output dataset.
    shared = _load_shared_converter()
    # Feed native frames to SAM3; this module resizes after applying both masks.
    shared._read_rgb = shared._read_native_rgb
    # The shared preview assumes its source reader already returned 224x224.
    shared._make_preview_frame = _make_preview_frame
    image_preprocessor = _make_preprocessor(args)
    output.parent.mkdir(parents=True, exist_ok=True)
    saved_manifest: list[dict[str, Any]] = []
    skipped = 0
    with tempfile.TemporaryDirectory(prefix=f".{output.name}-staging-", dir=output.parent) as temporary_directory:
        temporary_root = Path(temporary_directory)
        staging_v3 = temporary_root / "v3"
        staging_v21 = temporary_root / "v2.1"
        staging_preview = temporary_root / "preview.mp4"
        dataset = shared._create_dataset(args.repo_id, root=staging_v3)
        preview_writer = _NativePreviewVideoWriter(staging_preview, base.FPS) if args.preview_video else None
        try:
            for planned_index, item in enumerate(tqdm(slices, desc="Converting recolor pulls", unit="episode")):
                path, start, end, closed_start, closed_end, *_rest, z_min = item
                with h5py.File(path / "data.hdf5") as file:
                    _validate_episode(file, path)
                    state = base._read_state(file, start, end)
                    anchor_local_index = shared._grasp_detection_anchor_index(state, closed_start - start)
                    written = shared._write_episode_frames(
                        _ResizeDatasetImages(dataset),
                        path,
                        file,
                        state,
                        start,
                        image_preprocessor,
                        args.sam_batch_size,
                        task_prompt,
                        f"episode{planned_index}",
                        preview_writer=preview_writer,
                        detection_anchor_local_index=anchor_local_index,
                    )
                if not written:
                    skipped += 1
                    continue
                dataset.save_episode()
                saved_manifest.append(
                    {
                        "episode_index": len(saved_manifest),
                        "planned_episode_index": slice_offset + planned_index,
                        "source_episode": str(path.relative_to(root)),
                        "start": start,
                        "end": end,
                        "closed_start": closed_start,
                        "closed_end": closed_end,
                        "sam3_anchor": start + anchor_local_index,
                        "sam3_prompts": list(args.sam3.prompts),
                        "sam3_initialization": "text instance selected by gripper white cap",
                        "color1_rgb": list(args.recolor.color1_rgb),
                        "color1_name": _color_name(args.recolor.color1_rgb),
                        "color1_role": "grabbed_stick",
                        "color2_rgb": list(args.recolor.color2_rgb),
                        "color2_name": _color_name(args.recolor.color2_rgb),
                        "color2_role": "other_sticks",
                        "task_prompt": task_prompt,
                        "z_min": z_min,
                    }
                )
        finally:
            try:
                if saved_manifest:
                    dataset.finalize()
                elif hasattr(dataset, "stop_image_writer"):
                    dataset.stop_image_writer()
            finally:
                if preview_writer is not None:
                    preview_writer.close()

        if not saved_manifest:
            raise RuntimeError("SAM3 could not initialize both target and all-stick trackers for any episode")
        shared._convert_lerobot_v3_to_v21(staging_v3, staging_v21)
        with (staging_v21 / "pull_recolor_manifest.json").open("w", encoding="utf-8") as file:
            json.dump(saved_manifest, file, indent=2)
            file.write("\n")
        if output.exists():
            shutil.rmtree(output)
        shutil.move(staging_v21, output)
        if args.preview_video is not None:
            preview_video = args.preview_video.expanduser().resolve()
            preview_video.parent.mkdir(parents=True, exist_ok=True)
            if preview_video.exists():
                preview_video.unlink()
            shutil.move(staging_preview, preview_video)

    print(f"Saved {len(saved_manifest)} recolor episodes to {output}; skipped {skipped}")
    if args.preview_video is not None:
        print(f"Saved recolor preview video to {args.preview_video.expanduser().resolve()}")
    if args.push_to_hub:
        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(args.repo_id, repo_type="dataset", exist_ok=True)
        api.upload_folder(repo_id=args.repo_id, repo_type="dataset", folder_path=output)


if __name__ == "__main__":
    main(tyro.cli(Args))

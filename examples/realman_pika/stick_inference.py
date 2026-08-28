"""Target selection and visual prompting for RealMan-Pika pull-stick inference."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import dataclasses
import logging
import pathlib
from typing import Any

from image_preprocessing import Sam3EpisodeTrackerPreprocessor
import numpy as np
from scipy import ndimage

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class StickConfig:
    """Configuration whose hardware-specific calibration values are intentionally blank."""

    external_camera_serials: tuple[str, ...] = ()
    calibration_dir: pathlib.Path | None = None
    prompts: tuple[str, ...] = (
        "red stick",
        "blue stick",
        "orange stick",
        "yellow stick",
        "marker pen",
    )
    width: int = 1280
    height: int = 720
    fps: int = 30
    warmup_frames: int = 8
    min_stick_points: int = 100
    max_depth_m: float = 2.0
    model_input_size: int = 224
    score_threshold: float = 0.5
    mask_threshold: float = 0.5
    duplicate_iou_threshold: float = 0.6
    prompt_box_half_size: int = 32
    keypoint_rgb: tuple[int, int, int] = (255, 0, 255)
    keypoint_radius: int = 5
    keypoint_outline_rgb: tuple[int, int, int] = (255, 255, 255)
    keypoint_outline_width: int = 2
    white_min_value: int = 170
    white_max_chroma: int = 55
    white_search_dilation: int = 5

    def validate_hardware(self) -> None:
        if not self.external_camera_serials:
            raise ValueError(
                "--stick requires at least one --stick-config.external-camera-serials value"
            )
        if self.calibration_dir is None:
            raise ValueError("--stick requires --stick-config.calibration-dir")
        calibration_dir = pathlib.Path(self.calibration_dir).expanduser()
        missing = [
            calibration_dir / f"T_cam_to_world_{serial}.npy"
            for serial in self.external_camera_serials
            if not (calibration_dir / f"T_cam_to_world_{serial}.npy").is_file()
        ]
        if missing:
            raise FileNotFoundError(
                "Missing external-camera calibration file(s): "
                + ", ".join(str(path) for path in missing)
            )
        if min(self.width, self.height, self.fps, self.warmup_frames, self.min_stick_points) < 1:
            raise ValueError("stick camera dimensions, rates, warmup, and point count must be positive")


@dataclasses.dataclass(frozen=True)
class StickSelection:
    serial: str
    prompt: str
    vertical_angle_deg: float
    axis_world: np.ndarray
    top_point_world: np.ndarray
    preview_rgb: np.ndarray


@dataclasses.dataclass
class _Candidate:
    serial: str
    prompt: str
    score: float
    mask: np.ndarray
    image_rgb: np.ndarray
    axis_world: np.ndarray
    top_point_world: np.ndarray


def vertical_angle_deg(axis: Any) -> float:
    """Return the acute angle between an undirected 3-D axis and world Z."""
    axis = np.asarray(axis, dtype=np.float64)
    if axis.shape != (3,) or not np.isfinite(axis).all():
        raise ValueError(f"Expected a finite axis with shape (3,), got {axis}")
    norm = float(np.linalg.norm(axis))
    if norm <= 1e-8:
        raise ValueError("Stick axis has near-zero length")
    cosine = np.clip(abs(float(axis[2])) / norm, 0.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def select_most_vertical_index(axes: Sequence[Any]) -> int:
    angles = []
    for axis in axes:
        try:
            angles.append(vertical_angle_deg(axis))
        except ValueError:
            angles.append(np.inf)
    if not angles or not np.isfinite(angles).any():
        raise ValueError("No valid stick axes were detected")
    return int(np.argmin(angles))


def point_cloud_from_mask(
    depth_raw: np.ndarray,
    mask: np.ndarray,
    intrinsics: Any,
    depth_scale: float,
    camera_to_world: np.ndarray,
    *,
    max_depth_m: float,
) -> np.ndarray:
    """Deproject a masked aligned depth image and transform it into world coordinates."""
    depth = np.asarray(depth_raw)
    mask = np.asarray(mask, dtype=bool)
    camera_to_world = np.asarray(camera_to_world, dtype=np.float64)
    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    if depth.ndim != 2 or mask.shape != depth.shape:
        raise ValueError(f"Depth/mask shapes do not match: {depth.shape} vs {mask.shape}")
    if camera_to_world.shape != (4, 4) or not np.isfinite(camera_to_world).all():
        raise ValueError(
            f"Expected a finite 4x4 camera-to-world matrix, got {camera_to_world.shape}"
        )
    rows, cols = np.nonzero(mask & (depth > 0))
    if len(rows) == 0:
        return np.empty((0, 3), dtype=np.float64)
    z = depth[rows, cols].astype(np.float64) * float(depth_scale)
    keep = np.isfinite(z) & (z > 0.0) & (z <= float(max_depth_m))
    rows, cols, z = rows[keep], cols[keep], z[keep]
    x = (cols - float(intrinsics.ppx)) / float(intrinsics.fx) * z
    y = (rows - float(intrinsics.ppy)) / float(intrinsics.fy) * z
    camera_points = np.stack((x, y, z, np.ones_like(z)), axis=1)
    return (camera_to_world @ camera_points.T).T[:, :3]


def estimate_axis_and_top(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if len(points) < 3 or not np.isfinite(points).all():
        raise ValueError("At least three finite points are required for stick PCA")
    center = points.mean(axis=0)
    covariance = np.cov(points - center, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    axis = eigenvectors[:, int(np.argmax(eigenvalues))]
    axis /= np.linalg.norm(axis)
    if axis[2] < 0:
        axis *= -1
    projections = (points - center) @ axis
    top_count = max(3, int(np.ceil(len(points) * 0.08)))
    top_offset = float(np.mean(np.sort(projections)[-top_count:]))
    return axis, center + top_offset * axis


def _mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    intersection = np.count_nonzero(mask_a & mask_b)
    union = np.count_nonzero(mask_a | mask_b)
    return float(intersection / union) if union else 0.0


def _draw_selection_preview(candidate: _Candidate) -> np.ndarray:
    image = candidate.image_rgb.copy()
    tint = np.zeros_like(image)
    tint[..., 0] = 255
    image[candidate.mask] = np.rint(
        0.55 * image[candidate.mask] + 0.45 * tint[candidate.mask]
    ).astype(np.uint8)
    try:
        import cv2

        bgr = image[..., ::-1].copy()
        label = f"SELECTED: {candidate.prompt}, vertical={vertical_angle_deg(candidate.axis_world):.1f} deg"
        cv2.putText(bgr, label, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        image = bgr[..., ::-1]
    except ImportError:
        pass
    return image


class _Sam3InstanceDetector:
    def __init__(self, checkpoint: pathlib.Path, config: StickConfig, device: str | None) -> None:
        import torch
        from transformers import AutoConfig
        from transformers import Sam3Model
        from transformers import Sam3Processor

        self._torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        video_config = AutoConfig.from_pretrained(checkpoint, local_files_only=True)
        detector_config = video_config.detector_config
        detector_config.image_size = config.model_input_size
        self.model = Sam3Model.from_pretrained(
            checkpoint, config=detector_config, local_files_only=True
        ).to(self.device)
        self.model.eval()
        self.processor = Sam3Processor.from_pretrained(checkpoint, local_files_only=True)
        self.config = config

    def detect(self, image_rgb: np.ndarray) -> list[tuple[str, float, np.ndarray]]:
        prompts = self.config.prompts
        inputs = self.processor(
            images=[image_rgb] * len(prompts),
            text=list(prompts),
            return_tensors="pt",
            size={"height": self.config.model_input_size, "width": self.config.model_input_size},
        ).to(self.device)
        target_sizes = inputs["original_sizes"].detach().cpu().tolist()
        with self._torch.inference_mode():
            outputs = self.model(**inputs)
        results = self.processor.post_process_instance_segmentation(
            outputs,
            threshold=self.config.score_threshold,
            mask_threshold=self.config.mask_threshold,
            target_sizes=target_sizes,
        )
        detected: list[tuple[str, float, np.ndarray]] = []
        for prompt, result in zip(prompts, results, strict=True):
            masks = result.get("masks")
            scores = result.get("scores")
            if masks is None:
                continue
            masks = masks.detach().cpu().numpy() if hasattr(masks, "detach") else np.asarray(masks)
            if masks.ndim == 2:
                masks = masks[None]
            if scores is None:
                scores_array = np.ones(len(masks), dtype=np.float64)
            else:
                scores_array = scores.detach().cpu().numpy() if hasattr(scores, "detach") else np.asarray(scores)
            for score, raw_mask in zip(scores_array, masks, strict=True):
                cleaned_mask = ndimage.binary_closing(np.asarray(raw_mask, dtype=bool))
                if np.count_nonzero(cleaned_mask) >= self.config.min_stick_points:
                    detected.append((prompt, float(score), cleaned_mask))
        detected.sort(key=lambda item: item[1], reverse=True)
        unique: list[tuple[str, float, np.ndarray]] = []
        for item in detected:
            if not any(_mask_iou(item[2], kept[2]) >= self.config.duplicate_iou_threshold for kept in unique):
                unique.append(item)
        return unique


class StickTargetSelector:
    """Capture calibrated external RGB-D views and choose the most vertical stick."""

    def __init__(
        self,
        config: StickConfig,
        *,
        checkpoint: pathlib.Path,
        device: str | None = None,
    ) -> None:
        self.config = config
        self.checkpoint = pathlib.Path(checkpoint).expanduser()
        self.device = device

    def select(self) -> StickSelection:
        self.config.validate_hardware()
        if not self.checkpoint.is_dir():
            raise FileNotFoundError(f"SAM 3 checkpoint directory does not exist: {self.checkpoint}")
        import pyrealsense2 as rs

        detector = _Sam3InstanceDetector(self.checkpoint, self.config, self.device)
        candidates: list[_Candidate] = []
        try:
            for serial in self.config.external_camera_serials:
                camera_to_world = np.load(
                    pathlib.Path(self.config.calibration_dir).expanduser()
                    / f"T_cam_to_world_{serial}.npy"
                ).astype(np.float64)
                pipeline = rs.pipeline()
                rs_config = rs.config()
                rs_config.enable_device(serial)
                rs_config.enable_stream(
                    rs.stream.color,
                    self.config.width,
                    self.config.height,
                    rs.format.rgb8,
                    self.config.fps,
                )
                rs_config.enable_stream(
                    rs.stream.depth,
                    self.config.width,
                    self.config.height,
                    rs.format.z16,
                    self.config.fps,
                )
                profile = pipeline.start(rs_config)
                align = rs.align(rs.stream.color)
                try:
                    aligned = None
                    for _ in range(self.config.warmup_frames):
                        aligned = align.process(pipeline.wait_for_frames(10_000))
                    assert aligned is not None
                    color_frame = aligned.get_color_frame()
                    depth_frame = aligned.get_depth_frame()
                    if not color_frame or not depth_frame:
                        raise RuntimeError(f"External camera {serial} returned incomplete RGB-D")
                    image_rgb = np.asanyarray(color_frame.get_data())
                    depth_raw = np.asanyarray(depth_frame.get_data())
                    intrinsics = color_frame.profile.as_video_stream_profile().intrinsics
                    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
                finally:
                    pipeline.stop()

                for prompt, score, mask in detector.detect(image_rgb):
                    points = point_cloud_from_mask(
                        depth_raw,
                        mask,
                        intrinsics,
                        depth_scale,
                        camera_to_world,
                        max_depth_m=self.config.max_depth_m,
                    )
                    if len(points) < self.config.min_stick_points:
                        continue
                    try:
                        axis, top = estimate_axis_and_top(points)
                    except ValueError:
                        continue
                    candidates.append(
                        _Candidate(serial, prompt, score, mask, image_rgb, axis, top)
                    )
        finally:
            del detector

        if not candidates:
            raise RuntimeError("No valid calibrated stick candidate was detected")
        selected = candidates[select_most_vertical_index([item.axis_world for item in candidates])]
        angle = vertical_angle_deg(selected.axis_world)
        logger.info(
            "Selected most vertical stick: camera=%s prompt=%s angle=%.2f deg",
            selected.serial,
            selected.prompt,
            angle,
        )
        return StickSelection(
            serial=selected.serial,
            prompt=selected.prompt,
            vertical_angle_deg=angle,
            axis_world=selected.axis_world.copy(),
            top_point_world=selected.top_point_world.copy(),
            preview_rgb=_draw_selection_preview(selected),
        )


def draw_keypoint(
    image_rgb: np.ndarray,
    point_xy: tuple[float, float],
    *,
    rgb: tuple[int, int, int] = (255, 0, 255),
    radius: int = 5,
    outline_rgb: tuple[int, int, int] = (255, 255, 255),
    outline_width: int = 2,
) -> np.ndarray:
    image = np.asarray(image_rgb).copy()
    height, width = image.shape[:2]
    x, y = float(point_xy[0]), float(point_xy[1])
    yy, xx = np.ogrid[:height, :width]
    distance_sq = (xx - x) ** 2 + (yy - y) ** 2
    outer = distance_sq <= (radius + outline_width) ** 2
    inner = distance_sq <= radius**2
    image[outer] = outline_rgb
    image[inner] = rgb
    return image


def _white_tip_near_mask(
    image_rgb: np.ndarray,
    mask: np.ndarray,
    previous_xy: tuple[float, float],
    config: StickConfig,
) -> tuple[float, float] | None:
    rgb = np.asarray(image_rgb).astype(np.int16)
    white = (rgb.min(axis=-1) >= config.white_min_value) & (
        np.ptp(rgb, axis=-1) <= config.white_max_chroma
    )
    search = ndimage.binary_dilation(mask, iterations=config.white_search_dilation)
    labels, count = ndimage.label(white & search)
    candidates: list[tuple[float, np.ndarray]] = []
    previous = np.asarray(previous_xy, dtype=np.float64)
    for label_index in range(1, count + 1):
        yx = np.argwhere(labels == label_index)
        if len(yx) < 3:
            continue
        xy = yx[:, ::-1].mean(axis=0)
        candidates.append((float(np.linalg.norm(xy - previous)), xy))
    if not candidates:
        return None
    point = min(candidates, key=lambda item: item[0])[1]
    return float(point[0]), float(point[1])


class ManualSam3KeypointPreprocessor(Sam3EpisodeTrackerPreprocessor):
    """Track manually selected stick tips and draw training-compatible keypoints."""

    def __init__(
        self,
        checkpoint: pathlib.Path,
        *,
        initial_points: Mapping[str, tuple[float, float]],
        stick_config: StickConfig,
        device: str | None,
    ) -> None:
        import torch

        super().__init__(
            checkpoint,
            prompts=("unused: manual spatial prompts",),
            device=device or ("cuda" if torch.cuda.is_available() else "cpu"),
            score_threshold=stick_config.score_threshold,
            mask_threshold=stick_config.mask_threshold,
            min_component_area=3,
            model_input_size=stick_config.model_input_size,
            error_policy="raise",
        )
        self.initial_points = {
            name: (float(point[0]), float(point[1])) for name, point in initial_points.items()
        }
        self.stick_config = stick_config
        self._tip_points = dict(self.initial_points)

    def start_episode(self) -> None:
        super().start_episode()
        self._tip_points = dict(self.initial_points)

    def preprocess(self, images: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
        validated = self._validate_images(images)
        if set(validated) != set(self.initial_points):
            raise ValueError(
                f"Manual stick points must match cameras: {set(self.initial_points)} vs {set(validated)}"
            )
        if self._frame_index == 0:
            masks = {}
            half = float(self.stick_config.prompt_box_half_size)
            for name, image in validated.items():
                x, y = self.initial_points[name]
                height, width = image.shape[:2]
                box = (
                    max(0.0, x - half),
                    max(0.0, y - half),
                    min(width - 1.0, x + half),
                    min(height - 1.0, y + half),
                )
                masks[name] = self._start_tracker_with_spatial_prompt(name, image, (x, y), box)
                if not masks[name].any():
                    raise RuntimeError(f"SAM 3 failed to initialize the selected stick in {name}")
        else:
            masks = {}
            for name, image in validated.items():
                try:
                    masks[name] = self._track_frame(name, image)
                except Exception:
                    logger.exception(
                        "SAM 3 tracking failed for %s frame %d; keeping its previous keypoint",
                        name,
                        self._frame_index,
                    )
                    masks[name] = np.zeros(image.shape[:2], dtype=bool)

        output = {}
        for name, image in validated.items():
            point = _white_tip_near_mask(image, masks[name], self._tip_points[name], self.stick_config)
            if point is None:
                point = self._tip_points[name]
                logger.warning("Stick tip was not found in %s frame %d; keeping its previous keypoint", name, self._frame_index)
            else:
                self._tip_points[name] = point
            output[name] = draw_keypoint(
                image,
                point,
                rgb=self.stick_config.keypoint_rgb,
                radius=self.stick_config.keypoint_radius,
                outline_rgb=self.stick_config.keypoint_outline_rgb,
                outline_width=self.stick_config.keypoint_outline_width,
            )
        self._frame_index += 1
        return output


def select_policy_keypoints_interactively(
    reference_rgb: np.ndarray,
    policy_images: Mapping[str, np.ndarray],
) -> dict[str, tuple[float, float]]:
    """Ask the operator to click the selected white cap once in each policy view."""
    import tkinter as tk

    from PIL import Image
    from PIL import ImageTk

    required = tuple(policy_images)
    if not required:
        raise ValueError("At least one policy image is required for manual stick selection")
    selected: dict[str, tuple[float, float]] = {}
    cancelled = False
    root = tk.Tk()
    root.title("Select the automatically chosen pull-stick target")
    instruction = tk.Label(
        root,
        text=(
            "Reference: automatically selected most-vertical stick. "
            "Click that stick's white cap once in each policy view, then confirm."
        ),
    )
    instruction.pack(padx=10, pady=8)

    reference = Image.fromarray(np.asarray(reference_rgb, dtype=np.uint8))
    reference.thumbnail((720, 405))
    reference_photo = ImageTk.PhotoImage(reference)
    reference_label = tk.Label(root, image=reference_photo, text="External RGB-D reference", compound="top")
    reference_label.image = reference_photo
    reference_label.pack(padx=10, pady=5)

    views = tk.Frame(root)
    views.pack(padx=10, pady=5)
    labels: dict[str, tk.Label] = {}
    status_labels: dict[str, tk.Label] = {}
    photos: dict[str, Any] = {}

    def refresh(name: str) -> None:
        display = np.asarray(policy_images[name], dtype=np.uint8)
        if name in selected:
            display = draw_keypoint(display, selected[name])
        photo = ImageTk.PhotoImage(Image.fromarray(display))
        photos[name] = photo
        labels[name].configure(image=photo)
        status_labels[name].configure(text=f"{name}: {selected.get(name, 'click white cap')}")

    def click(name: str, event: Any) -> None:
        image = np.asarray(policy_images[name])
        widget_width = max(1, labels[name].winfo_width())
        widget_height = max(1, labels[name].winfo_height())
        x = np.clip(event.x * image.shape[1] / widget_width, 0, image.shape[1] - 1)
        y = np.clip(event.y * image.shape[0] / widget_height, 0, image.shape[0] - 1)
        selected[name] = (float(x), float(y))
        refresh(name)
        confirm_button.configure(state="normal" if len(selected) == len(required) else "disabled")

    for column, (name, image) in enumerate(policy_images.items()):
        image_array = np.asarray(image)
        if image_array.ndim != 3 or image_array.shape[-1] != 3:
            root.destroy()
            raise ValueError(f"Policy image {name!r} must be HWC RGB, got {image_array.shape}")
        pane = tk.Frame(views)
        pane.grid(row=0, column=column, padx=8)
        status_labels[name] = tk.Label(pane, text=name)
        status_labels[name].pack()
        labels[name] = tk.Label(pane, cursor="crosshair")
        labels[name].pack()
        labels[name].bind("<Button-1>", lambda event, camera=name: click(camera, event))
        refresh(name)

    def confirm() -> None:
        root.quit()

    def cancel() -> None:
        nonlocal cancelled
        cancelled = True
        root.quit()

    buttons = tk.Frame(root)
    buttons.pack(pady=10)
    confirm_button = tk.Button(buttons, text="Confirm", command=confirm, state="disabled")
    confirm_button.pack(side="left", padx=5)
    tk.Button(buttons, text="Cancel", command=cancel).pack(side="left", padx=5)
    root.bind("<Return>", lambda event: confirm() if len(selected) == len(required) else None)
    root.bind("<Escape>", lambda event: cancel())
    root.protocol("WM_DELETE_WINDOW", cancel)
    logger.info("Waiting for manual white-cap clicks in %s", required)
    root.mainloop()
    root.destroy()
    if cancelled:
        raise KeyboardInterrupt("Stick target selection cancelled by operator")
    return selected

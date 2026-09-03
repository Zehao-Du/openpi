"""Track a tennis ball with SAM3 and replace its circular mask with a square mask.

Run from the repository root:

    uv run --project examples/realman_pika --no-sync python examples/realman_pika/tennis_ball/annotate_square_mask.py \
        --video-path /inspire/hdd/project/robot-dna/baojiachun-CZXS25130063/zehao/wangqiu.mp4 \
        --output-dir /inspire/hdd/project/robot-dna/baojiachun-CZXS25130063/zehao/tmp_dir/wangqiu_tennis_ball_square \
        --overwrite

SAM3 text-detects a circular tennis-ball mask on several candidate frames, selects
the most ball-like instance, and tracks it in both temporal directions. The final
mask is an axis-aligned square centered at the SAM3-mask centroid. Its side length
is the longer side of the SAM3 bounding box multiplied by ``square_scale``.
"""

from __future__ import annotations

# ruff: noqa: E402
import dataclasses
import json
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import av
import cv2
from image_preprocessing import Sam3EpisodeTrackerPreprocessor
from image_preprocessing import clean_mask
import numpy as np
from openpi_client import image_tools
from tqdm.auto import tqdm
import tyro

DEFAULT_VIDEO_PATH = Path("/inspire/hdd/project/robot-dna/baojiachun-CZXS25130063/zehao/wangqiu.mp4")
DEFAULT_CHECKPOINT = Path(__file__).resolve().parents[4] / "foundation_models" / "SAM3"
DEFAULT_OUTPUT_DIR = Path(
    "/inspire/hdd/project/robot-dna/baojiachun-CZXS25130063/zehao/tmp_dir/wangqiu_tennis_ball_square"
)
DEFAULT_PROMPTS = ("tennis ball", "yellow tennis ball", "green tennis ball")


@dataclasses.dataclass
class Args:
    video_path: Path = DEFAULT_VIDEO_PATH
    output_dir: Path = DEFAULT_OUTPUT_DIR
    checkpoint: Path = DEFAULT_CHECKPOINT
    start_frame: int = 0
    end_frame: int | None = None
    # wangqiu.mp4 stores the main camera in the left two thirds.
    crop_x_min: float = 0.0
    crop_x_max: float = 1.0
    crop_y_min: float = 0.0
    crop_y_max: float = 1.0
    image_size: int = 504
    model_input_size: int = 504
    prompts: tuple[str, ...] = DEFAULT_PROMPTS
    candidate_fractions: tuple[float, ...] = (0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95)
    detector_batch_size: int = 3
    score_threshold: float = 0.2
    mask_threshold: float = 0.3
    min_mask_area: int = 40
    max_mask_area_ratio: float = 0.08
    min_circularity: float = 0.35
    max_aspect_ratio: float = 1.8
    square_scale: float = 1.05
    overlay_rgb: tuple[int, int, int] = (255, 0, 255)
    overlay_alpha: float = 0.65
    device: str = "cuda"
    detection_only: bool = False
    overwrite: bool = False


class VideoWriter:
    def __init__(self, path: Path, fps: float, width: int, height: int) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.container = av.open(str(path), "w")
        self.stream = self.container.add_stream("libx264", rate=round(fps))
        self.stream.width = width
        self.stream.height = height
        self.stream.pix_fmt = "yuv420p"

    def add(self, image_rgb: np.ndarray) -> None:
        frame = av.VideoFrame.from_ndarray(image_rgb, format="rgb24")
        for packet in self.stream.encode(frame):
            self.container.mux(packet)

    def close(self) -> None:
        for packet in self.stream.encode():
            self.container.mux(packet)
        self.container.close()


def read_video(args: Args) -> tuple[list[np.ndarray], float, int, tuple[int, int, int, int]]:
    capture = cv2.VideoCapture(str(args.video_path.expanduser()))
    if not capture.isOpened():
        raise ValueError(f"Could not open {args.video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    end = total if args.end_frame is None else args.end_frame
    if not 0 <= args.start_frame < end <= total:
        capture.release()
        raise ValueError(f"Invalid frame interval [{args.start_frame}, {end}) for {total} frames")
    fractions = (args.crop_x_min, args.crop_x_max, args.crop_y_min, args.crop_y_max)
    if not (0.0 <= fractions[0] < fractions[1] <= 1.0 and 0.0 <= fractions[2] < fractions[3] <= 1.0):
        capture.release()
        raise ValueError("Crop fractions must satisfy 0 <= min < max <= 1")

    capture.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame)
    frames: list[np.ndarray] = []
    crop_xyxy: tuple[int, int, int, int] | None = None
    for frame_index in tqdm(range(args.start_frame, end), desc="Decoding", unit="frame"):
        ok, bgr = capture.read()
        if not ok:
            capture.release()
            raise RuntimeError(f"Failed to decode frame {frame_index}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        x0, x1 = round(rgb.shape[1] * args.crop_x_min), round(rgb.shape[1] * args.crop_x_max)
        y0, y1 = round(rgb.shape[0] * args.crop_y_min), round(rgb.shape[0] * args.crop_y_max)
        crop_xyxy = (x0, y0, x1, y1)
        frames.append(image_tools.resize_with_pad(rgb[y0:y1, x0:x1], args.image_size, args.image_size))
    capture.release()
    assert crop_xyxy is not None
    return frames, fps, end, crop_xyxy


def candidate_indices(length: int, fractions: tuple[float, ...]) -> list[int]:
    if length < 1 or not fractions or any(not 0.0 <= value <= 1.0 for value in fractions):
        raise ValueError("candidate_fractions must contain values in [0, 1]")
    return sorted({min(length - 1, round((length - 1) * value)) for value in fractions})


def mask_shape_metrics(mask: np.ndarray) -> tuple[float, float]:
    """Return (circularity, symmetric aspect ratio) for a non-empty binary mask."""
    mask_u8 = np.asarray(mask, dtype=np.uint8)
    points_yx = np.argwhere(mask_u8)
    if len(points_yx) == 0:
        return 0.0, float("inf")
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    perimeter = sum(cv2.arcLength(contour, closed=True) for contour in contours)
    circularity = 0.0 if perimeter <= 0 else float(4.0 * np.pi * len(points_yx) / perimeter**2)
    height, width = np.ptp(points_yx, axis=0) + 1
    aspect_ratio = float(max(width / height, height / width))
    return min(circularity, 1.0), aspect_ratio


def detect_anchor(
    tracker: Sam3EpisodeTrackerPreprocessor,
    frames: list[np.ndarray],
    prompts: tuple[str, ...],
    indices: list[int],
    args: Args,
) -> tuple[int, np.ndarray, float, float, str, float, float]:
    candidates = [(frame_index, prompt) for frame_index in indices for prompt in prompts]
    # ranking, frame index, mask, detector score, prompt, circularity, aspect ratio
    best: tuple[float, int, np.ndarray, float, str, float, float] | None = None
    image_area = args.image_size**2
    for offset in tqdm(range(0, len(candidates), args.detector_batch_size), desc="SAM3 detection", unit="batch"):
        batch = candidates[offset : offset + args.detector_batch_size]
        inputs = tracker._processor(  # noqa: SLF001
            images=[frames[index] for index, _ in batch],
            text=[prompt for _, prompt in batch],
            return_tensors="pt",
            size={"height": args.model_input_size, "width": args.model_input_size},
        ).to(tracker.device)
        sizes = inputs["original_sizes"].detach().cpu().tolist()
        with tracker._torch.inference_mode():  # noqa: SLF001
            outputs = tracker._model(**inputs)  # noqa: SLF001
        results = tracker._processor.post_process_instance_segmentation(  # noqa: SLF001
            outputs,
            threshold=args.score_threshold,
            mask_threshold=args.mask_threshold,
            target_sizes=sizes,
        )
        for (frame_index, prompt), result in zip(batch, results, strict=True):
            masks, scores = result.get("masks"), result.get("scores")
            if masks is None:
                continue
            masks = masks.detach().cpu().numpy() if hasattr(masks, "detach") else np.asarray(masks)
            if scores is None:
                scores = np.ones(len(masks), dtype=np.float32)
            elif hasattr(scores, "detach"):
                scores = scores.detach().cpu().numpy()
            for raw_mask, score in zip(masks, np.asarray(scores), strict=True):
                mask = clean_mask(np.asarray(raw_mask, dtype=bool), args.min_mask_area)
                area = int(mask.sum())
                if area < args.min_mask_area or area > image_area * args.max_mask_area_ratio:
                    continue
                circularity, aspect_ratio = mask_shape_metrics(mask)
                if circularity < args.min_circularity or aspect_ratio > args.max_aspect_ratio:
                    continue
                ranking = float(score) + 0.25 * circularity + 0.15 / aspect_ratio
                if best is None or ranking > best[0]:
                    best = ranking, frame_index, mask, float(score), prompt, circularity, aspect_ratio
    if best is None:
        raise RuntimeError(
            "SAM3 found no plausible tennis-ball mask. Verify that the selected crop contains a tennis ball, "
            "or adjust prompts/score_threshold/min_circularity."
        )
    return best[1], best[2], best[0], best[3], best[4], best[5], best[6]


def track_bidirectionally(
    tracker: Sam3EpisodeTrackerPreprocessor,
    frames: list[np.ndarray],
    anchor_index: int,
    anchor_mask: np.ndarray,
) -> list[np.ndarray]:
    masks: list[np.ndarray | None] = [None] * len(frames)
    masks[anchor_index] = anchor_mask
    tracker.start_episode()
    tracker._start_tracker("image", frames[anchor_index], anchor_mask)  # noqa: SLF001
    for index in tqdm(range(anchor_index + 1, len(frames)), desc="Tracking forward", unit="frame"):
        masks[index] = tracker._track_frame("image", frames[index])  # noqa: SLF001
    tracker.start_episode()
    tracker._start_tracker("image", frames[anchor_index], anchor_mask)  # noqa: SLF001
    for index in tqdm(range(anchor_index - 1, -1, -1), desc="Tracking backward", unit="frame"):
        masks[index] = tracker._track_frame("image", frames[index])  # noqa: SLF001
    empty = np.zeros(frames[0].shape[:2], dtype=bool)
    return [empty.copy() if mask is None else np.asarray(mask, dtype=bool) for mask in masks]


def square_mask_from_object(
    object_mask: np.ndarray,
    *,
    scale: float = 1.0,
) -> tuple[np.ndarray, tuple[float, float] | None, tuple[int, int, int, int] | None]:
    """Create a clipped square centered at the object's pixel centroid."""
    if scale <= 0:
        raise ValueError("scale must be positive")
    object_mask = np.asarray(object_mask, dtype=bool)
    points_yx = np.argwhere(object_mask)
    output = np.zeros_like(object_mask)
    if len(points_yx) == 0:
        return output, None, None
    center_y, center_x = points_yx.mean(axis=0)
    span_y, span_x = np.ptp(points_yx, axis=0) + 1
    side = max(1, int(np.ceil(max(span_x, span_y) * scale)))
    x0 = int(np.floor(center_x - (side - 1) / 2))
    y0 = int(np.floor(center_y - (side - 1) / 2))
    x1, y1 = x0 + side, y0 + side
    clipped_x0, clipped_y0 = max(0, x0), max(0, y0)
    clipped_x1, clipped_y1 = min(object_mask.shape[1], x1), min(object_mask.shape[0], y1)
    output[clipped_y0:clipped_y1, clipped_x0:clipped_x1] = True
    return output, (float(center_x), float(center_y)), (clipped_x0, clipped_y0, clipped_x1, clipped_y1)


def overlay_mask(image: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], alpha: float) -> np.ndarray:
    output = image.copy()
    output[mask] = np.clip(
        np.rint((1.0 - alpha) * output[mask].astype(np.float32) + alpha * np.asarray(color)), 0, 255
    ).astype(np.uint8)
    return output


def render_comparison(
    image: np.ndarray,
    sam_mask: np.ndarray,
    square_mask: np.ndarray,
    center: tuple[float, float] | None,
    color: tuple[int, int, int],
    alpha: float,
) -> np.ndarray:
    original_view = image.copy()
    sam_view = overlay_mask(image, sam_mask, (0, 255, 0), 0.55)
    square_view = overlay_mask(image, square_mask, color, alpha)
    if center is not None:
        point = tuple(round(value) for value in center)
        cv2.circle(sam_view, point, 4, (255, 0, 0), -1, lineType=cv2.LINE_AA)
        cv2.circle(square_view, point, 4, (255, 255, 255), -1, lineType=cv2.LINE_AA)
    for panel, label in zip(
        (original_view, sam_view, square_view), ("original", "SAM3 mask", "square mask"), strict=True
    ):
        cv2.putText(panel, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    return np.concatenate((original_view, sam_view, square_view), axis=1)


def validate_args(args: Args) -> None:
    if not args.video_path.expanduser().is_file():
        raise FileNotFoundError(args.video_path)
    if not args.checkpoint.expanduser().is_dir():
        raise FileNotFoundError(args.checkpoint)
    if not args.prompts:
        raise ValueError("prompts must not be empty")
    if args.image_size < 1 or args.model_input_size < 1 or args.model_input_size % 14:
        raise ValueError("image_size must be positive and model_input_size must be a positive multiple of 14")
    if args.detector_batch_size < 1:
        raise ValueError("detector_batch_size must be positive")
    if not 0.0 <= args.overlay_alpha <= 1.0:
        raise ValueError("overlay_alpha must be in [0, 1]")
    if len(args.overlay_rgb) != 3 or any(not 0 <= value <= 255 for value in args.overlay_rgb):
        raise ValueError("overlay_rgb must contain three values in [0, 255]")


def main(args: Args) -> None:
    validate_args(args)
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and not args.overwrite:
        raise FileExistsError(f"{output_dir} exists; pass --overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)
    frames, fps, source_end, crop_xyxy = read_video(args)
    tracker = Sam3EpisodeTrackerPreprocessor(
        args.checkpoint,
        prompts=(args.prompts[0],),
        device=args.device,
        score_threshold=args.score_threshold,
        mask_threshold=args.mask_threshold,
        min_component_area=args.min_mask_area,
        model_input_size=args.model_input_size,
        error_policy="raise",
    )
    anchor = detect_anchor(
        tracker, frames, args.prompts, candidate_indices(len(frames), args.candidate_fractions), args
    )
    anchor_index, anchor_mask, ranking, detector_score, prompt, circularity, aspect_ratio = anchor
    print(
        f"Selected source frame {args.start_frame + anchor_index}: prompt={prompt!r}, "
        f"score={detector_score:.3f}, circularity={circularity:.3f}, aspect={aspect_ratio:.3f}"
    )
    anchor_square, anchor_center, _ = square_mask_from_object(anchor_mask, scale=args.square_scale)
    anchor_preview = render_comparison(
        frames[anchor_index], anchor_mask, anchor_square, anchor_center, args.overlay_rgb, args.overlay_alpha
    )
    anchor_path = output_dir / "anchor_detection.jpg"
    cv2.imwrite(str(anchor_path), cv2.cvtColor(anchor_preview, cv2.COLOR_RGB2BGR))
    if args.detection_only:
        print(f"Saved detection preview to {anchor_path}")
        return
    sam_masks = track_bidirectionally(tracker, frames, anchor_index, anchor_mask)

    comparison_writer = VideoWriter(output_dir / "comparison.mp4", fps, args.image_size * 3, args.image_size)
    mask_writer = VideoWriter(output_dir / "square_mask.mp4", fps, args.image_size, args.image_size)
    overlay_writer = VideoWriter(output_dir / "square_overlay.mp4", fps, args.image_size, args.image_size)
    annotations: list[dict[str, Any]] = []
    try:
        for local_index, (image, sam_mask) in enumerate(
            tqdm(zip(frames, sam_masks, strict=True), total=len(frames), desc="Rendering", unit="frame")
        ):
            square_mask, center, box = square_mask_from_object(sam_mask, scale=args.square_scale)
            annotations.append(
                {
                    "frame_index": local_index,
                    "source_frame": args.start_frame + local_index,
                    "visible": center is not None,
                    "center_xy": center,
                    "square_xyxy": box,
                    "sam_mask_area": int(sam_mask.sum()),
                    "square_mask_area": int(square_mask.sum()),
                }
            )
            comparison_writer.add(
                render_comparison(image, sam_mask, square_mask, center, args.overlay_rgb, args.overlay_alpha)
            )
            overlay_writer.add(overlay_mask(image, square_mask, args.overlay_rgb, args.overlay_alpha))
            mask_writer.add(np.repeat((square_mask * 255).astype(np.uint8)[..., None], 3, axis=2))
    finally:
        comparison_writer.close()
        mask_writer.close()
        overlay_writer.close()

    payload = {
        "source_video": str(args.video_path.expanduser().resolve()),
        "source_interval": [args.start_frame, source_end],
        "source_crop_xyxy": crop_xyxy,
        "processed_image_size": [args.image_size, args.image_size],
        "square_scale": args.square_scale,
        "anchor": {
            "frame_index": anchor_index,
            "source_frame": args.start_frame + anchor_index,
            "prompt": prompt,
            "detector_score": detector_score,
            "ranking_score": ranking,
            "circularity": circularity,
            "aspect_ratio": aspect_ratio,
        },
        "annotations": annotations,
    }
    with (output_dir / "annotations.json").open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)
        file.write("\n")
    print(f"Saved {len(annotations)} frames to {output_dir}")


if __name__ == "__main__":
    main(tyro.cli(Args))

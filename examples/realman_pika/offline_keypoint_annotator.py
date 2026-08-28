"""Fully automatic offline keypoint annotation with SAM 3.

The annotator searches the latter part of one completed grasp episode for the
wooden stick held by the gripper, initializes SAM 3 from the best text-detected
instance, tracks its mask both forward and backward, and places one keypoint
inside that mask on every valid frame. No click, depth, or TCP projection is
used.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import av
import cv2
from image_preprocessing import Sam3EpisodeTrackerPreprocessor
from image_preprocessing import clean_mask
import numpy as np
from openpi_client import image_tools
from tqdm.auto import tqdm
import tyro


DEFAULT_VIDEO_PATH = Path(
    "/inspire/hdd/project/robot-dna/baojiachun-CZXS25130063/zehao/choumugun.mp4"
)
DEFAULT_CHECKPOINT = Path(__file__).resolve().parents[3] / "foundation_models" / "SAM3"
DEFAULT_OUTPUT_DIR = Path("outputs/choumugun_episode0_keypoints")
DEFAULT_PROMPTS = (
    "wooden stick held by robot gripper",
    "colored stick held by robot gripper",
    "wooden stick in robot gripper",
)
IMAGE_SIZE = 224


@dataclasses.dataclass
class Args:
    video_path: Path = DEFAULT_VIDEO_PATH
    output_dir: Path = DEFAULT_OUTPUT_DIR
    checkpoint: Path = DEFAULT_CHECKPOINT
    start_frame: int = 54
    end_frame: int | None = 276
    crop_x_min: float = 0.0
    crop_x_max: float = 2.0 / 3.0
    prompts: tuple[str, ...] = DEFAULT_PROMPTS
    candidate_fractions: tuple[float, ...] = (0.50, 0.60, 0.70, 0.80, 0.90)
    score_threshold: float = 0.25
    mask_threshold: float = 0.3
    min_mask_area: int = 20
    max_mask_area_ratio: float = 0.2
    device: str = "cuda"
    overwrite: bool = False


class PreviewWriter:
    def __init__(self, path: Path, fps: float) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.container = av.open(str(path), "w")
        self.stream = self.container.add_stream("libx264", rate=round(fps))
        self.stream.width = IMAGE_SIZE
        self.stream.height = IMAGE_SIZE
        self.stream.pix_fmt = "yuv420p"

    def add(self, image: np.ndarray) -> None:
        frame = av.VideoFrame.from_ndarray(image, format="rgb24")
        for packet in self.stream.encode(frame):
            self.container.mux(packet)

    def close(self) -> None:
        for packet in self.stream.encode():
            self.container.mux(packet)
        self.container.close()


def read_episode(args: Args) -> tuple[list[np.ndarray], float, int]:
    capture = cv2.VideoCapture(str(args.video_path.expanduser()))
    if not capture.isOpened():
        raise ValueError(f"Could not open {args.video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    end = total if args.end_frame is None else args.end_frame
    if not 0 <= args.start_frame < end <= total:
        capture.release()
        raise ValueError(f"Invalid frame interval [{args.start_frame}, {end}) for {total} frames")
    if not 0.0 <= args.crop_x_min < args.crop_x_max <= 1.0:
        capture.release()
        raise ValueError("crop-x-min/max must satisfy 0 <= min < max <= 1")
    capture.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame)
    frames: list[np.ndarray] = []
    for frame_index in tqdm(range(args.start_frame, end), desc="Decoding", unit="frame"):
        ok, bgr = capture.read()
        if not ok:
            capture.release()
            raise RuntimeError(f"Failed to decode frame {frame_index}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        x0 = round(rgb.shape[1] * args.crop_x_min)
        x1 = round(rgb.shape[1] * args.crop_x_max)
        frames.append(image_tools.resize_with_pad(rgb[:, x0:x1], IMAGE_SIZE, IMAGE_SIZE))
    capture.release()
    return frames, fps, end


def candidate_indices(length: int, fractions: tuple[float, ...]) -> list[int]:
    if length < 1 or not fractions or any(not 0.0 <= value <= 1.0 for value in fractions):
        raise ValueError("candidate-fractions must be non-empty values in [0, 1]")
    return sorted({min(length - 1, round((length - 1) * value)) for value in fractions})


def detect_anchor(
    tracker: Sam3EpisodeTrackerPreprocessor,
    frames: list[np.ndarray],
    prompts: tuple[str, ...],
    indices: list[int],
    *,
    score_threshold: float,
    min_mask_area: int,
    max_mask_area_ratio: float,
) -> tuple[int, np.ndarray, float, float, str, float]:
    batch_images = [frames[index] for index in indices for _ in prompts]
    batch_prompts = [prompt for _ in indices for prompt in prompts]
    inputs = tracker._processor(  # noqa: SLF001
        images=batch_images,
        text=batch_prompts,
        return_tensors="pt",
        size={"height": IMAGE_SIZE, "width": IMAGE_SIZE},
    ).to(tracker.device)
    sizes = inputs["original_sizes"].detach().cpu().tolist()
    with tracker._torch.inference_mode():  # noqa: SLF001
        outputs = tracker._model(**inputs)  # noqa: SLF001
    results = tracker._processor.post_process_instance_segmentation(  # noqa: SLF001
        outputs,
        threshold=score_threshold,
        mask_threshold=tracker.mask_threshold,
        target_sizes=sizes,
    )

    motion_cache: dict[int, np.ndarray] = {}

    def motion_magnitude(frame_index: int) -> np.ndarray:
        if frame_index not in motion_cache:
            before = max(0, frame_index - 4)
            after = min(len(frames) - 1, frame_index + 4)
            gray_before = cv2.cvtColor(frames[before], cv2.COLOR_RGB2GRAY)
            gray_after = cv2.cvtColor(frames[after], cv2.COLOR_RGB2GRAY)
            flow = cv2.calcOpticalFlowFarneback(
                gray_before, gray_after, None, 0.5, 3, 15, 3, 5, 1.2, 0
            )
            motion_cache[frame_index] = np.linalg.norm(flow, axis=-1)
        return motion_cache[frame_index]

    # ranking, frame, mask, raw detector score, prompt, mean mask motion.
    best: tuple[float, int, np.ndarray, float, str, float] | None = None
    image_area = IMAGE_SIZE * IMAGE_SIZE
    for batch_index, result in enumerate(results):
        frame_index = indices[batch_index // len(prompts)]
        prompt = prompts[batch_index % len(prompts)]
        masks = result.get("masks")
        scores = result.get("scores")
        if masks is None:
            continue
        if hasattr(masks, "detach"):
            masks = masks.detach().cpu().numpy()
        if scores is None:
            scores = np.ones(len(masks), dtype=np.float32)
        elif hasattr(scores, "detach"):
            scores = scores.detach().cpu().numpy()
        for mask, score in zip(np.asarray(masks, dtype=bool), np.asarray(scores), strict=True):
            mask = clean_mask(mask, min_mask_area)
            area = int(mask.sum())
            if area < min_mask_area or area > image_area * max_mask_area_ratio:
                continue
            mean_motion = float(motion_magnitude(frame_index)[mask].mean())
            specificity_bonus = 0.02 * (len(prompts) - batch_index % len(prompts))
            ranking_score = float(score) + specificity_bonus + 0.12 * min(mean_motion, 8.0)
            if best is None or ranking_score > best[0]:
                best = ranking_score, frame_index, mask, float(score), prompt, mean_motion
    if best is None:
        raise RuntimeError(
            "SAM 3 found no plausible held-stick instance; lower --score-threshold "
            "or adjust --candidate-fractions/prompts"
        )
    return best[1], best[2], best[0], best[3], best[4], best[5]


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
    for index in tqdm(range(anchor_index + 1, len(frames)), desc="Tracking forward"):
        masks[index] = tracker._track_frame("image", frames[index])  # noqa: SLF001

    tracker.start_episode()
    tracker._start_tracker("image", frames[anchor_index], anchor_mask)  # noqa: SLF001
    for index in tqdm(range(anchor_index - 1, -1, -1), desc="Tracking backward"):
        masks[index] = tracker._track_frame("image", frames[index])  # noqa: SLF001
    return [np.zeros(frames[0].shape[:2], dtype=bool) if mask is None else mask for mask in masks]


def mask_keypoint(mask: np.ndarray) -> tuple[float, float] | None:
    points_yx = np.argwhere(mask)
    if len(points_yx) == 0:
        return None
    centroid = points_yx.mean(axis=0)
    point_yx = points_yx[np.argmin(np.square(points_yx - centroid).sum(axis=1))]
    return float(point_yx[1]), float(point_yx[0])


def render_frame(image: np.ndarray, mask: np.ndarray, point: tuple[float, float] | None) -> np.ndarray:
    rendered = image.copy()
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(rendered, contours, -1, (0, 255, 0), 1)
    if point is not None:
        center = tuple(round(value) for value in point)
        cv2.circle(rendered, center, 6, (255, 0, 0), -1, lineType=cv2.LINE_AA)
        cv2.circle(rendered, center, 8, (255, 255, 255), 1, lineType=cv2.LINE_AA)
    return rendered


def main(args: Args) -> None:
    if not args.video_path.expanduser().is_file():
        raise FileNotFoundError(args.video_path)
    if not args.checkpoint.expanduser().is_dir():
        raise FileNotFoundError(args.checkpoint)
    if not args.prompts:
        raise ValueError("prompts must not be empty")
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and not args.overwrite:
        raise FileExistsError(f"{output_dir} exists; pass --overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)

    frames, fps, source_end = read_episode(args)
    tracker = Sam3EpisodeTrackerPreprocessor(
        args.checkpoint,
        prompts=(args.prompts[0],),
        device=args.device,
        score_threshold=args.score_threshold,
        mask_threshold=args.mask_threshold,
        min_component_area=args.min_mask_area,
        model_input_size=IMAGE_SIZE,
        error_policy="raise",
    )
    indices = candidate_indices(len(frames), args.candidate_fractions)
    anchor_index, anchor_mask, ranking_score, detection_score, prompt, mean_motion = detect_anchor(
        tracker,
        frames,
        args.prompts,
        indices,
        score_threshold=args.score_threshold,
        min_mask_area=args.min_mask_area,
        max_mask_area_ratio=args.max_mask_area_ratio,
    )
    print(
        f"Selected anchor source frame {args.start_frame + anchor_index}, "
        f"prompt={prompt!r}, detector_score={detection_score:.3f}, "
        f"motion={mean_motion:.3f}, ranking={ranking_score:.3f}, area={anchor_mask.sum()}"
    )
    masks = track_bidirectionally(tracker, frames, anchor_index, anchor_mask)

    anchor_area = float(anchor_mask.sum())
    annotations: list[dict[str, Any]] = []
    writer = PreviewWriter(output_dir / "keypoint_preview.mp4", fps)
    try:
        for local_index, (image, mask) in enumerate(zip(frames, masks, strict=True)):
            point = mask_keypoint(mask)
            area = float(mask.sum())
            area_stability = min(area / anchor_area, anchor_area / area) if area > 0 else 0.0
            confidence = detection_score * area_stability
            annotations.append(
                {
                    "frame_index": local_index,
                    "source_frame": args.start_frame + local_index,
                    "x": None if point is None else point[0] / (IMAGE_SIZE - 1),
                    "y": None if point is None else point[1] / (IMAGE_SIZE - 1),
                    "visible": point is not None,
                    "confidence": confidence,
                    "mask_area": int(area),
                }
            )
            writer.add(render_frame(image, mask, point))
    finally:
        writer.close()

    payload = {
        "source_video": str(args.video_path.expanduser().resolve()),
        "source_interval": [args.start_frame, source_end],
        "crop_x_fraction": [args.crop_x_min, args.crop_x_max],
        "image_size": [IMAGE_SIZE, IMAGE_SIZE],
        "anchor_frame": anchor_index,
        "anchor_source_frame": args.start_frame + anchor_index,
        "anchor_prompt": prompt,
        "anchor_detector_score": detection_score,
        "anchor_motion": mean_motion,
        "anchor_ranking_score": ranking_score,
        "annotations": annotations,
    }
    with (output_dir / "keypoints.json").open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)
        file.write("\n")
    print(f"Saved {len(annotations)} annotations and preview to {output_dir}")


if __name__ == "__main__":
    main(tyro.cli(Args))

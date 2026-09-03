from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
import annotate_square_mask as annotator


def test_square_mask_uses_centroid_and_longest_bbox_side() -> None:
    object_mask = np.zeros((12, 14), dtype=bool)
    object_mask[3:8, 4:11] = True

    square, center, box = annotator.square_mask_from_object(object_mask)

    assert center == (7.0, 5.0)
    assert box == (4, 2, 11, 9)
    assert square.sum() == 49


def test_square_mask_clips_at_image_boundary() -> None:
    object_mask = np.zeros((8, 8), dtype=bool)
    object_mask[0:3, 0:5] = True

    square, center, box = annotator.square_mask_from_object(object_mask)

    assert center == (2.0, 1.0)
    assert box == (0, 0, 5, 4)
    assert square.sum() == 20


def test_square_mask_empty_object() -> None:
    square, center, box = annotator.square_mask_from_object(np.zeros((5, 6), dtype=bool))

    assert not square.any()
    assert center is None
    assert box is None


def test_square_mask_rejects_nonpositive_scale() -> None:
    with pytest.raises(ValueError, match="scale must be positive"):
        annotator.square_mask_from_object(np.ones((2, 2), dtype=bool), scale=0)


def test_mask_shape_metrics_distinguishes_round_and_thin_masks() -> None:
    round_mask = np.zeros((31, 31), dtype=np.uint8)
    import cv2

    cv2.circle(round_mask, (15, 15), 8, 1, -1)
    thin_mask = np.zeros_like(round_mask)
    thin_mask[14:17, 4:27] = 1

    round_circularity, round_aspect = annotator.mask_shape_metrics(round_mask)
    thin_circularity, thin_aspect = annotator.mask_shape_metrics(thin_mask)

    assert round_circularity > thin_circularity
    assert round_aspect == pytest.approx(1.0)
    assert thin_aspect > 5.0

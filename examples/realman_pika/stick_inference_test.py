from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import stick_inference


def test_stick_config_requires_camera_serial_and_calibration() -> None:
    with pytest.raises(ValueError, match="external-camera-serials"):
        stick_inference.StickConfig().validate_hardware()

    config = stick_inference.StickConfig(external_camera_serials=("123",))
    with pytest.raises(ValueError, match="calibration-dir"):
        config.validate_hardware()


def test_stick_config_reports_missing_calibration_file(tmp_path: Path) -> None:
    config = stick_inference.StickConfig(
        external_camera_serials=("123",),
        calibration_dir=tmp_path,
    )

    with pytest.raises(FileNotFoundError, match="T_cam_to_world_123.npy"):
        config.validate_hardware()


def test_select_most_vertical_uses_acute_world_z_angle() -> None:
    axes = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.3, 0.0, 0.95],
            [0.01, 0.0, -1.0],
        ]
    )

    assert stick_inference.select_most_vertical_index(axes) == 2
    assert stick_inference.vertical_angle_deg([0.0, 0.0, -2.0]) == pytest.approx(0.0)


def test_point_cloud_from_mask_deprojects_and_transforms() -> None:
    depth = np.zeros((3, 3), dtype=np.uint16)
    depth[1, 2] = 1000
    mask = depth > 0
    intrinsics = SimpleNamespace(ppx=1.0, ppy=1.0, fx=2.0, fy=2.0)
    camera_to_world = np.eye(4)
    camera_to_world[:3, 3] = (1.0, 2.0, 3.0)

    points = stick_inference.point_cloud_from_mask(
        depth,
        mask,
        intrinsics,
        0.001,
        camera_to_world,
        max_depth_m=2.0,
    )

    np.testing.assert_allclose(points, [[1.5, 2.0, 4.0]])


def test_estimate_axis_and_top_finds_line_direction() -> None:
    z = np.linspace(0.0, 0.2, 100)
    points = np.stack((0.02 * z, np.zeros_like(z), z), axis=1)

    axis, top = stick_inference.estimate_axis_and_top(points)

    assert stick_inference.vertical_angle_deg(axis) < 2.0
    assert top[2] > 0.18


def test_draw_keypoint_uses_training_magenta_and_white_outline() -> None:
    image = np.zeros((21, 21, 3), dtype=np.uint8)

    output = stick_inference.draw_keypoint(image, (10.0, 10.0), radius=3, outline_width=2)

    np.testing.assert_array_equal(output[10, 10], (255, 0, 255))
    np.testing.assert_array_equal(output[10, 15], (255, 255, 255))
    np.testing.assert_array_equal(output[0, 0], (0, 0, 0))


def test_tracking_failure_keeps_previous_keypoint() -> None:
    preprocessor = stick_inference.ManualSam3KeypointPreprocessor.__new__(
        stick_inference.ManualSam3KeypointPreprocessor
    )
    preprocessor.initial_points = {"fisheye": (8.0, 9.0)}
    preprocessor._tip_points = dict(preprocessor.initial_points)  # noqa: SLF001
    preprocessor.stick_config = stick_inference.StickConfig(
        keypoint_radius=2,
        keypoint_outline_width=1,
    )
    preprocessor._frame_index = 1  # noqa: SLF001
    preprocessor._sessions = {"fisheye": object()}  # noqa: SLF001
    preprocessor._track_frame = lambda name, image: (_ for _ in ()).throw(  # type: ignore[method-assign]  # noqa: SLF001
        RuntimeError("lost")
    )

    output = preprocessor.preprocess(
        {"fisheye": np.zeros((20, 20, 3), dtype=np.uint8)}
    )

    np.testing.assert_array_equal(output["fisheye"][9, 8], (255, 0, 255))
    assert preprocessor._tip_points["fisheye"] == (8.0, 9.0)  # noqa: SLF001

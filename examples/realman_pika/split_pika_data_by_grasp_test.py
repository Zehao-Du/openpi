from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import split_pika_data_by_grasp as splitter


def test_detect_grasp_cycles_ignores_startup_and_incomplete_tail() -> None:
    gripper = np.asarray(
        [0.0] * 4
        + [0.097] * 8
        + [0.08] * 2
        + [0.05] * 7
        + [0.097] * 9
        + [0.05] * 6
        + [0.097] * 8
        + [0.0] * 5,
        dtype=np.float64,
    )

    cycles = splitter.detect_grasp_cycles(
        gripper,
        min_state_frames=3,
        post_release_frames=2,
    )

    assert cycles == [
        splitter.GraspCycle(start=4, close=14, release=21, end=24),
        splitter.GraspCycle(start=24, close=30, release=36, end=39),
    ]


def _write_source_episode(episode: Path) -> tuple[np.ndarray, np.ndarray]:
    episode.mkdir(parents=True)
    gripper = np.asarray(
        [0.0] * 2 + [0.097] * 4 + [0.05] * 4 + [0.097] * 5 + [0.05] * 3 + [0.097] * 4,
        dtype=np.float64,
    )
    timestamps = np.arange(len(gripper), dtype=np.float64)
    string_dtype = h5py.string_dtype(encoding="utf-8")
    fisheye_paths = np.asarray([f"camera/fisheye/{index}.jpg" for index in range(len(gripper))], dtype=object)
    wrist_paths = np.asarray([f"camera/wrist/{index}.jpg" for index in range(len(gripper))], dtype=object)

    with h5py.File(episode / "data.hdf5", "w") as file:
        file.attrs["source"] = "test"
        file.create_dataset(splitter.GRIPPER_KEY, data=gripper)
        file.create_dataset("timestamp", data=timestamps, chunks=True)
        file.create_dataset("localization/pose/pika", data=np.arange(len(gripper) * 6).reshape(-1, 6))
        file.create_dataset("camera/color/pikaFisheyeCamera", data=fisheye_paths, dtype=string_dtype)
        file.create_dataset("camera/color/pikaDepthCamera", data=wrist_paths, dtype=string_dtype)
        file.create_dataset("camera/colorIntrinsic/pikaDepthCamera", data=np.eye(3))
        file.create_dataset(splitter.SIZE_KEY, data=len(gripper))
        file.create_dataset("instruction", data="instructions.npy", dtype=string_dtype)

    for relative_path in (*fisheye_paths, *wrist_paths):
        path = episode / str(relative_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(str(relative_path).encode())
    (episode / "instructions.npy").write_bytes(b"instruction-array")
    (episode / "instructions.json").write_text('{"instruction": "collect blocks"}\n')
    return gripper, timestamps


def test_split_dataset_slices_hdf5_and_copies_only_referenced_files(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    source_episode = source_root / "episode7"
    gripper, timestamps = _write_source_episode(source_episode)
    output_root = tmp_path / "output"

    manifest = splitter.split_dataset(
        splitter.Args(
            data_dir=source_root,
            output_dir=output_root,
            min_state_frames=2,
            post_release_frames=1,
        )
    )

    assert [(entry["source_start"], entry["source_end"]) for entry in manifest] == [(2, 12), (12, 20)]
    for output_index, (start, end) in enumerate(((2, 12), (12, 20))):
        output_episode = output_root / f"episode{output_index}"
        with h5py.File(output_episode / "data.hdf5", "r") as file:
            np.testing.assert_array_equal(file[splitter.GRIPPER_KEY][:], gripper[start:end])
            np.testing.assert_array_equal(file["timestamp"][:], timestamps[start:end])
            np.testing.assert_array_equal(file["camera/colorIntrinsic/pikaDepthCamera"][:], np.eye(3))
            assert file[splitter.SIZE_KEY][()] == end - start
            assert file.attrs["source"] == "test"
        assert (output_episode / f"camera/fisheye/{start}.jpg").is_file()
        assert (output_episode / f"camera/wrist/{end - 1}.jpg").is_file()
        assert not (output_episode / "camera/fisheye/0.jpg").exists()
        assert (output_episode / "instructions.npy").is_file()
        assert (output_episode / "instructions.json").is_file()
        assert (output_episode / "split_info.json").is_file()
    assert (output_root / "split_manifest.json").is_file()

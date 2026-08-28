"""Convert double-close-delimited pull-stick recordings to visual-prompt LeRobot data.

This adapter combines ``convert_double_close_pika_data_to_lerobot.py`` splitting
with the established SAM3 + CoTracker visual-prompt conversion pipeline.

Example:
uv run --project examples/realman_pika --no-sync python \
  examples/realman_pika/pull_stick/visualprompt_convert_double_close_pika_data_to_lerobot.py \
  --debug-video-dir /tmp/pull_stick_double_close_keypoints \
  --debug-overwrite --max-recordings 1

Copy an existing compatible dataset to a new repo ID, then append new episodes:

uv run --project examples/realman_pika --no-sync python \
  examples/realman_pika/pull_stick/visualprompt_convert_double_close_pika_data_to_lerobot.py \
  --append-from-repo-id Zehao123/existing_keypoint_dataset \
  --repo-id Zehao123/existing_keypoint_dataset_extended \
  --start-episode 100

Build target D from LeRobot datasets A and B, then append visual-prompt
episodes converted from Pika dataset C:

uv run --project examples/realman_pika --no-sync python \
  examples/realman_pika/pull_stick/visualprompt_convert_double_close_pika_data_to_lerobot.py \
  --append-from-repo-id Zehao123/A \
  --append-lerobot-repo-id Zehao123/B \
  --data-dir /absolute/path/to/pika_C \
  --repo-id Zehao123/D
"""

from __future__ import annotations

# The adapter intentionally replaces two internal extension points in the
# established visual-prompt converter.
# ruff: noqa: SLF001
import dataclasses
import importlib.util
from pathlib import Path
import re
import sys

import h5py
import numpy as np
import tyro

HERE = Path(__file__).resolve().parent


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


splitter = _load_module(
    "pika_double_close_converter",
    HERE / "convert_double_close_pika_data_to_lerobot.py",
)
visualprompt = _load_module(
    "pika_pull_stick_visualprompt_converter",
    HERE / "visualprompt_convert_pika_data_to_lerobot.py",
)

DEFAULT_REPO_ID = "Zehao123/pika_pull_stick_0827_37_gripper_224_224_keypoint"
DEFAULT_TASK_PROMPT = (
    "pull the wooden stick with a ${color} dot on its top and place it on the table"
)
_recursive_find_episode_dirs = visualprompt.pull.base._find_episode_dirs


@dataclasses.dataclass
class Args(visualprompt.Args):
    data_dir: Path = splitter.DEFAULT_DATA_DIR
    repo_id: str = DEFAULT_REPO_ID
    task_prompt: str = DEFAULT_TASK_PROMPT
    append_from_repo_id: str | None = None
    append_lerobot_repo_id: str | None = None
    split: splitter.SplitConfig = dataclasses.field(default_factory=splitter.SplitConfig)


def _find_episode_dirs(data_dir: Path) -> list[Path]:
    """Avoid recursively scanning every raw image for the common flat layout."""

    direct = [path for path in data_dir.glob("episode*") if path.is_dir()]
    if not direct:
        return _recursive_find_episode_dirs(data_dir)
    missing = [path for path in direct if not (path / "data.hdf5").is_file()]
    if missing:
        raise FileNotFoundError(
            f"Found {len(direct)} episode directories under {data_dir}, but {len(missing)} "
            "have no data.hdf5. Run the Pika data_sync.py and data_to_hdf5.py tools first."
        )
    return sorted(direct, key=lambda path: visualprompt.pull.base._episode_sort_key(path, data_dir))


def _warning_tuple(warning: str, episode_dirs: list[Path]) -> tuple[Path, int, int, str]:
    source = next(
        (path for path in episode_dirs if warning.startswith(f"{path}:")),
        episode_dirs[0],
    )
    reason = warning.removeprefix(f"{source}: ")
    match = re.search(r"\[(\d+):(\d+)\]", reason)
    if match:
        start, end = (int(value) for value in match.groups())
    else:
        with h5py.File(source / "data.hdf5", "r") as file:
            start, end = 0, len(file[visualprompt.pull.base.GRIPPER_KEY])
    return source, start, end, reason


def _plan(
    episode_dirs: list[Path],
    config: splitter.SplitConfig,
    strict: bool,  # noqa: FBT001 - called positionally by the reused converter.
) -> tuple[list[tuple], list[tuple[Path, int, int, str]]]:
    """Adapt double-close slices to the visual-prompt converter's legacy tuple."""

    planned, warnings = splitter._plan(episode_dirs, config, strict=strict)
    z_min_by_source: dict[Path, float] = {}
    compatible = []
    for item in planned:
        if item.source_dir not in z_min_by_source:
            with h5py.File(item.source_dir / "data.hdf5", "r") as file:
                pose = np.asarray(file[visualprompt.pull.base.TCP_KEY], dtype=float)
            z_min_by_source[item.source_dir] = float(pose[:, 2].min())
        closed_start, closed_end = max(
            item.sustained_closes,
            key=lambda run: run[1] - run[0],
        )
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
    return compatible, [_warning_tuple(warning, episode_dirs) for warning in warnings]


# Inject double-close discovery and planning while retaining the complete
# visual-prompt implementation and its CLI behavior.
visualprompt.pull.base._find_episode_dirs = _find_episode_dirs
visualprompt.pull._plan = _plan


def _keypoint_color_name(rgb: tuple[int, int, int]) -> str:
    # Preserve the exact, familiar name of the default synthetic keypoint.
    if tuple(rgb) == (255, 0, 255):
        return "magenta"
    return visualprompt.visual._target_rgb_color_name(tuple(rgb))


def _resolve_task_prompt(template: str, rgb: tuple[int, int, int]) -> str:
    color = _keypoint_color_name(rgb)
    prompt = template.replace("${color}", color).replace("{color}", color).strip()
    if not prompt:
        raise ValueError("--task-prompt must not be empty")
    return prompt


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.task_prompt = _resolve_task_prompt(args.task_prompt, args.keypoint.rgb)
    print(f"Task prompt: {args.task_prompt}")
    visualprompt.main(args)

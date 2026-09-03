"""Convert every pull-stick recording with directory-aware splitting.

Recordings whose parent directory name ends in ``gripper`` use the double-close
splitter. All other recordings use the standard pull-stick splitter. The output
is always a LeRobot v2.1 dataset and keypoint radii are dynamic by default.

Example:
uv run --project examples/realman_pika --no-sync python \
  examples/realman_pika/pull_stick/visualprompt_convert_all_pika_data_to_lerobot.py \
  --repo-id Zehao123/pika_pull_stick_all_224_224_keypoint \
  --debug-video-dir /tmp/pull_stick_all_keypoints --debug-overwrite

Validate splitting without loading SAM3 or writing a dataset:
uv run --project examples/realman_pika --no-sync python \
  examples/realman_pika/pull_stick/visualprompt_convert_all_pika_data_to_lerobot.py \
  --test-mode
"""

from __future__ import annotations

# Reuse the existing converters' internal extension points intentionally.
# ruff: noqa: SLF001
import dataclasses
import importlib.util
import json
from pathlib import Path
import re
import shutil
import sys
import tempfile

import h5py
import numpy as np
import tyro

HERE = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = Path(
    "/inspire/hdd/project/robot-dna/baojiachun-CZXS25130063/zehao/dataset/pika/pull_stick"
)
DEFAULT_REPO_ID = "Zehao123/pi05_keypoint_pika_pull_stick_1437_v21"
DEFAULT_TASK_PROMPT = "pull the wooden stick with a ${color} dot on its top and place it on the table"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


visualprompt = _load_module(
    "pika_pull_stick_all_visualprompt_converter",
    HERE / "visualprompt_convert_pika_data_to_lerobot.py",
)
double_close = _load_module(
    "pika_pull_stick_all_double_close_converter",
    HERE / "convert_double_close_pika_data_to_lerobot.py",
)
v21_merger = _load_module(
    "pika_pull_stick_v21_merger",
    HERE.parent / "merge_lerobot_v21_shards.py",
)
_standard_plan = visualprompt.pull._plan


@dataclasses.dataclass
class SplitConfig:
    standard: visualprompt.pull.SplitConfig = dataclasses.field(default_factory=visualprompt.pull.SplitConfig)
    gripper: double_close.SplitConfig = dataclasses.field(default_factory=double_close.SplitConfig)


@dataclasses.dataclass
class Args(visualprompt.Args):
    data_dir: Path = DEFAULT_DATA_DIR
    repo_id: str = DEFAULT_REPO_ID
    task_prompt: str = DEFAULT_TASK_PROMPT
    append_from_repo_id: str | None = None
    append_lerobot_repo_id: str | None = None
    split: SplitConfig = dataclasses.field(default_factory=SplitConfig)
    keypoint: visualprompt.KeypointConfig = dataclasses.field(
        default_factory=lambda: visualprompt.KeypointConfig(dynamic_radius=True)
    )


def _uses_double_close(episode_dir: Path) -> bool:
    return episode_dir.parent.name.endswith("gripper")


def _double_warning_tuple(warning: str, episode_dirs: list[Path]) -> tuple[Path, int, int, str]:
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


def _adapt_double_close_plan(episode_dirs: list[Path], config: double_close.SplitConfig, strict: bool):
    planned, warnings = double_close._plan(episode_dirs, config, strict=strict)
    z_min_by_source: dict[Path, float] = {}
    compatible = []
    for item in planned:
        if item.source_dir not in z_min_by_source:
            with h5py.File(item.source_dir / "data.hdf5", "r") as file:
                pose = np.asarray(file[visualprompt.pull.base.TCP_KEY], dtype=float)
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
    strict: bool,  # noqa: FBT001 - the reused converter calls this positionally.
) -> tuple[list[tuple], list[tuple[Path, int, int, str]]]:
    standard_dirs = [path for path in episode_dirs if not _uses_double_close(path)]
    gripper_dirs = [path for path in episode_dirs if _uses_double_close(path)]
    planned: list[tuple] = []
    warnings: list[tuple[Path, int, int, str]] = []
    if standard_dirs:
        standard_planned, standard_warnings = _standard_plan(standard_dirs, config.standard, strict)
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


def _keypoint_color_name(rgb: tuple[int, int, int]) -> str:
    if tuple(rgb) == (255, 0, 255):
        return "magenta"
    return visualprompt.visual._target_rgb_color_name(tuple(rgb))


def _resolve_task_prompt(template: str, rgb: tuple[int, int, int]) -> str:
    color = _keypoint_color_name(rgb)
    prompt = template.replace("${color}", color).replace("{color}", color).strip()
    if not prompt:
        raise ValueError("--task-prompt must not be empty")
    return prompt


def _output_signature(output: Path) -> tuple[int, int, int] | None:
    info_path = output / "meta" / "info.json"
    if not info_path.is_file():
        return None
    stat = info_path.stat()
    return stat.st_ino, stat.st_mtime_ns, stat.st_size


def _ensure_v21(output: Path) -> None:
    info_path = output / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"Converted dataset metadata does not exist: {info_path}")
    with info_path.open(encoding="utf-8") as file:
        version = json.load(file).get("codebase_version")
    if version == "v2.1":
        return
    if version != "v3.0":
        raise ValueError(f"Expected LeRobot v2.1 or v3.0 output, got {version!r}")

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}-v21-", dir=output.parent) as temporary_directory:
        temporary_root = Path(temporary_directory)
        converted = temporary_root / "converted"
        original = temporary_root / "original"
        visualprompt.visual._convert_lerobot_v3_to_v21(output, converted)
        manifest = output / "pull_keypoint_manifest.json"
        if manifest.is_file():
            shutil.copy2(manifest, converted / manifest.name)
        output.rename(original)
        try:
            converted.rename(output)
        except BaseException:
            original.rename(output)
            raise
    print(f"Converted staged LeRobot v3.0 output to v2.1: {output}")


def _push_to_hub(repo_id: str, output: Path) -> None:
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id, repo_type="dataset", exist_ok=True)
    api.upload_folder(repo_id=repo_id, repo_type="dataset", folder_path=output)


def _append_v21_shards(args: Args, output: Path) -> None:
    if args.append_lerobot_repo_id is not None and args.append_from_repo_id is None:
        raise ValueError("--append-lerobot-repo-id requires --append-from-repo-id")
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists: {output}. Pass --overwrite to replace it.")
    original_base_hf_home = visualprompt.pull.base.HF_LEROBOT_HOME
    lerobot_metadata = sys.modules["lerobot.datasets.dataset_metadata"]
    original_metadata_hf_home = lerobot_metadata.HF_LEROBOT_HOME

    original_hf_home = visualprompt.pull.HF_LEROBOT_HOME
    append_repo_ids = tuple(
        repo_id
        for repo_id in (args.append_from_repo_id, args.append_lerobot_repo_id)
        if repo_id is not None
    )
    if any(not repo_id.strip() for repo_id in append_repo_ids):
        raise ValueError("Append repo IDs must not be empty")
    append_roots = tuple((original_hf_home / repo_id.strip()).resolve() for repo_id in append_repo_ids)
    for root in append_roots:
        if not (root / "meta" / "info.json").is_file():
            raise FileNotFoundError(f"Append source dataset does not exist: {root}")

    output.parent.mkdir(parents=True, exist_ok=True)
    saved_append_from = args.append_from_repo_id
    saved_append_lerobot = args.append_lerobot_repo_id
    try:
        with tempfile.TemporaryDirectory(prefix=f".{output.name}-new-v21-", dir=output.parent) as temporary:
            temporary_hf_home = Path(temporary)
            staged_output = temporary_hf_home / args.repo_id
            visualprompt.pull.HF_LEROBOT_HOME = temporary_hf_home
            visualprompt.pull.base.HF_LEROBOT_HOME = temporary_hf_home
            lerobot_metadata.HF_LEROBOT_HOME = temporary_hf_home
            args.append_from_repo_id = None
            args.append_lerobot_repo_id = None
            visualprompt.main(args)
            _ensure_v21(staged_output)
            v21_merger.main(
                v21_merger.Args(
                    inputs=tuple(str(root) for root in (*append_roots, staged_output)),
                    repo_id=args.repo_id,
                    output_root=output,
                    overwrite=args.overwrite,
                )
            )
    finally:
        visualprompt.pull.HF_LEROBOT_HOME = original_hf_home
        visualprompt.pull.base.HF_LEROBOT_HOME = original_base_hf_home
        lerobot_metadata.HF_LEROBOT_HOME = original_metadata_hf_home
        args.append_from_repo_id = saved_append_from
        args.append_lerobot_repo_id = saved_append_lerobot


def main(args: Args) -> None:
    args.keypoint.dynamic_radius = True
    args.task_prompt = _resolve_task_prompt(args.task_prompt, args.keypoint.rgb)
    print(f"Task prompt: {args.task_prompt}")
    print("Dynamic keypoint radius: enabled")

    visualprompt.pull._plan = _plan
    output = visualprompt.pull.HF_LEROBOT_HOME / args.repo_id
    before = _output_signature(output)
    push_to_hub = args.push_to_hub
    args.push_to_hub = False
    if args.test_mode or (args.append_from_repo_id is None and args.append_lerobot_repo_id is None):
        visualprompt.main(args)
    else:
        _append_v21_shards(args, output)

    if args.test_mode or args.rewrite_existing_prompt:
        return
    after = _output_signature(output)
    if after is None or after == before:
        return
    _ensure_v21(output)
    if push_to_hub:
        _push_to_hub(args.repo_id, output)


if __name__ == "__main__":
    main(tyro.cli(Args))

# Repository Guidelines

## 环境

conda 环境使用 zehao_openpi
运行指令时使用uv，格式为 ```uv run --project xxx```
预计运行时间较长的数据处理、训练或评测任务必须在 `tmux` 会话中启动，禁止依赖前台终端、`nohup` 或临时 `setsid` 进程；启动后应记录会话名，并通过 `tmux capture-pane` 或日志检查进度和最终状态。

## pikarealman 开发

这个repo的代码开发主要集中于 `examples/realman_pika`，职责为
1. 进行数据转换，将pika采集的数据转换成可供pi05微调的lerobot格式数据
    - pi05 微调需要 v2.1 lerobot 数据集格式，每个数据转换脚本均需兼容 v3.0 与 v2.1 输入，输出 v2.1 数据集
    - LeRobot v3 环境中的 `LeRobotDataset` 不能直接加载 v2.1 数据集，否则会抛出 `BackwardCompatibilityError`。不能只修改 `meta/info.json` 中的版本号，因为 v2.1 与 v3.0 的 parquet、metadata 和视频布局并不兼容。
    - 将新数据 append 到 v2.1 image 数据集时，应先把新数据独立写入 v3 staging，转成 v2.1 后，再使用 `examples/realman_pika/merge_lerobot_v21_shards.py` 做 parquet 级合并；禁止为了 append 而用 v3 loader 打开 v2.1 源数据。合并必须保持图片字节不重新编码，重建连续的 episode/frame/global index，并保留对应的 keypoint 或 recolor manifest。
    - v2.1 合并前必须校验 `fps`、`robot_type`、features、tasks 和数据路径等 metadata 一致性；日志中的 episode 切分 warning 与 `BackwardCompatibilityError` 是两类独立问题，不应混为转换失败原因。
    - 若 staging 通过临时 `HF_LEROBOT_HOME` 隔离输出，必须同步处理已导入模块缓存的路径（包括底层 writer 使用的 `lerobot.datasets.dataset_metadata.HF_LEROBOT_HOME`），或显式向 writer 传入 `root`；所有临时全局值必须在 `finally` 中恢复，避免误写正式数据目录。
    - 每个任务的数据转换脚本存放于 `examples/realman_pika/task_name` 目录下，如 `examples/realman_pika/collect_block`
2. 推理端部署，conda环境为zehao_lerobot，主要由 `examples/realman_pika/main.py` 负责


### 文件格式

`examples/realman_pika/README.md` 保持简短
`examples/realman_pika` 下每个可单独运行的文件，开头都要有运行指令示意

任务子目录中的脚本（例如 `examples/realman_pika/collect_block/foo.py`）直接运行时，Python 默认只把脚本所在的任务子目录加入 `sys.path`。`uv run --project examples/realman_pika` 只负责选择项目环境，不会自动把 `examples/realman_pika` 加入模块搜索路径。因此，若脚本需要导入 `examples/realman_pika` 根目录的共享模块，必须在导入共享模块前显式加入父目录，例如：

```python
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
```

否则 job 中会出现 `ModuleNotFoundError: No module named 'image_preprocessing'` 一类错误。开发后必须使用 job 中相同的 `uv run --project ... --no-sync python path/to/script.py` 调用方式做最小启动或 `--help` 验证，不能只依赖 IDE、测试进程或手工设置的 `PYTHONPATH`。

补丁应用产生的 `*.rej` 和 `*.orig` 文件仅用于排查或恢复未合入的片段；等当前任务完成并确认对应改动已存在后，应及时删除
每次开发完新的数据集处理脚本，都必须选取一个 episode 实际运行，并将处理前后对比视频保存到 `/inspire/hdd/project/robot-dna/baojiachun-CZXS25130063/zehao/tmp_dir`；检查视频确认处理效果后，开发才算完成。

lerobot 数据集命名格式为 "模型_(visual prompt)_数采方式_任务_episode条数_版本(可能为时间)"
其中 `episode条数` 必须按数据处理前的原始 episode（或 recording）数量填写，不按切分、过滤或转换后的输出 episode 数量填写。例如原始输入有 200 条 recording，即使处理后生成 1437 个 episode，命名中仍应写 `200`。
模型命名格式与lerobot保持一致

## visual prompt 

现在有两种 visual prompt 实现
1. recolor，即把image中某些物体替换为一种特定的颜色
2. keypoint，即在image中某些特定地方打上关键点

## 训练脚本

`scripts/job_on_qz` 目录下的文件目的为方便copy到命令行执行，没有格式要求，一般为人工更改
`scripts/job_on_qz` 目录下的目录格式为 `task_name/模型_(visual prompt)_数采方式_任务_episode条数_版本(可能为时间)`，其中 `episode条数` 同样按处理前的原始 episode（或 recording）数量填写。目录目的为交 job 时直接运行，但是交 job 只有交互式截面，由 agent 写，人类提交。
所有可能覆盖已有输出的脚本只能通过显式 `--overwrite` 参数确认覆盖，禁止使用 `input()` 或其他方式等待交互确认；交 job 的命令需要覆盖时必须直接传入 `--overwrite`。

### config文件

config 文件位于 `openpi/src/openpi/training/config.py`，写好训练脚本之后一定要先在本机器上过一遍 debug 版本的运行 （debug版本即只处理一个episode）

### 长时间数据处理的 GPU keepalive

若单个 GPU job 内的数据转换和 norm stats 计算预计超过 3 小时，为避免 GPU 利用率低于 40% 被自动停止，可仅在预处理阶段运行 `scripts/job_on_qz/gpu_keepalive.py`：

- 使用 `setsid uv run --project . python scripts/job_on_qz/gpu_keepalive.py &` 启动，使 keepalive 位于独立进程组并覆盖所有可见 GPU。
- job 必须保存进程组 ID，并注册 `EXIT` trap；数据转换和 norm stats 完成后，对整个进程组发送 `TERM`，必要时发送 `KILL`，确认完全退出后才能启动训练。
- keepalive 不得与正式训练同时运行。新 job 接入后需在本机通过 `nvidia-smi pmon` 验证每张卡的 SM 利用率超过 40%，并验证退出后无残留 worker。

### 交 job 的格式为

```sh
#!/bin/bash

/bin/bash -lc '
cd /inspire/hdd/project/robot-dna/baojiachun-CZXS25130063/zehao/openpi \
&& source /inspire/hdd/project/robot-dna/baojiachun-CZXS25130063/miniconda3/etc/profile.d/conda.sh \
&& conda activate zehao_openpi \
&& export OPENPI_ROOT=/inspire/hdd/project/robot-dna/baojiachun-CZXS25130063/zehao/openpi \
&& export PATH="/inspire/hdd/project/robot-dna/baojiachun-CZXS25130063/zehao/uv-bin:$PATH" \
&& export HF_LEROBOT_HOME=/inspire/hdd/project/robot-dna/baojiachun-CZXS25130063/zehao/dataset/lerobot \
&& export WANDB_MODE=offline \
&& environment=realmanpika_umi \
&& LogName=realmanpika_umi_CollectBlocks_8workers \
&& 数据处理命令 \
&& 计算 norm 命令 \
&& 训练命令 \
'
```


## Project Structure & Module Organization

Core Python code lives in `src/openpi/`: models are under `models/` and `models_pytorch/`, robot adapters under `policies/`, training code under `training/`, and inference serving under `serving/`. The lightweight client is a separate workspace package in `packages/openpi-client/`. Use `scripts/` for training, statistics, conversion, and serving entry points. Platform-specific demonstrations and Docker setups belong in `examples/`; longer operational notes belong in `docs/`. Treat `third_party/` as vendored code and avoid unrelated edits there. Tests are colocated with implementation as `*_test.py` files. Generated checkpoints and local datasets should not be committed.

## Build, Test, and Development Commands

- `git submodule update --init --recursive` initializes bundled dependencies.
- `GIT_LFS_SKIP_SMUDGE=1 uv sync --all-extras --dev` creates the Python 3.11+ development environment used by CI.
- `uv run pytest --strict-markers -m "not manual"` runs the standard test suite.
- `uv run pytest src/openpi/policies/policy_test.py` runs one focused test module.
- `uv run ruff check .` checks lint and import rules; add `--fix` for safe automatic fixes.
- `uv run ruff format .` formats Python code.
- `uv run pre-commit run --all-files` reproduces the repository-wide pre-commit checks.

Training and inference are GPU-heavy. For example, compute statistics with `uv run scripts/compute_norm_stats.py --config-name pi05_libero`, then consult `README.md` for the matching training or serving command.

## Coding Style & Naming Conventions

Use four-space indentation, Python 3.11 syntax, type annotations for public APIs, and a 120-character line limit. Ruff defines formatting, lint, and import ordering; do not hand-format against it. Use `snake_case` for modules, functions, and variables, `PascalCase` for classes, and descriptive configuration names such as `pi05_libero`. Keep imports single-line where Ruff requires them.

## Testing Guidelines

Use pytest and place tests beside the code they exercise, named `*_test.py`; name test functions `test_<behavior>`. Add regression tests for bug fixes and focused unit tests for transforms, policies, loaders, or client serialization. Mark hardware-, dataset-, or operator-dependent checks with `@pytest.mark.manual`; CI excludes that marker. No numeric coverage threshold is configured, so prioritize meaningful changed-path coverage.

## Commit & Pull Request Guidelines

Recent commits use short, imperative summaries such as `Add visual-prompt training configuration`; some include an issue or PR number. Keep commits focused and avoid mixing vendored or generated changes with source edits. Pull requests should have a clear title and description, explain motivation and validation, link relevant issues, and include logs or screenshots when behavior is visual or hardware-facing. Before submission, run tests, Ruff, and pre-commit; discuss large robot or environment additions in an issue or GitHub Discussion first.

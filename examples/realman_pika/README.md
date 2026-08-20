# RealMan + Pika hardware client

This client connects the RealMan-Pika hardware implementation to an OpenPI
websocket policy server. The policy server and robot client intentionally use
separate Python environments: OpenPI requires Python 3.11 and NumPy 1.x, while
the RealMan-Pika LeRobot driver requires Python 3.12 and NumPy 2.x.

The client project pins the robot driver to a reproducible commit in
`Zehao-Du/lerobot`; a sibling LeRobot checkout and `PYTHONPATH` are not needed.
On the robot computer, clone this OpenPI repository and create the client
environment:

```bash
cd /path/to/openpi
uv sync --project examples/realman_pika
```

Start the OpenPI policy server on the inference computer using the Pika training
config and checkpoint:

```bash
uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config <pi05-realman-pika-config> \
  --policy.dir /path/to/checkpoint
```

On the robot computer, first run one query without executing any action:

```bash
uv run --project examples/realman_pika python examples/realman_pika/main.py \
  --host <policy-server-ip> \
  --prompt "pick all blocks into the drawer"
```

Inspect the printed action chunk. Then explicitly enable execution; every chunk
still requires terminal confirmation by default:

```bash
uv run --project examples/realman_pika python examples/realman_pika/main.py \
  --host <policy-server-ip> \
  --prompt "pick all blocks into the drawer" \
  --execute
```

The current OpenPI output transform returns absolute TCP poses, so the client
defaults to `--server-action-mode absolute` and converts them back to Pika-local
deltas before calling `RealmanPika.send_action()`. If the policy output pipeline
is later changed to return deltas directly, use `--server-action-mode delta`.

Camera and state mapping:

- Pika fisheye → `observation/image`
- RealSense RGB → `observation/wrist_image`
- RealMan TCP `eef_{x,y,z,rx,ry,rz}.pos` is converted to an absolute Pika
  gripper rotvec pose; `gripper.pos` is appended to form `observation/state`.

The robot observation must expose an absolute RealMan TCP pose in the same
base/world frame used by the training data. The client applies the fixed
RealMan-TCP-to-Pika-gripper extrinsic before sending the state to the policy.

## Optional SAM 3 recoloring

The robot client can use SAM 3 to segment configurable text-prompted objects.
Both camera images are first resize-padded to the policy resolution (224×224
by default), then segmented and recolored before they are sent to the policy.
It is disabled by default. The model is loaded once and reused for the entire
rollout; fisheye and RealSense images, including all configured prompts, share
one batched model forward per policy request.

Download a pinned SAM 3 revision into the default local checkpoint directory:

```bash
hf download facebook/sam3 \
  --revision 3c879f39826c281e95690f02c7821c4de09afae7 \
  --local-dir ../foundation_models/SAM3
```

Alternatively, point `--sam3.checkpoint` at an existing local Transformers
checkpoint. Enable the default `pink block` to blue transformation with:

```bash
uv run --project examples/realman_pika python examples/realman_pika/main.py \
  --host <policy-server-ip> \
  --prompt "pick all blocks into the drawer" \
  --sam3.enabled \
  --sam3.checkpoint ../foundation_models/SAM3
```

Prompts, color, device, thresholds, blending, and component filtering can all
be overridden. For example:

```bash
uv run --project examples/realman_pika python examples/realman_pika/main.py \
  --host <policy-server-ip> \
  --sam3.enabled \
  --sam3.prompts "pink block" "pink cube" \
  --sam3.target-rgb 0 0 255 \
  --sam3.device cuda \
  --sam3.score-threshold 0.5 \
  --sam3.mask-threshold 0.5 \
  --sam3.alpha 0.9 \
  --sam3.min-component-area 64 \
  --sam3.model-input-size 224
```

If SAM 3 cannot be imported or its checkpoint cannot be loaded, client startup
fails. A runtime inference error is logged and that request falls back to the
original camera image. SAM 3 requires Transformers 5.4–5.5, which is installed
only in this Python 3.12 robot-client project through LeRobot's
`transformers-dep` extra; the OpenPI server environment remains on its pinned
Transformers 4.53.2.

## Split collect-blocks recordings by grasp

Before applying offline visual prompts, split each long `collect_blocks`
recording into episodes that contain exactly one completed pick-and-place. The
splitter detects stable gripper `open -> closed -> open` cycles, slices every
time-aligned HDF5 dataset, preserves static calibration data, and copies only
the image/depth files used by each slice.

Inspect all detected boundaries without writing output first:

```bash
uv run --project examples/realman_pika python \
  examples/realman_pika/split_pika_data_by_grasp.py \
  --data-dir /absolute/path/to/collect_blocks \
  --output-dir /absolute/path/to/collect_blocks_single_grasp \
  --dry-run
```

Remove `--dry-run` to write globally renumbered `episode0...episodeM-1`
directories. Each output episode contains `split_info.json`, and the output
root contains `split_manifest.json`. The default open/closed thresholds are
`0.085` and `0.075`, with three stable frames required and ten frames retained
after release; use `--help` to see the corresponding overrides. Existing
output is never replaced unless `--overwrite` is supplied.

## Offline visual-prompt dataset conversion

The visual-prompt converter accepts the original long `collect_blocks`
recordings directly. It detects each completed grasp cycle in memory and
writes one final LeRobot episode per cycle, so no intermediate split dataset
is required. First, inspect all proposed slices and automatically inferred
grasp colors without loading SAM 3:

```bash
uv run --project examples/realman_pika python \
  examples/realman_pika/visualprompt_convert_pika_data_to_lerobot.py \
  --data-dir /absolute/path/to/collect_blocks \
  --classify-only
```

For each output episode, the converter examines three RealSense frames before
the recorded gripper-close transition, classifies the centered target as red,
green, blue, or pink, and rejects low-confidence classifications. Run the full
conversion with (the shown `--data-dir` is also the script default):

```bash
uv run --project examples/realman_pika python \
  examples/realman_pika/visualprompt_convert_pika_data_to_lerobot.py \
  --data-dir /inspire/hdd/project/robot-dna/baojiachun-CZXS25130063/zehao/dataset/pika/collect_blocks \
  --sam3.checkpoint ../foundation_models/SAM3 \
  --sam3.cross-camera-mapping /absolute/path/to/realsense_to_fisheye_mapping.json \
  --sam3.device cuda
```

The default output repo ID is
`Zehao123/pika_collect_blocks_224_224_visualprompt`. Override it with
`--repo-id`. The inferred color fills the `{color}` placeholder in the SAM 3
prompt template, for example `green block`. The same loaded model is reused as
the prompt changes between episodes. On the first frame of each episode, SAM 3
runs text detection once. The RealSense mask initializes its tracker directly;
its centroid and outline are also projected through the configured polynomial
camera mapping to create a positive point and bounding-box prompt for the
fisheye tracker. If projection or the spatial prompt fails, conversion falls
back to the fisheye text-detector mask. Every later frame uses pure tracker
propagation without a per-frame spatial prompt. If the fisheye mask falls below
50% of its recent reference area, the converter runs text detection once,
restarts the fisheye tracker, and resumes propagation. Re-detection has a
15-frame cooldown; configure these values with `--sam3.redetect-area-ratio` and
`--sam3.redetect-cooldown-frames`. Tracker state is discarded at the episode
boundary. Both cameras are resize-padded to 224×224 before SAM 3
and the selected block is recolored to the fixed `--sam3.target-rgb` value,
blue by default. Native 640×480 mapping coordinates are converted to and from
the padded 224×224 coordinates automatically. The mapping defaults to
`/inspire/hdd/project/robot-dna/baojiachun-CZXS25130063/zehao/pika/dataset_pika/realsense_to_fisheye_mapping.json`.

Each output frame also receives a color-specific language task such as
`grasp the green block and place it into the drawer`. The task keeps the
original block color even though the corresponding pixels are changed to the
fixed visual-prompt color.

To generate only the first single-grasp episode for visual inspection, use a
separate output repo ID:

```bash
uv run --project examples/realman_pika python \
  examples/realman_pika/visualprompt_convert_pika_data_to_lerobot.py \
  --max-episodes 1 \
  --repo-id Zehao123/pika_collect_blocks_visualprompt_preview \
  --preview-video /absolute/path/to/visualprompt_preview.mp4
```

The offline converter uses CUDA by default. Override it with
`--sam3.device cpu` only when a GPU is unavailable. The preview is a 30 FPS
2×2 comparison: original fisheye and RealSense images on the left, with their
SAM 3 recolored versions on the right. The color-specific task is overlaid at
the bottom. An existing video is never overwritten.

The converter processes eight frames (16 camera images) per forward by
default; use `--sam-batch-size` to change this. Each completed dataset contains
`grasp_color_manifest.json` with source slice boundaries, inferred color,
confidence, reference frames, per-color scores, SAM 3 prompt, and task prompt
for every episode. Any color
classification or SAM 3 error aborts conversion so an output dataset cannot
silently contain incorrectly processed images. Use `--test-mode` to validate
episode discovery, grasp-cycle planning, and sliced state arrays without
loading SAM 3. The standalone splitter above remains useful when an explicit
single-grasp Pika dataset is needed for inspection or another pipeline.

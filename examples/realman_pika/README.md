# RealMan + Pika hardware client

This client connects the RealMan-Pika hardware implementation from the sibling
LeRobot repository to an OpenPI websocket policy server.

The required robot implementation currently lives on the `myfork/main` branch
of the LeRobot repository. Use that checkout on the robot computer and install
the small runtime dependencies:

```bash
cd /path/to/lerobot
git switch main
git pull myfork main
pip install -e '.[intelrealsense]'
pip install tyro scipy websockets msgpack
```

`openpi-client` currently pins NumPy below version 2 while recent LeRobot pins
NumPy 2. To avoid changing the working robot environment, expose the client
source directly instead of installing its package:

```bash
export PYTHONPATH=/path/to/openpi/packages/openpi-client/src:/path/to/lerobot/src:$PYTHONPATH
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
python examples/realman_pika/main.py \
  --host <policy-server-ip> \
  --prompt "pick all blocks into the drawer"
```

Inspect the printed action chunk. Then explicitly enable execution; every chunk
still requires terminal confirmation by default:

```bash
python examples/realman_pika/main.py \
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
- `eef_{x,y,z,rx,ry,rz}.pos` plus `gripper.pos` → `observation/state`

The robot driver exposes TCP state relative to the pose captured at connection,
whereas the current converted training dataset stores absolute localization
poses. This representation mismatch must be resolved before evaluating policy
quality; query-only mode is intended to validate transport and shape contracts
first.

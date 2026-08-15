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

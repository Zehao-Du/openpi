"""Keep visible GPUs active while the job performs CPU-heavy preprocessing.

Run from the repository root with:
uv run --project . python scripts/job_on_qz/pull_stick/pi05_keypoint_pika_pull_stick_1437_v21/gpu_keepalive.py
"""

from __future__ import annotations

import multiprocessing as mp
import signal
import time


def _worker(device_index: int, stop: mp.synchronize.Event) -> None:
    import torch

    torch.cuda.set_device(device_index)
    # A moderate allocation with a duty cycle above the cluster's 40% threshold.
    matrix = torch.randn((4096, 4096), device="cuda", dtype=torch.float16)
    while not stop.is_set():
        active_until = time.monotonic() + 1.3
        while time.monotonic() < active_until and not stop.is_set():
            torch.mm(matrix, matrix)
        torch.cuda.synchronize()
        stop.wait(0.7)


def main() -> None:
    import torch

    device_count = torch.cuda.device_count()
    if device_count == 0:
        raise RuntimeError("No visible CUDA devices")

    context = mp.get_context("spawn")
    stop = context.Event()

    def request_stop(_signum: int, _frame: object) -> None:
        stop.set()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    workers = [context.Process(target=_worker, args=(index, stop)) for index in range(device_count)]
    for worker in workers:
        worker.start()
    try:
        while not stop.wait(1.0):
            if any(not worker.is_alive() for worker in workers):
                raise RuntimeError("A GPU keepalive worker exited unexpectedly")
    finally:
        stop.set()
        for worker in workers:
            worker.join(timeout=10)
        for worker in workers:
            if worker.is_alive():
                worker.terminate()


if __name__ == "__main__":
    main()

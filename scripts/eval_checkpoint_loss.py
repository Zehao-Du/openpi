"""Evaluate a JAX checkpoint's flow-matching loss on its configured dataset."""

from __future__ import annotations

import dataclasses
import logging
import pathlib
import time

import jax
import jax.numpy as jnp
import numpy as np
import tqdm
import tyro

import openpi.models.model as _model
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader


def main(
    config_name: str,
    checkpoint_dir: pathlib.Path,
    *,
    batch_size: int = 2,
    num_batches: int = 64,
    num_workers: int = 2,
    seed: int = 42,
    shuffle: bool = True,
) -> None:
    """Report loss using checkpoint EMA parameters and checkpoint normalization stats."""
    checkpoint_dir = checkpoint_dir.resolve()
    params_dir = checkpoint_dir / "params"
    assets_dir = checkpoint_dir / "assets"
    if not params_dir.is_dir():
        raise FileNotFoundError(f"Checkpoint params directory does not exist: {params_dir}")
    if not assets_dir.is_dir():
        raise FileNotFoundError(f"Checkpoint assets directory does not exist: {assets_dir}")
    if batch_size <= 0 or num_batches <= 0:
        raise ValueError("batch_size and num_batches must both be positive")

    config = _config.get_config(config_name)
    data_factory = dataclasses.replace(
        config.data,
        assets=dataclasses.replace(config.data.assets, assets_dir=str(assets_dir)),
    )
    config = dataclasses.replace(
        config,
        data=data_factory,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    logging.info("Loading EMA parameters from %s", params_dir)
    params = _model.restore_params(params_dir, dtype=jnp.bfloat16)
    model = config.model.load(params)
    model.eval()
    compute_loss = nnx_utils.module_jit(model.compute_loss, static_argnames=("train",))

    loader = _data_loader.create_data_loader(
        config,
        shuffle=shuffle,
        num_batches=num_batches,
    )

    total = 0.0
    total_squared = 0.0
    count = 0
    started = time.monotonic()
    base_rng = jax.random.key(seed)
    progress = tqdm.tqdm(enumerate(loader), total=num_batches, desc="Evaluating loss")
    for batch_index, (observation, actions) in progress:
        rng = jax.random.fold_in(base_rng, batch_index)
        losses = np.asarray(compute_loss(rng, observation, actions, train=False))
        total += float(losses.sum(dtype=np.float64))
        total_squared += float(np.square(losses, dtype=np.float64).sum(dtype=np.float64))
        count += losses.size
        progress.set_postfix(loss=f"{total / count:.6f}")

    mean = total / count
    variance = max(total_squared / count - mean**2, 0.0)
    standard_error = np.sqrt(variance / count)
    elapsed = time.monotonic() - started
    print(f"checkpoint: {checkpoint_dir}")
    print(f"config: {config_name}")
    print("mode: eval (no image augmentation), EMA parameters")
    print(f"seed: {seed}, shuffle: {shuffle}")
    print(f"batches: {num_batches}, batch_size: {batch_size}, samples: {num_batches * batch_size}")
    print(f"loss: {mean:.8f}")
    print(f"element_standard_error: {standard_error:.8f}")
    print(f"elapsed_seconds: {elapsed:.1f}")


if __name__ == "__main__":
    tyro.cli(main)

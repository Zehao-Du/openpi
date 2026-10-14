"""Tests for the B1 grounding auxiliary head (train-time mask supervision)."""

from types import SimpleNamespace

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np

from openpi.models import model as model_lib
from openpi.models import pi0 as pi0_lib
import openpi.models.gemma as _gemma
from openpi.models.pi0 import GroundingAuxHead
from openpi.models.pi0 import _downsample_grounding
from openpi.models.pi0 import _grounding_mask_iou
from openpi.models.pi0 import _grounding_mask_losses
import openpi.models.pi0_config as _pi0_config
from openpi.policies import realman_pika_policy

FOUR_CAMERAS = ("base_0_rgb", "left_wrist_0_rgb", "base_0_rgb_recolor", "left_wrist_0_rgb_recolor")


def _make_obs(batch: int, cameras: tuple[str, ...], *, with_masks: bool) -> model_lib.Observation:
    rng = jax.random.key(0)
    images = {}
    masks = {}
    for key in cameras:
        rng, sub = jax.random.split(rng)
        images[key] = jax.random.uniform(sub, (batch, 224, 224, 3), minval=-1.0, maxval=1.0)
        masks[key] = jnp.ones((batch,), dtype=jnp.bool_)
    grounding = None
    if with_masks:
        grounding = {}
        for key in cameras:
            rng, sub = jax.random.split(rng)
            # Labels in {-1, 0, 1}; single channel = "active visual prompt referent".
            raw = jax.random.randint(sub, (batch, 224, 224, 1), minval=-1, maxval=2)
            grounding[key] = raw.astype(jnp.float32)
    return model_lib.Observation(
        images=images,
        image_masks=masks,
        state=jnp.zeros((batch, 32)),
        grounding_masks=grounding,
    )


def test_inputs_spec_includes_grounding_masks():
    obs, _ = _pi0_config.Pi0Config(use_grounding_aux=True, image_keys=FOUR_CAMERAS).inputs_spec(batch_size=3)
    assert obs.grounding_masks is not None
    assert tuple(obs.grounding_masks) == FOUR_CAMERAS
    for spec in obs.grounding_masks.values():
        assert spec.shape == (3, 224, 224, 1)

    obs, _ = _pi0_config.Pi0Config().inputs_spec()
    assert obs.grounding_masks is None


def test_observation_from_dict_roundtrip():
    obs = _make_obs(2, ("base_0_rgb",), with_masks=True)
    data = obs.to_dict()
    assert "grounding_masks" in data
    restored = model_lib.Observation.from_dict(dict(data))
    assert restored.grounding_masks is not None
    np.testing.assert_array_equal(
        np.asarray(restored.grounding_masks["base_0_rgb"]), np.asarray(obs.grounding_masks["base_0_rgb"])
    )
    # Without the key, the field stays None.
    data.pop("grounding_masks")
    assert model_lib.Observation.from_dict(dict(data)).grounding_masks is None


def test_preprocess_grounding_passthrough_and_aug_lockstep():
    obs = _make_obs(2, FOUR_CAMERAS, with_masks=True)

    processed = model_lib.preprocess_observation(None, obs, image_keys=FOUR_CAMERAS)
    assert tuple(processed.grounding_masks) == FOUR_CAMERAS
    for key in FOUR_CAMERAS:
        assert processed.grounding_masks[key].shape == (2, 224, 224, 1)
        np.testing.assert_array_equal(np.asarray(processed.grounding_masks[key]), np.asarray(obs.grounding_masks[key]))

    # With augmentation, masks keep shape and stay in the label range (crop/rotate only, no jitter).
    processed = model_lib.preprocess_observation(jax.random.key(7), obs, train=True, image_keys=FOUR_CAMERAS)
    for key in FOUR_CAMERAS:
        mask = np.asarray(processed.grounding_masks[key])
        assert mask.shape == (2, 224, 224, 1)
        assert mask.min() >= -1.0 - 1e-5
        assert mask.max() <= 1.0 + 1e-5

    # A constant foreground mask stays mostly foreground under the mild spatial aug.
    obs.images.update({k: jnp.zeros((2, 224, 224, 3)) for k in FOUR_CAMERAS})
    obs = obs.replace(grounding_masks=dict.fromkeys(FOUR_CAMERAS, jnp.ones((2, 224, 224, 1))))
    processed = model_lib.preprocess_observation(jax.random.key(7), obs, train=True, image_keys=FOUR_CAMERAS)
    assert float(processed.grounding_masks["base_0_rgb"].mean()) > 0.8

    # Grounding keys that are not model image keys are dropped.
    obs = _make_obs(1, ("base_0_rgb",), with_masks=True)
    obs = obs.replace(
        images={"base_0_rgb": obs.images["base_0_rgb"], "extra_cam": obs.images["base_0_rgb"]},
        image_masks={"base_0_rgb": obs.image_masks["base_0_rgb"], "extra_cam": obs.image_masks["base_0_rgb"]},
        grounding_masks={"base_0_rgb": obs.grounding_masks["base_0_rgb"], "extra_cam": jnp.ones((1, 224, 224, 1))},
    )
    processed = model_lib.preprocess_observation(None, obs, image_keys=("base_0_rgb",))
    assert tuple(processed.grounding_masks) == ("base_0_rgb",)


def test_grounding_aux_head_shape():
    head = GroundingAuxHead(1152, hidden_dim=256, rngs=nnx.Rngs(0))
    feats = jnp.ones((3, 256, 1152), dtype=jnp.float32)
    logits = head(feats)
    assert logits.shape == (3, 16, 16, 1)
    num_params = sum(x.size for x in jax.tree.leaves(nnx.state(head, nnx.Param)))
    assert num_params < 5_000_000  # design constraint: lightweight head


def test_downsample_grounding():
    gt = jnp.ones((1, 224, 224, 1))
    pooled = _downsample_grounding(gt, 16)
    assert pooled.shape == (1, 16, 16, 1)
    np.testing.assert_allclose(pooled, 1.0, atol=1e-6)

    # Checkerboard foreground / unlabeled averages out to ~0 per pooled cell (i.e. ignored).
    checker = jnp.where((jnp.arange(224)[None, :, None, None] + jnp.arange(224)[None, None, :, None]) % 2, 1.0, -1.0)
    gt = jnp.broadcast_to(checker, (1, 224, 224, 1))
    pooled = _downsample_grounding(gt, 16)
    np.testing.assert_allclose(pooled, 0.0, atol=1e-6)


def test_grounding_mask_losses():
    # Perfect prediction -> ~zero loss, IoU 1.
    gt = jnp.where(jax.random.uniform(jax.random.key(1), (2, 16, 16, 2)) > 0.5, 1.0, 0.0)
    logits = gt * 20.0 - 10.0  # +1 -> +10, 0 -> -10
    loss = _grounding_mask_losses(logits, gt)
    assert loss.shape == (2, 2)
    np.testing.assert_allclose(loss, 0.0, atol=0.1)
    np.testing.assert_allclose(_grounding_mask_iou(logits, gt), 1.0, atol=1e-6)

    # Fully unlabeled channel contributes exactly zero (no gradient signal needed).
    gt = jnp.where(jnp.arange(2)[None, None, None, :] == 0, 1.0, -1.0)  # ch0 labeled, ch1 not
    gt = jnp.broadcast_to(gt, (2, 16, 16, 2))
    loss = _grounding_mask_losses(jnp.zeros((2, 16, 16, 2)), gt)
    np.testing.assert_allclose(loss[:, 1], 0.0, atol=1e-7)
    assert float(loss[:, 0].mean()) > 0.0


def _grounding_stub():
    head = GroundingAuxHead(8, hidden_dim=8, rngs=nnx.Rngs(0))
    return SimpleNamespace(grounding_aux_head=head, grounding_aux_lambda=0.05)


def test_embed_prefix_collects_grounding_feats():
    obs = _make_obs(1, ("base_0_rgb", "left_wrist_0_rgb"), with_masks=False)
    feats = jnp.ones((1, 256, 8))

    def encoder(image, *, train):
        return jnp.zeros((image.shape[0], 4, 16)), {"encoder": {"block20": {"+mlp": feats}}}

    stub = SimpleNamespace(PaliGemma=SimpleNamespace(img=encoder), grounding_aux_layer=20)
    _, _, _, collected = pi0_lib.Pi0.embed_prefix(stub, obs, collect_grounding_feats=True)
    assert set(collected) == {"base_0_rgb", "left_wrist_0_rgb"}
    np.testing.assert_array_equal(np.asarray(collected["base_0_rgb"]), np.asarray(feats))

    _, _, _, not_collected = pi0_lib.Pi0.embed_prefix(stub, obs)
    assert not_collected == {}


def test_grounding_aux_loss_metrics():
    stub = _grounding_stub()
    obs = _make_obs(2, ("base_0_rgb",), with_masks=True)
    feats = {"base_0_rgb": jax.random.normal(jax.random.key(3), (2, 256, 8))}
    total, info = pi0_lib.Pi0._grounding_aux_loss(stub, feats, obs)  # noqa: SLF001
    assert total.shape == (2,)
    assert np.isfinite(np.asarray(total)).all()
    assert set(info) == {
        "aux/vp_mask_loss",
        "aux/vp_mask_iou",
        "aux/grounding_loss",
        "aux/grounding_lambda",
    }
    # Cameras without supervision labels are skipped.
    total, info = pi0_lib.Pi0._grounding_aux_loss(stub, {}, obs)  # noqa: SLF001
    assert total is None
    assert info == {}


def test_decode_grounding_bitmask():
    bitmask = np.zeros((224, 224), dtype=np.uint8)
    bitmask[10:20, 10:20] = 0b01  # foreground (active prompt referent)
    bitmask[50:60, 50:60] = 0b10  # unlabeled
    decoded = realman_pika_policy._decode_grounding_bitmask(bitmask)  # noqa: SLF001
    assert decoded.shape == (224, 224, 1)
    assert decoded.dtype == np.float32
    assert decoded[15, 15, 0] == 1.0
    assert (decoded[55, 55] == -1.0).all()
    assert (decoded[0, 0] == 0.0).all()

    # CHW float image-feature layout (values in [0, 1]) decodes identically.
    chw = np.moveaxis(bitmask.astype(np.float32), -1, 0)[None] / 255.0
    np.testing.assert_array_equal(realman_pika_policy._decode_grounding_bitmask(chw), decoded)  # noqa: SLF001


def test_gemma_capture_prefix():
    """The scan-carry capture returns the tapped block's prefix hidden states without perturbing the forward."""
    cfg = _gemma.get_config("dummy")
    module = _gemma.Module(configs=[cfg, cfg], embed_dtype="float32")
    variables = module.init_with_output(jax.random.key(0), use_adarms=[False, False], method="init")[1]

    prefix = jax.random.normal(jax.random.key(1), (2, 6, cfg.width))
    suffix = jax.random.normal(jax.random.key(2), (2, 3, cfg.width))
    positions = jnp.broadcast_to(jnp.arange(9, dtype=jnp.int32)[None, :], (2, 9))
    mask = jnp.ones((2, 9, 9), dtype=bool)

    def run(capture_at):
        return module.apply(variables, [prefix, suffix], positions, mask, capture_prefix_at=capture_at)

    outs_plain, _, captured_none = run(None)
    assert captured_none is None

    outs_tapped, _, captured = run(1)
    assert captured.shape == prefix.shape
    # Capturing must not change the trunk outputs.
    for a, b in zip(jax.tree.leaves(outs_plain), jax.tree.leaves(outs_tapped), strict=True):
        np.testing.assert_allclose(a, b, rtol=0, atol=1e-6)
    # A mid-trunk snapshot differs from both the final (normed) prefix output and another layer.
    assert not np.allclose(np.asarray(captured), np.asarray(outs_tapped[0]))
    _, _, captured_first = run(0)
    assert not np.allclose(np.asarray(captured_first), np.asarray(captured))

    grads = jax.grad(lambda v: module.apply(v, [prefix, suffix], positions, mask, capture_prefix_at=1)[2].sum())(
        variables
    )
    assert any(float(jnp.abs(g).sum()) > 0.0 for g in jax.tree.leaves(grads["params"]))


def test_slice_llm_prefix_feats():
    obs = _make_obs(1, ("base_0_rgb", "left_wrist_0_rgb"), with_masks=False)
    captured = jnp.arange(1 * (512 + 7) * 4, dtype=jnp.float32).reshape(1, 512 + 7, 4)
    feats = pi0_lib.Pi0._slice_llm_prefix_feats(None, captured, obs)  # noqa: SLF001
    assert tuple(feats) == ("base_0_rgb", "left_wrist_0_rgb")
    assert feats["base_0_rgb"].shape == (1, 256, 4)
    np.testing.assert_array_equal(feats["left_wrist_0_rgb"], captured[:, 256:512])


def test_llm_prefix_tap_end_to_end():
    config = _pi0_config.Pi0Config(
        pi05=True,
        action_horizon=10,
        discrete_state_input=False,
        image_keys=FOUR_CAMERAS,
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        use_grounding_aux=True,
        grounding_aux_source="llm_prefix",
        grounding_aux_layer=2,
    )
    model = pi0_lib.Pi0(config, nnx.Rngs(0))

    # The head consumes trunk-width features (dummy width 64), not SigLIP's 1152.
    head_params = jax.tree.leaves(nnx.state(model.grounding_aux_head, nnx.Param))
    assert any(x.shape == (1, 1, 64, 256) for x in head_params)  # proj conv: trunk width in

    obs = _make_obs(2, FOUR_CAMERAS, with_masks=True)
    actions = jnp.ones((2, config.action_horizon, config.action_dim))
    rng = jax.random.key(7)

    loss_aux, info = model.compute_loss_with_aux(rng, obs, actions, train=True)
    # Same model/rng without labels: the pure-flow path (no tap, no aux). The gap must be exactly
    # lambda * aux, which also proves the trunk capture does not perturb the flow loss itself.
    loss_plain = model.compute_loss(rng, obs.replace(grounding_masks=None), actions, train=True)
    assert loss_aux.shape == (2, config.action_horizon)
    assert np.isfinite(np.asarray(loss_aux)).all()
    assert np.isfinite(float(info["aux/grounding_loss"]))
    # The capture must not perturb the flow loss: the gap is exactly lambda * aux.
    np.testing.assert_allclose(
        np.mean(np.asarray(loss_aux) - np.asarray(loss_plain)),
        0.05 * float(info["aux/grounding_loss"]),
        rtol=1e-4,
    )

    # The aux gradient reaches the Gemma trunk, not just the head.
    grads = nnx.grad(lambda m: jnp.mean(m.compute_loss_with_aux(rng, obs, actions, train=True)[0]))(model)
    trunk_leaves = [x for x in jax.tree.leaves(grads.PaliGemma.llm) if isinstance(x, jax.Array)]
    assert trunk_leaves
    assert any(float(jnp.abs(g).sum()) > 0.0 for g in trunk_leaves)

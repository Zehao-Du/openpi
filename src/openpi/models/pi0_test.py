from types import SimpleNamespace

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openpi.models import model as model_lib
from openpi.models.pi0 import Pi0
import openpi.models.pi0_config as _pi0_config


def _get_frozen_state(config: _pi0_config.Pi0Config) -> nnx.State:
    abstract_model = nnx.eval_shape(config.create, jax.random.key(0))

    freeze_filter = config.get_freeze_filter()
    return nnx.state(abstract_model, nnx.All(nnx.Param, freeze_filter)).flat_state()


def test_pi0_full_finetune():
    config = _pi0_config.Pi0Config()
    state = _get_frozen_state(config)
    assert len(state) == 0


def test_pi0_gemma_lora():
    config = _pi0_config.Pi0Config(paligemma_variant="gemma_2b_lora")
    state = _get_frozen_state(config)
    assert len(state) == 9
    assert all("lora" not in p for p in state)
    assert all("llm" in p for p in state)
    assert all("_1" not in p for p in state)


def test_pi0_action_expert_lora():
    config = _pi0_config.Pi0Config(action_expert_variant="gemma_300m_lora")
    state = _get_frozen_state(config)
    # excluding embedder, rest of the params should be same as gemma_lora.
    assert len(state) == 8
    assert all("lora" not in p for p in state)
    assert all("llm" in p for p in state)
    # all frozen params should have _1 in their path since it's the action expert.
    assert all(any("_1" in p for p in path) for path in state)


def test_pi0_all_lora():
    config = _pi0_config.Pi0Config(paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora")
    state = _get_frozen_state(config)
    # sum of gemma_lora and action_expert_lora's frozen params.
    assert len(state) == 17
    assert all("lora" not in p for p in state)
    assert all("llm" in p for p in state)


FOUR_CAMERAS = ("base_0_rgb", "left_wrist_0_rgb", "base_0_rgb_recolor", "left_wrist_0_rgb_recolor")


def test_four_camera_preprocess_and_shared_encoder_tokens() -> None:
    config = _pi0_config.Pi0Config(image_keys=FOUR_CAMERAS)
    obs = config.fake_obs()
    obs = model_lib.Observation(
        images={key: jnp.full((1, 224, 224, 3), i / 4) for i, key in enumerate(FOUR_CAMERAS)},
        image_masks=obs.image_masks,
        state=obs.state,
    )
    calls = []

    def encoder(image, *, train):
        assert not train
        calls.append(image)
        return jnp.mean(image, axis=(1, 2))[:, None, :], None

    processed = model_lib.preprocess_observation(None, obs, image_keys=config.image_keys)
    stub = SimpleNamespace(PaliGemma=SimpleNamespace(img=encoder))
    tokens, mask, _, grounding_feats = Pi0.embed_prefix(stub, processed)
    assert grounding_feats == {}
    assert len(calls) == 4
    assert tokens.shape == (1, 4, 3)
    np.testing.assert_allclose(tokens[0, 3], 0.75)
    assert mask.all()
    del obs.images[FOUR_CAMERAS[-1]]
    with pytest.raises(ValueError, match="missing keys"):
        model_lib.preprocess_observation(None, obs, image_keys=config.image_keys)


def test_paired_cameras_receive_identical_augmentation() -> None:
    config = _pi0_config.Pi0Config(image_keys=FOUR_CAMERAS)
    obs = config.fake_obs()
    pixels = jax.random.uniform(jax.random.key(1), (1, 224, 224, 3), minval=-1, maxval=1)
    obs.images.update(dict.fromkeys(FOUR_CAMERAS, pixels))
    result = model_lib.preprocess_observation(jax.random.key(2), obs, train=True, image_keys=FOUR_CAMERAS)
    for key in FOUR_CAMERAS[:2]:
        np.testing.assert_array_equal(result.images[key], result.images[f"{key}_recolor"])


def test_default_camera_spec_is_unchanged() -> None:
    obs, _ = _pi0_config.Pi0Config().inputs_spec()
    assert tuple(obs.images) == model_lib.IMAGE_KEYS

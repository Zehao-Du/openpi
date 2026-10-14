import dataclasses
from typing import TYPE_CHECKING, Literal

import flax.nnx as nnx
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
import openpi.models.gemma as _gemma
from openpi.shared import array_typing as at
import openpi.shared.nnx_utils as nnx_utils

if TYPE_CHECKING:
    from openpi.models.pi0 import Pi0


@dataclasses.dataclass(frozen=True)
class Pi0Config(_model.BaseModelConfig):
    dtype: str = "bfloat16"
    paligemma_variant: _gemma.Variant = "gemma_2b"
    action_expert_variant: _gemma.Variant = "gemma_300m"

    # Set the model specific defaults.
    action_dim: int = 32
    action_horizon: int = 50
    max_token_len: int = None  # type: ignore
    # Pi05 has two differences from Pi0:
    # - the state input is part of the discrete language tokens rather than a continuous input that is part of the suffix
    # - the action expert uses adaRMSNorm to inject the flow matching timestep
    pi05: bool = False
    image_keys: tuple[str, ...] = _model.IMAGE_KEYS
    # This config option is not used directly by the model, but it is read by the ModelTransformFactory.
    discrete_state_input: bool = None  # type: ignore

    # B1 grounding auxiliary head (train-time only, dropped at inference; see the B1 design doc in
    # the visual_prompt repo). The head decodes grounding masks from intermediate visual features
    # and its loss is added to the flow-matching loss with weight `grounding_aux_lambda`.
    use_grounding_aux: bool = False
    # Which features feed the head:
    #  - "siglip_block": SigLIP encoder block output (pure vision, gradient stays in the vision tower).
    #  - "llm_prefix": Gemma trunk prefix (image-token) hidden states -- language-conditioned via
    #    the fully bidirectional prefix block, so the head supervises the features the action
    #    expert actually reads; the aux gradient shapes the whole trunk up to that layer.
    grounding_aux_source: Literal["siglip_block", "llm_prefix"] = "siglip_block"
    # Source-dependent layer index (0-based): SigLIP encoder block (depth 27) for "siglip_block",
    # Gemma trunk block (depth 18) for "llm_prefix".
    grounding_aux_layer: int = 20
    # Weight of the auxiliary mask loss added to the flow loss.
    grounding_aux_lambda: float = 0.05
    # Hidden width of the lightweight auxiliary decoding head (<1M params).
    grounding_aux_hidden_dim: int = 256

    pytorch_compile_mode: str | None = "max-autotune"

    def __post_init__(self):
        if not self.image_keys or len(set(self.image_keys)) != len(self.image_keys):
            raise ValueError("image_keys must be nonempty and unique")
        if self.max_token_len is None:
            object.__setattr__(self, "max_token_len", 200 if self.pi05 else 48)
        if self.discrete_state_input is None:
            object.__setattr__(self, "discrete_state_input", self.pi05)
        if self.pytorch_compile_mode is not None:
            assert self.pytorch_compile_mode in [
                "default",
                "reduce-overhead",
                "max-autotune",
                "max-autotune-no-cudagraphs",
            ]

    @property
    @override
    def model_type(self) -> _model.ModelType:
        if self.pi05:
            return _model.ModelType.PI05
        return _model.ModelType.PI0

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0":
        from openpi.models.pi0 import Pi0

        return Pi0(self, rngs=nnx.Rngs(rng))

    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[_model.Observation, _model.Actions]:
        image_spec = jax.ShapeDtypeStruct([batch_size, *_model.IMAGE_RESOLUTION, 3], jnp.float32)
        image_mask_spec = jax.ShapeDtypeStruct([batch_size], jnp.bool_)

        with at.disable_typechecking():
            observation_spec = _model.Observation(
                images=dict.fromkeys(self.image_keys, image_spec),
                image_masks=dict.fromkeys(self.image_keys, image_mask_spec),
                state=jax.ShapeDtypeStruct([batch_size, self.action_dim], jnp.float32),
                tokenized_prompt=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32),
                tokenized_prompt_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], bool),
            )
            if self.use_grounding_aux:
                # Two supervision channels (target / reference object mask) per camera. Fake datasets
                # generate random labels from this spec, which is enough to smoke-test the aux path.
                grounding_spec = jax.ShapeDtypeStruct([batch_size, *_model.IMAGE_RESOLUTION, 2], jnp.float32)
                observation_spec = dataclasses.replace(
                    observation_spec,
                    grounding_masks=dict.fromkeys(self.image_keys, grounding_spec),
                )
        action_spec = jax.ShapeDtypeStruct([batch_size, self.action_horizon, self.action_dim], jnp.float32)

        return observation_spec, action_spec

    def get_freeze_filter(self) -> nnx.filterlib.Filter:
        """Returns the freeze filter based on the model config."""
        filters = []
        has_lora = False
        gemma_params_filter = nnx_utils.PathRegex(".*llm.*")
        action_expert_params_filter = nnx_utils.PathRegex(".*llm.*_1.*")
        if "lora" in self.paligemma_variant:
            filters.append(
                gemma_params_filter,
            )
            if "lora" not in self.action_expert_variant:
                # If only freeze gemma params, exclude action expert params.
                filters.append(
                    nnx.Not(action_expert_params_filter),
                )
            has_lora = True
        elif "lora" in self.action_expert_variant:
            filters.append(
                action_expert_params_filter,
            )
            has_lora = True

        if has_lora:
            # If any lora is used, exclude all lora params.
            filters.append(
                nnx.Not(nnx_utils.PathRegex(".*lora.*")),
            )
        if not filters:
            return nnx.Nothing
        return nnx.All(*filters)

import logging

import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
from openpi.models import pi0_config
import openpi.models.gemma as _gemma
import openpi.models.siglip as _siglip
from openpi.shared import array_typing as at

logger = logging.getLogger("openpi")


def make_attn_mask(input_mask, mask_ar):
    """Adapted from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` bool[?B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: bool[?B, N] mask that's true where previous tokens cannot depend on
        it and false where it shares the same attention mask as the previous token.
    """
    mask_ar = jnp.broadcast_to(mask_ar, input_mask.shape)
    cumsum = jnp.cumsum(mask_ar, axis=1)
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
    return jnp.logical_and(attn_mask, valid_mask)


class GroundingAuxHead(nnx.Module):
    """B1 lightweight decoding head: predicts grounding masks from SigLIP patch features.

    Exists only during training -- its loss shapes the visual features; inference never calls it
    (see the B1 design doc in the visual_prompt repo).
    """

    def __init__(self, input_dim: int, *, hidden_dim: int = 256, num_masks: int = 2, rngs: nnx.Rngs):
        self.proj = nnx.Conv(input_dim, hidden_dim, kernel_size=(1, 1), rngs=rngs)
        self.mask_conv = nnx.Conv(hidden_dim, hidden_dim // 2, kernel_size=(3, 3), padding="SAME", rngs=rngs)
        self.mask_head = nnx.Conv(hidden_dim // 2, num_masks, kernel_size=(1, 1), rngs=rngs)

    def __call__(self, feats: at.Float[at.Array, "b n c"]) -> at.Float[at.Array, "b h w m"]:
        side = round(feats.shape[1] ** 0.5)
        x = einops.rearrange(feats, "b (h w) c -> b h w c", h=side, w=side)
        x = nnx.gelu(self.proj(x))
        x = nnx.gelu(self.mask_conv(x))
        return self.mask_head(x)


# Supervision channel names, in order (matches the bitmask layout in the data pipeline).
GROUNDING_CHANNEL_NAMES = ("target", "reference")


def _downsample_grounding(gt: at.Float[at.Array, "b h w c"], grid: int) -> at.Float[at.Array, "b g g c"]:
    """Average-pools grounding labels to (grid, grid). Unlabeled (-1) and foreground (1) pool toward
    zero, so partially unlabeled edge cells stay (partially) ignored."""
    h, w = gt.shape[1:3]
    if h == grid and w == grid:
        return gt
    if h % grid == 0 and w % grid == 0:
        return einops.reduce(gt, "b (h p) (w q) c -> b h w c", p=h // grid, q=w // grid, reduction="mean")
    return jax.image.resize(gt, (gt.shape[0], grid, grid, gt.shape[-1]), method="area")


def _grounding_mask_losses(
    logits: at.Float[at.Array, "b h w c"], gt: at.Float[at.Array, "b h w c"]
) -> at.Float[at.Array, "b c"]:
    """Dice + BCE per sample and channel. Labels are in [-1, 1]; negative = unlabeled (excluded)."""
    valid = (gt >= 0.0).astype(jnp.float32)
    target = jnp.clip(gt, 0.0, 1.0)
    num = jnp.maximum(jnp.sum(valid, axis=(1, 2)), 1.0)

    # Numerically stable soft BCE: softplus(z) - y * z.
    bce = jax.nn.softplus(logits) - target * logits
    bce = jnp.sum(bce * valid, axis=(1, 2)) / num

    prob = jax.nn.sigmoid(logits)
    intersection = jnp.sum(prob * target * valid, axis=(1, 2))
    union = jnp.sum((prob + target) * valid, axis=(1, 2))
    dice = 1.0 - (2.0 * intersection + 1.0) / (union + 1.0)

    return bce + dice


def _grounding_mask_iou(
    logits: at.Float[at.Array, "b h w c"], gt: at.Float[at.Array, "b h w c"]
) -> at.Float[at.Array, " c"]:
    """Hard IoU between thresholded predictions and labels, averaged over the batch."""
    pred = (jax.nn.sigmoid(logits) > 0.5) & (gt >= 0.0)
    target = gt > 0.5
    intersection = jnp.sum(pred & target, axis=(0, 1, 2)).astype(jnp.float32)
    union = jnp.sum(pred | target, axis=(0, 1, 2)).astype(jnp.float32)
    return intersection / jnp.maximum(union, 1.0)


@at.typecheck
def posemb_sincos(
    pos: at.Real[at.Array, " b"], embedding_dim: int, min_period: float, max_period: float
) -> at.Float[at.Array, "b {embedding_dim}"]:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if embedding_dim % 2 != 0:
        raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by 2")

    fraction = jnp.linspace(0.0, 1.0, embedding_dim // 2)
    period = min_period * (max_period / min_period) ** fraction
    sinusoid_input = jnp.einsum(
        "i,j->ij",
        pos,
        1.0 / period * 2 * jnp.pi,
        precision=jax.lax.Precision.HIGHEST,
    )
    return jnp.concatenate([jnp.sin(sinusoid_input), jnp.cos(sinusoid_input)], axis=-1)


class Pi0(_model.BaseModel):
    def __init__(self, config: pi0_config.Pi0Config, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        self.pi05 = config.pi05
        self.image_keys = config.image_keys
        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)
        # TODO: rewrite gemma in NNX. For now, use bridge.
        llm = nnx_bridge.ToNNX(
            _gemma.Module(
                configs=[paligemma_config, action_expert_config],
                embed_dtype=config.dtype,
                adarms=config.pi05,
            )
        )
        llm.lazy_init(rngs=rngs, method="init", use_adarms=[False, True] if config.pi05 else [False, False])
        img = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=paligemma_config.width,
                variant="So400m/14",
                pool_type="none",
                scan=True,
                dtype_mm=config.dtype,
            )
        )
        img.lazy_init(next(iter(config.fake_obs().images.values())), train=False, rngs=rngs)
        self.PaliGemma = nnx.Dict(llm=llm, img=img)
        self.action_in_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
        if config.pi05:
            self.time_mlp_in = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        else:
            self.state_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_in = nnx.Linear(2 * action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        self.action_out_proj = nnx.Linear(action_expert_config.width, config.action_dim, rngs=rngs)

        # B1 grounding auxiliary head (train-time only; see the B1 design doc in the visual_prompt repo).
        self.grounding_aux_layer = config.grounding_aux_layer
        self.grounding_aux_lambda = config.grounding_aux_lambda
        self.grounding_aux_source = config.grounding_aux_source
        self.grounding_aux_head = None
        if config.use_grounding_aux:
            if config.grounding_aux_source == "llm_prefix":
                if not 0 <= config.grounding_aux_layer < paligemma_config.depth:
                    raise ValueError(
                        f"grounding_aux_layer={config.grounding_aux_layer} out of range for "
                        f"llm trunk depth {paligemma_config.depth}"
                    )
                # Taps Gemma trunk hidden states, which are language-conditioned via the fully
                # bidirectional prefix block; feature width equals the trunk width.
                input_dim = paligemma_config.width
            else:
                input_dim = _siglip.decode_variant("So400m/14")["width"]
            self.grounding_aux_head = GroundingAuxHead(input_dim, hidden_dim=config.grounding_aux_hidden_dim, rngs=rngs)

        # This attribute gets automatically set by model.train() and model.eval().
        self.deterministic = True

    @at.typecheck
    def embed_prefix(
        self, obs: _model.Observation, *, collect_grounding_feats: bool = False
    ) -> tuple[
        at.Float[at.Array, "b s emb"],
        at.Bool[at.Array, "b s"],
        at.Bool[at.Array, " s"],
        dict[str, at.Float[at.Array, "b n c"]],
    ]:
        input_mask = []
        ar_mask = []
        tokens = []
        grounding_feats: dict[str, at.Float[at.Array, "b n c"]] = {}
        # embed images
        for name in obs.images:
            image_tokens, img_out = self.PaliGemma.img(obs.images[name], train=False)
            if collect_grounding_feats:
                # Raw SigLIP features (1152-d, before the 2048-d head projection) from a mid-late
                # block: semantic enough to know *what* the prompt refers to, spatially detailed
                # enough to know *where*.
                block = f"block{self.grounding_aux_layer:02d}"
                grounding_feats[name] = img_out["encoder"][block]["+mlp"]

            tokens.append(image_tokens)
            input_mask.append(
                einops.repeat(
                    obs.image_masks[name],
                    "b -> b s",
                    s=image_tokens.shape[1],
                )
            )
            # image tokens attend to each other
            ar_mask += [False] * image_tokens.shape[1]

        # add language (aka tokenized inputs)
        if obs.tokenized_prompt is not None:
            tokenized_inputs = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
            tokens.append(tokenized_inputs)
            input_mask.append(obs.tokenized_prompt_mask)
            # full attention between image and language inputs
            ar_mask += [False] * tokenized_inputs.shape[1]
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask, grounding_feats

    @at.typecheck
    def embed_suffix(
        self, obs: _model.Observation, noisy_actions: _model.Actions, timestep: at.Float[at.Array, " b"]
    ) -> tuple[
        at.Float[at.Array, "b s emb"],
        at.Bool[at.Array, "b s"],
        at.Bool[at.Array, " s"],
        at.Float[at.Array, "b emb"] | None,
    ]:
        input_mask = []
        ar_mask = []
        tokens = []
        if not self.pi05:
            # add a single state token
            state_token = self.state_proj(obs.state)[:, None, :]
            tokens.append(state_token)
            input_mask.append(jnp.ones((obs.state.shape[0], 1), dtype=jnp.bool_))
            # image/language inputs do not attend to state or actions
            ar_mask += [True]

        action_tokens = self.action_in_proj(noisy_actions)
        # embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = posemb_sincos(timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0)
        if self.pi05:
            # time MLP (for adaRMS)
            time_emb = self.time_mlp_in(time_emb)
            time_emb = nnx.swish(time_emb)
            time_emb = self.time_mlp_out(time_emb)
            time_emb = nnx.swish(time_emb)
            action_expert_tokens = action_tokens
            adarms_cond = time_emb
        else:
            # mix timestep + action information using an MLP (no adaRMS)
            time_tokens = einops.repeat(time_emb, "b emb -> b s emb", s=self.action_horizon)
            action_time_tokens = jnp.concatenate([action_tokens, time_tokens], axis=-1)
            action_time_tokens = self.action_time_mlp_in(action_time_tokens)
            action_time_tokens = nnx.swish(action_time_tokens)
            action_time_tokens = self.action_time_mlp_out(action_time_tokens)
            action_expert_tokens = action_time_tokens
            adarms_cond = None
        tokens.append(action_expert_tokens)
        input_mask.append(jnp.ones(action_expert_tokens.shape[:2], dtype=jnp.bool_))
        # image/language/state inputs do not attend to action tokens
        ar_mask += [True] + ([False] * (self.action_horizon - 1))
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask, adarms_cond

    @override
    def compute_loss(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ) -> at.Float[at.Array, "*b ah"]:
        loss, _ = self._compute_loss_with_grounding(rng, observation, actions, train=train)
        return loss

    def compute_loss_with_aux(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ) -> tuple[at.Float[at.Array, "*b ah"], dict[str, at.Array]]:
        """Same as `compute_loss`, but also returns grounding aux metrics for logging."""
        return self._compute_loss_with_grounding(rng, observation, actions, train=train)

    def _compute_loss_with_grounding(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool
    ) -> tuple[at.Float[at.Array, "*b ah"], dict[str, at.Array]]:
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        observation = _model.preprocess_observation(
            preprocess_rng, observation, train=train, image_keys=self.image_keys
        )

        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded = time[..., None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        # one big forward pass of prefix + suffix at once
        use_grounding_aux = train and self.grounding_aux_head is not None and observation.grounding_masks is not None
        tap_llm = use_grounding_aux and self.grounding_aux_source == "llm_prefix"
        prefix_tokens, prefix_mask, prefix_ar_mask, grounding_feats = self.embed_prefix(
            observation, collect_grounding_feats=use_grounding_aux and not tap_llm
        )
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, time)
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        attn_mask = make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1
        (prefix_out, suffix_out), _, captured_prefix = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens],
            mask=attn_mask,
            positions=positions,
            adarms_cond=[None, adarms_cond],
            capture_prefix_at=self.grounding_aux_layer if tap_llm else None,
        )
        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

        flow_loss = jnp.mean(jnp.square(v_t - u_t), axis=-1)
        if not use_grounding_aux:
            return flow_loss, {}

        if tap_llm:
            grounding_feats = self._slice_llm_prefix_feats(captured_prefix, observation)
        aux_loss, aux_info = self._grounding_aux_loss(grounding_feats, observation)
        if aux_loss is None:
            return flow_loss, {}
        # Broadcast the per-sample aux loss over the action horizon so that the mean taken by the
        # training loop yields `mean(flow) + lambda * mean(aux)`.
        return flow_loss + (self.grounding_aux_lambda * aux_loss)[..., None], aux_info

    def _slice_llm_prefix_feats(
        self, captured: at.Float[at.Array, "b s d"], observation: _model.Observation
    ) -> dict[str, at.Float[at.Array, "b n d"]]:
        """Slice captured prefix hidden states into per-camera image-token spans.

        ``embed_prefix`` lays out the prefix as [cam0 256 | cam1 256 | ... | text], with each
        camera contributing the SigLIP patch grid ((224/14)**2 = 256) tokens in ``obs.images``
        order, so the spans are static.
        """
        tokens_per_image = (_model.IMAGE_RESOLUTION[0] // 14) ** 2  # SigLIP So400m/14 patch grid
        feats = {}
        offset = 0
        for name in observation.images:
            feats[name] = captured[:, offset : offset + tokens_per_image]
            offset += tokens_per_image
        assert offset <= captured.shape[1], "prefix shorter than the image token spans"
        return feats

    def _grounding_aux_loss(
        self, feats: dict[str, at.Float[at.Array, "b n c"]], observation: _model.Observation
    ) -> tuple[at.Float[at.Array, " b"] | None, dict[str, at.Array]]:
        """Per-sample grounding aux loss (sum over supervised cameras/channels) + log metrics."""
        total = None
        info: dict[str, at.Array] = {}
        for name, gt in observation.grounding_masks.items():
            if name not in feats:
                continue
            logits = self.grounding_aux_head(feats[name].astype(jnp.float32))[..., : gt.shape[-1]]
            gt_small = _downsample_grounding(gt, logits.shape[1])
            per_channel = _grounding_mask_losses(logits, gt_small)  # [b, c]
            iou = _grounding_mask_iou(logits, gt_small)  # [c]
            for c in range(per_channel.shape[-1]):
                key = GROUNDING_CHANNEL_NAMES[c] if c < len(GROUNDING_CHANNEL_NAMES) else f"mask{c}"
                info[f"aux/{key}_mask_loss"] = jnp.mean(per_channel[:, c])
                info[f"aux/{key}_mask_iou"] = iou[c]
            sample_loss = jnp.sum(per_channel, axis=-1)
            total = sample_loss if total is None else total + sample_loss
        if total is None:
            return None, {}
        info["aux/grounding_loss"] = jnp.mean(total)
        info["aux/grounding_lambda"] = jnp.asarray(self.grounding_aux_lambda, jnp.float32)
        return total, info

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
    ) -> _model.Actions:
        observation = _model.preprocess_observation(None, observation, train=False, image_keys=self.image_keys)
        # note that we use the convention more common in diffusion literature, where t=1 is noise and t=0 is the target
        # distribution. yes, this is the opposite of the pi0 paper, and I'm sorry.
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        # first fill KV cache with a forward pass of the prefix
        prefix_tokens, prefix_mask, prefix_ar_mask, _ = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache, _ = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

        def step(carry):
            x_t, time = carry
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                observation, x_t, jnp.broadcast_to(time, batch_size)
            )
            # `suffix_attn_mask` is shape (b, suffix_len, suffix_len) indicating how the suffix tokens can attend to each
            # other
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            # `prefix_attn_mask` is shape (b, suffix_len, prefix_len) indicating how the suffix tokens can attend to the
            # prefix tokens
            prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            # `combined_mask` is shape (b, suffix_len, prefix_len + suffix_len) indicating how the suffix tokens (which
            # generate the queries) can attend to the full prefix + suffix sequence (which generates the keys and values)
            full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
            assert full_attn_mask.shape == (
                batch_size,
                suffix_tokens.shape[1],
                prefix_tokens.shape[1] + suffix_tokens.shape[1],
            )
            # `positions` is shape (b, suffix_len) indicating the positions of the suffix tokens
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            (prefix_out, suffix_out), _, _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
            )
            assert prefix_out is None
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

            return x_t + dt * v_t, time + dt

        def cond(carry):
            x_t, time = carry
            # robust to floating-point error
            return time >= -dt / 2

        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        return x_0

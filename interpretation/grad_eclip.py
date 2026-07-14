"""Differentiable Grad-ECLIP for Hugging Face CLIP vision transformers.

The attribution follows Zhao et al., "Gradient-based Visual Explanation for
CLIP" (ICML 2024): the class-token gradient at the final self-attention output
weights patch value vectors, while class-query/patch-key similarity supplies a
spatial weight.  Unlike the inference-only reference implementation, this
version keeps the higher-order graph needed by explanation-guided training.

Reference implementation: https://github.com/Cyang-Zhao/Grad-Eclip
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F


def _unwrap_model(model):
    """Return the underlying CLIPModel when ``model`` is wrapped by DDP."""
    return model.module if hasattr(model, "module") else model


def _last_vision_attention(clip_model):
    vision_backbone = clip_model.vision_model
    # Transformers releases that use CLIPVisionModel add one wrapper level;
    # older releases expose CLIPVisionTransformer directly.
    if hasattr(vision_backbone, "vision_model"):
        vision_backbone = vision_backbone.vision_model
    try:
        return vision_backbone.encoder.layers[-1].self_attn
    except (AttributeError, IndexError) as exc:
        raise TypeError(
            "Grad-ECLIP expects a Hugging Face CLIPModel with a ViT vision "
            "encoder ending in a self-attention layer."
        ) from exc


def _reshape_map(
    patch_scores: torch.Tensor,
    grid_hw: Optional[Tuple[int, int]],
) -> torch.Tensor:
    num_patches = patch_scores.shape[1]
    if grid_hw is None:
        side = math.isqrt(num_patches)
        if side * side != num_patches:
            raise ValueError(
                f"Cannot infer a square patch grid from {num_patches} tokens; "
                "pass grid_hw explicitly."
            )
        grid_hw = (side, side)

    height, width = grid_hw
    if height * width != num_patches:
        raise ValueError(
            f"grid_hw={grid_hw} contains {height * width} positions, but the "
            f"vision encoder produced {num_patches} patch tokens."
        )
    return patch_scores.reshape(patch_scores.shape[0], 1, height, width)


def grad_eclip_vit(
    model,
    pixel_values: torch.Tensor,
    text_features: torch.Tensor,
    labels: torch.Tensor,
    grid_hw: Optional[Tuple[int, int]] = None,
    normalize: bool = True,
):
    """Return differentiable Grad-ECLIP maps and the model's true logits.

    Args:
        model: A Hugging Face ``CLIPModel``, optionally wrapped by DDP.
        pixel_values: Normalized images with shape ``[B, 3, H, W]``.
        text_features: L2-normalized class embeddings with shape ``[D, C]``.
        labels: Target class indices with shape ``[B]``.
        grid_hw: Patch-grid shape. If omitted, a square grid is inferred.
        normalize: Divide each non-negative map by its maximum. This preserves
            the scale expected by the existing MEGL/XIL alignment losses.

    The target differentiated for Grad-ECLIP is the selected image-text cosine
    similarity, matching the reference implementation. Classification logits
    additionally include CLIP's learned logit scale.
    """
    clip_model = _unwrap_model(model)
    attention = _last_vision_attention(clip_model)
    captured = {}

    def save_output(name):
        def hook(_module, _inputs, output):
            captured[name] = output

        return hook

    def save_attention_input(_module, inputs):
        captured["attention_output"] = inputs[0]

    handles = [
        attention.q_proj.register_forward_hook(save_output("query")),
        attention.k_proj.register_forward_hook(save_output("key")),
        attention.v_proj.register_forward_hook(save_output("value")),
        attention.out_proj.register_forward_pre_hook(save_attention_input),
    ]
    try:
        vision_output = clip_model.vision_model(
            pixel_values=pixel_values,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )
    finally:
        for handle in handles:
            handle.remove()

    missing = {"query", "key", "value", "attention_output"} - captured.keys()
    if missing:
        raise RuntimeError(
            "The CLIP attention forward pass did not expose the tensors needed "
            f"by Grad-ECLIP: {sorted(missing)}"
        )

    image_features = clip_model.visual_projection(vision_output.pooler_output)
    image_features = F.normalize(image_features, dim=-1)
    cosine_logits = image_features @ text_features
    logits = cosine_logits * clip_model.logit_scale.exp()
    target_score = cosine_logits.gather(1, labels.reshape(-1, 1)).sum()

    attention_output = captured["attention_output"]
    attention_gradient = torch.autograd.grad(
        outputs=target_score,
        inputs=attention_output,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    # The reference code applies the attention output projection to q and k
    # before measuring their cosine similarity.  It treats this spatial weight
    # as a fixed attribution coefficient, so keep that branch detached.
    with torch.no_grad():
        query = F.linear(
            captured["query"].detach(),
            attention.out_proj.weight.detach(),
            None
            if attention.out_proj.bias is None
            else attention.out_proj.bias.detach(),
        )
        key = F.linear(
            captured["key"].detach(),
            attention.out_proj.weight.detach(),
            None
            if attention.out_proj.bias is None
            else attention.out_proj.bias.detach(),
        )
        query_cls = F.normalize(query[:, :1, :], dim=-1)
        key_patch = F.normalize(key[:, 1:, :], dim=-1)
        spatial_weight = (query_cls * key_patch).sum(dim=-1)
        spatial_min = spatial_weight.amin(dim=1, keepdim=True)
        spatial_max = spatial_weight.amax(dim=1, keepdim=True)
        spatial_weight = (spatial_weight - spatial_min) / (
            spatial_max - spatial_min + 1e-6
        )

    gradient_cls = attention_gradient[:, :1, :]
    value_patch = captured["value"][:, 1:, :]
    patch_scores = torch.relu(
        (gradient_cls * value_patch * spatial_weight.unsqueeze(-1)).sum(dim=-1)
    )
    explanation = _reshape_map(patch_scores, grid_hw)

    if normalize:
        explanation = explanation / (
            explanation.amax(dim=(2, 3), keepdim=True) + 1e-6
        )
    return explanation, logits

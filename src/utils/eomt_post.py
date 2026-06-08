"""
src/utils/eomt_post.py

Central post-processing module for EoMT (and any MaskFormer-style model).
All anomaly scoring functions live here and are imported by runners/sweeps.

Functions:
    pixel_probs_from_masks      — MaskFormer-style class×mask composition -> per-pixel probs
    anomaly_from_pixel_probs    — MSP / MaxEntropy from pixel probs
    anomaly_maxlogit_from_masks — MaxLogit directly from raw logits (correct for mask models)
    rba_from_masks              — RbA: Region-based Anomaly scoring
"""

import torch
import torch.nn.functional as F


def pixel_probs_from_masks(
    mask_logits: torch.Tensor,   # [B, Q, H, W]
    class_logits: torch.Tensor,  # [B, Q, C(+1)]
    num_classes: int,
    temperature: float,
) -> torch.Tensor:
    """
    MaskFormer-style composition:
    1. Softmax with temperature on class logits (drop 'no-object' if present).
    2. Sigmoid on mask logits -> occupancy probabilities.
    3. Pixel-wise prob: sum_q [ class_prob[q,c] * mask_prob[q,h,w] ].
    4. Renormalize over classes to get a valid per-pixel distribution.

    Returns: [B, C, H, W]
    """
    if class_logits.shape[-1] == num_classes + 1:
        class_logits = class_logits[..., :num_classes]

    class_prob = F.softmax(class_logits / temperature, dim=-1)  # [B, Q, C]
    mask_prob  = torch.sigmoid(mask_logits)                     # [B, Q, H, W]

    pixel = torch.einsum("bqc,bqhw->bchw", class_prob, mask_prob)  # [B, C, H, W]

    den = pixel.sum(dim=1, keepdim=True).clamp_min(1e-8)
    return pixel / den


def anomaly_from_pixel_probs(
    pixel_probs: torch.Tensor,  # [B, C, H, W]
    method: str,
) -> torch.Tensor:
    """
    Anomaly score from normalized per-pixel class probabilities.
    Higher output = more anomalous.

    Supported methods:
        msp        — 1 - max(P)
        maxentropy — Shannon entropy: -sum(P * log P)

    Note: MaxLogit does NOT go through this function.
    Use anomaly_maxlogit_from_masks instead (works on raw logits, not probs).

    Returns: [B, H, W]
    """
    method = method.lower()

    if method == "msp":
        return 1.0 - pixel_probs.max(dim=1).values

    if method == "maxentropy":
        return -(pixel_probs * pixel_probs.clamp_min(1e-12).log()).sum(dim=1)

    raise ValueError(
        f"Unknown method for pixel_probs path: '{method}'. "
        "For maxlogit use anomaly_maxlogit_from_masks directly."
    )


def anomaly_maxlogit_from_masks(
    mask_logits: torch.Tensor,   # [B, Q, H, W]
    class_logits: torch.Tensor,  # [B, Q, C(+1)]
    num_classes: int,
) -> torch.Tensor:
    """
    MaxLogit anomaly score for mask-based models (EoMT / Mask2Former).

    Key difference from pixel-based MaxLogit:
    class logits are per-query, not per-pixel. We aggregate them into pixel
    space by weighting each query's class logits by its mask probability,
    BEFORE taking the max — preserving the raw logit scale (no softmax applied).

    Temperature has no effect on MaxLogit by definition (logits are pre-softmax).

    Steps:
        1. mask_prob    = sigmoid(mask_logits)                       [B, Q, H, W]
        2. pixel_logits = sum_q(mask_prob[q,h,w] * class_logits[q]) [B, C, H, W]
        3. anomaly      = 1 - max_c(pixel_logits)                   [B, H, W]

    Returns: [B, H, W]
    """
    if class_logits.shape[-1] == num_classes + 1:
        class_logits = class_logits[..., :num_classes]

    mask_prob    = torch.sigmoid(mask_logits)
    pixel_logits = torch.einsum("bqc,bqhw->bchw", class_logits, mask_prob)  # [B, C, H, W]

    return 1.0 - pixel_logits.max(dim=1).values


def rba_from_masks(
    mask_logits: torch.Tensor,   # [B, Q, H, W]
    class_logits: torch.Tensor,  # [B, Q, C(+1)]
    num_classes: int,
    temperature: float,
    area_pow: float = 0.5,       # non usato — mantenuto per compatibilità firma
) -> torch.Tensor:
    """
    RbA — Rejected by All (Nayal et al., ICCV 2023, arXiv 2211.14293).

    Formula esatta dal paper:
        L_k(x) = sum_q [ softmax(class_logits/T)[q,k] * sigmoid(mask_logits)[q,x] ]
        RbA(x) = -sum_k tanh(L_k(x))

    Higher output = more anomalous.

    Returns: [B, H, W]
    """
    if class_logits.shape[-1] == num_classes + 1:
        class_logits = class_logits[..., :num_classes]

    class_prob   = F.softmax(class_logits / temperature, dim=-1)  # [B, Q, C]
    mask_prob    = torch.sigmoid(mask_logits)                      # [B, Q, H, W]
    pixel_logits = torch.einsum("bqc,bqhw->bchw", class_prob, mask_prob)  # [B, C, H, W]

    return -torch.tanh(pixel_logits).sum(dim=1)                    # [B, H, W]

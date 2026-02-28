from typing import Optional

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
    1. Apply Softmax with Temperature scaling to class logits (drop 'no-object' if present).
    2. Apply Sigmoid to mask logits to get occupancy probabilities.
    3. Compute pixel-wise probability: sum over queries [q] of (class_prob[q,c] * mask_prob[q,h,w]).
    4. Normalize over class dimension [C] to ensure a valid per-pixel distribution.
    """
    # Drop 'no-object' class if the model outputs C+1 channels
    if class_logits.shape[-1] == num_classes + 1:
        class_logits = class_logits[..., :num_classes]

    class_prob = F.softmax(class_logits / temperature, dim=-1)  # [B, Q, C]
    mask_prob = torch.sigmoid(mask_logits)                      # [B, Q, H, W]

    # Weighted sum of masks using class probabilities as weights
    pixel = torch.einsum("bqc,bqhw->bchw", class_prob, mask_prob)  # [B, C, H, W]
    
    # Re-normalize across classes to handle overlapping masks
    den = pixel.sum(dim=1, keepdim=True).clamp_min(1e-8)
    pixel = pixel / den
    return pixel


def anomaly_from_pixel_probs(pixel_probs: torch.Tensor, method: str) -> torch.Tensor:
    """
    Computes an anomaly score from normalized pixel probabilities.
    Input: pixel_probs [B, C, H, W]
    Output: anomaly score [B, H, W] (higher values indicate Out-of-Distribution regions).
    """
    method = method.lower()

    if method == "msp":
        # Maximum Softmax Probability: Anomaly = 1 - max(P)
        msp = pixel_probs.max(dim=1).values
        return 1.0 - msp

    if method == "maxentropy":
        # Shannon Entropy: higher entropy suggests model uncertainty
        ent = -(pixel_probs * pixel_probs.clamp_min(1e-12).log()).sum(dim=1)
        return ent

    if method == "maxlogit":
        # MaxLogit proxy: negative maximum log-probability
        logp = pixel_probs.clamp_min(1e-12).log()
        m = logp.max(dim=1).values
        return -m

    raise ValueError(f"Unknown OOD detection method: {method}")


def rba_from_masks(
    mask_logits: torch.Tensor,   # [B, Q, H, W]
    class_logits: torch.Tensor,  # [B, Q, C(+1)]
    num_classes: int,
    temperature: float,
    area_pow: float = 0.5,
) -> torch.Tensor:
    """
    Region-Based Anomaly (RBA) implementation:
    1. Extract max confidence per query from class logits.
    2. Calculate mask area (mean occupancy) per query.
    3. Compute 'reliability' as a product of confidence and area (weighted by area_pow).
    4. Compute normality as the maximum reliability-weighted occupancy across all queries.
    5. Anomaly = 1 - normality.
    """
    if class_logits.shape[-1] == num_classes + 1:
        class_logits = class_logits[..., :num_classes]

    class_prob = F.softmax(class_logits / temperature, dim=-1)  # [B, Q, C]
    conf = class_prob.max(dim=-1).values                         # [B, Q]

    mask_prob = torch.sigmoid(mask_logits)                       # [B, Q, H, W]
    area = mask_prob.mean(dim=(-2, -1))                          # [B, Q]

    # Reliability scoring: rewards high confidence and substantial mask area
    reliability = conf * (area.clamp_min(1e-6) ** area_pow)      # [B, Q]
    reliability = reliability.unsqueeze(-1).unsqueeze(-1)        # [B, Q, 1, 1]

    # Normality is the max presence of a 'reliable' query at each pixel
    normality = (reliability * mask_prob).amax(dim=1)            # [B, H, W]
    return 1.0 - normality
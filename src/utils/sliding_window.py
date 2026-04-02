# src/utils/sliding_window.py
#
# Sliding window inference for EoMT — faithful port of the professor's
# window_imgs_semantic / revert_window_logits_semantic from
# eomt/training/lightning_module.py, extracted as a standalone utility
# with no Lightning / training dependencies.
#
# Pipeline (Option B — faithful to professor's approach):
#
#   1. scale_img_size()      — scale image preserving aspect ratio until
#                              the shorter side == img_size
#   2. window_image()        — tile the scaled image into overlapping
#                              square crops of shape (img_size, img_size)
#   3. [caller] run EoMT forward on each crop batch
#   4. to_pixel_logits()     — compose mask_logits + class_logits into
#                              per-pixel logits [C, h, w] per crop
#                              (same einsum as professor's code)
#   5. revert_window_logits()— accumulate crop pixel logits back onto
#                              the full scaled canvas via sum+count,
#                              then bilinear-resize to original resolution
#   6. [caller] compute anomaly score from final pixel logits [C, H, W]
#
# Usage in run_eomt_eval.py:
#
#   from src.utils.sliding_window import SlidingWindow
#   sw = SlidingWindow(img_size=1024, device=device)
#
#   crops, origins, orig_hw = sw.window_image(x)   # x: [1, C, H, W]
#   for crop_batch in sw.iter_batches(crops, batch_size=4):
#       ml, cl = model.forward_masks_and_classes(crop_batch)
#       pl = sw.to_pixel_logits(ml, cl, num_classes)
#       sw.accumulate(pl, origins, orig_hw)
#   pixel_logits = sw.finalize(orig_hw)             # [C, H, W]

import math
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


class SlidingWindow:
    """
    Stateful sliding window inference helper for EoMT.

    One instance per image — call reset() or create a new instance
    between images.
    """

    def __init__(self, img_size: int, device: torch.device):
        """
        Args:
            img_size : model's native input resolution (e.g. 640 or 1024).
                       Crops will be (img_size × img_size).
            device   : torch device for all intermediate tensors.
        """
        self.img_size = img_size
        self.device   = device

        # Accumulation buffers — set by window_image(), consumed by finalize()
        self._logit_sum:   torch.Tensor | None = None
        self._logit_count: torch.Tensor | None = None
        self._scaled_hw:   Tuple[int, int] | None = None

    # -----------------------------------------------------------------------
    # Step 1 — scale image preserving aspect ratio
    # -----------------------------------------------------------------------

    def scale_img_size(self, orig_hw: Tuple[int, int]) -> Tuple[int, int]:
        """
        Computes the scaled (H, W) such that the shorter side == img_size
        and the aspect ratio is preserved.

        Faithful port of professor's scale_img_size_semantic().
        """
        h, w   = orig_hw
        factor = max(self.img_size / h, self.img_size / w)
        return (round(h * factor), round(w * factor))

    # -----------------------------------------------------------------------
    # Step 2 — tile into overlapping crops
    # -----------------------------------------------------------------------

    def window_image(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, List[Tuple[int, int]], Tuple[int, int]]:
        """
        Scales and tiles a single image into square crops.

        Args:
            x : float tensor [1, 3, H, W] on any device, values in [0, 1]

        Returns:
            crops    : float tensor [N_crops, 3, img_size, img_size]
            origins  : list of (start, end) tuples indicating where each
                       crop was taken from along the long axis
            orig_hw  : original (H, W) before scaling — needed for revert
        """
        assert x.shape[0] == 1, "window_image processes one image at a time"

        img    = x[0]                                # [3, H, W]
        orig_h = img.shape[-2]
        orig_w = img.shape[-1]
        orig_hw = (orig_h, orig_w)

        # Scale preserving aspect ratio
        new_h, new_w = self.scale_img_size(orig_hw)

        # PIL resize (BILINEAR) — same as professor's code
        pil = Image.fromarray(
            (img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        )
        pil_resized = pil.resize((new_w, new_h), Image.BILINEAR)
        scaled = (
            torch.from_numpy(np.array(pil_resized)).permute(2, 0, 1).float() / 255.0
        ).to(self.device)  # [3, new_h, new_w]

        self._scaled_hw = (new_h, new_w)

        # Determine sliding direction (long axis)
        long_axis_size = max(new_h, new_w)
        is_portrait    = new_h > new_w   # slide vertically

        # Number of crops and overlap — faithful to professor
        num_crops        = math.ceil(long_axis_size / self.img_size)
        overlap          = num_crops * self.img_size - long_axis_size
        overlap_per_crop = (overlap / (num_crops - 1)) if num_crops > 1 else 0

        crops: List[torch.Tensor] = []
        origins: List[Tuple[int, int]] = []

        for j in range(num_crops):
            start = int(j * (self.img_size - overlap_per_crop))
            end   = start + self.img_size
            if is_portrait:
                crop = scaled[:, start:end, :]       # [3, img_size, new_w]
            else:
                crop = scaled[:, :, start:end]       # [3, new_h, img_size]
            crops.append(crop)
            origins.append((start, end))

        crops_tensor = torch.stack(crops)             # [N, 3, img_size, img_size]
        return crops_tensor, origins, orig_hw

    # -----------------------------------------------------------------------
    # Step 3 — batch iterator for memory-efficient forward passes
    # -----------------------------------------------------------------------

    def iter_batches(
        self,
        crops: torch.Tensor,
        batch_size: int = 1,
    ):
        """Yields sub-batches of crops for GPU-friendly forward passes."""
        for i in range(0, len(crops), batch_size):
            yield crops[i : i + batch_size]

    # -----------------------------------------------------------------------
    # Step 4 — compose mask_logits + class_logits into pixel logits
    # -----------------------------------------------------------------------

    @staticmethod
    def to_pixel_logits(
        mask_logits:  torch.Tensor,
        class_logits: torch.Tensor,
        num_classes:  int,
    ) -> torch.Tensor:
        """
        Converts EoMT mask/class outputs to per-pixel logits.

        Faithful port of professor's to_per_pixel_logits_semantic():
            pixel[c,h,w] = sum_q sigmoid(mask)[q,h,w] * softmax(class)[q,c]

        The no-object class (last column, index num_classes) is dropped
        before the einsum — same as professor's [... :-1] slice.

        Args:
            mask_logits  : [B, Q, h, w]  raw mask logits
            class_logits : [B, Q, C+1]   raw class logits (C classes + no-obj)
            num_classes  : number of semantic classes (C), excluding no-object

        Returns:
            pixel_logits : [B, C, h, w]  per-pixel class logits
        """
        # Drop no-object column — [B, Q, C]
        cl = class_logits[..., :num_classes]

        # Professor's einsum: sigmoid(mask) @ softmax(class)
        pixel_logits = torch.einsum(
            "bqhw, bqc -> bchw",
            mask_logits.sigmoid(),
            cl.softmax(dim=-1),
        )
        return pixel_logits   # [B, C, h, w]

    # -----------------------------------------------------------------------
    # Step 5 — accumulate crop pixel logits onto the scaled canvas
    # -----------------------------------------------------------------------

    def accumulate(
        self,
        pixel_logits: torch.Tensor,
        origins:      List[Tuple[int, int]],
        orig_hw:      Tuple[int, int],
        crop_indices: List[int],
    ) -> None:
        """
        Adds crop pixel logits into the accumulation buffers.

        Must be called for every crop (or batch of crops) in the same
        order as returned by window_image().

        Args:
            pixel_logits : [N, C, h, w]  pixel logits for this batch of crops
            origins      : full list of (start, end) from window_image()
            orig_hw      : original (H, W) — used to init buffers on first call
            crop_indices : which entries in origins correspond to this batch
        """
        C          = pixel_logits.shape[1]
        new_h, new_w = self._scaled_hw
        is_portrait  = new_h > new_w

        # Init buffers on first accumulate call
        if self._logit_sum is None:
            self._logit_sum   = torch.zeros(
                (C, new_h, new_w), device=self.device, dtype=torch.float32
            )
            self._logit_count = torch.zeros(
                (C, new_h, new_w), device=self.device, dtype=torch.float32
            )

        for k, idx in enumerate(crop_indices):
            start, end = origins[idx]
            # Upsample crop logits to img_size if needed
            crop_pl = F.interpolate(
                pixel_logits[k : k + 1],
                size=(self.img_size, self.img_size),
                mode="bilinear",
                align_corners=False,
            )[0]  # [C, img_size, img_size]

            if is_portrait:
                self._logit_sum[:, start:end, :]   += crop_pl
                self._logit_count[:, start:end, :] += 1
            else:
                self._logit_sum[:, :, start:end]   += crop_pl
                self._logit_count[:, :, start:end] += 1

    # -----------------------------------------------------------------------
    # Step 6 — finalize: average + resize to original resolution
    # -----------------------------------------------------------------------

    def finalize(self, orig_hw: Tuple[int, int]) -> torch.Tensor:
        """
        Computes the averaged pixel logits and resizes to original resolution.

        Returns:
            pixel_logits : [C, H, W]  at original image resolution
        """
        assert self._logit_sum is not None, \
            "No crops accumulated — call accumulate() before finalize()"

        averaged = self._logit_sum / self._logit_count.clamp(min=1e-6)

        # Bilinear resize to original resolution — same as professor's interpolate
        result = F.interpolate(
            averaged.unsqueeze(0),
            size=orig_hw,
            mode="bilinear",
            align_corners=False,
        )[0]  # [C, H, W]

        # Reset state for next image
        self._logit_sum   = None
        self._logit_count = None
        self._scaled_hw   = None

        return result

    def reset(self) -> None:
        """Manually reset accumulation buffers (called automatically by finalize)."""
        self._logit_sum   = None
        self._logit_count = None
        self._scaled_hw   = None

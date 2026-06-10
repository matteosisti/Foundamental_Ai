# src/models/eomt_wrapper.py
#
# Thin wrapper around the EoMT semantic-segmentation model.
#
# Two responsibilities:
#   1. Build the model — instantiate the encoder (a DINOv2 ViT from timm)
#      and the EoMT head, taking care of an import-path quirk in the
#      upstream EoMT package (see _alias_eomt_subpackages below).
#   2. Load checkpoints — provide two loading strategies:
#        * "robust"     : fuzzy matching, tolerant of prefix differences
#                         (Lightning / DataParallel) and shape mismatches.
#                         Optionally bicubically resizes the pos_embed when
#                         the checkpoint was trained at a different
#                         resolution than the one the model is built with.
#        * "prof-exact" : strict matching, raises on any architectural
#                         mismatch. Useful as a regression check.
#
# Forward inference exposes a single convenience method,
# forward_masks_and_classes(), returning the mask and class logits from
# the FINAL decoder layer only.

import re
import sys
import importlib
from typing import Dict, Tuple, Optional

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Sub-package aliasing
# ---------------------------------------------------------------------------

def _alias_eomt_subpackages():
    """
    Make the upstream EoMT package importable without modifying its source.

    The upstream EoMT codebase uses bare absolute imports like
        from models.xxx import ...
    which only resolve when its top-level directory is the current working
    directory. When EoMT is vendored inside a larger project as
    ``eomt/models``, those imports would fail.

    To avoid editing third-party source files, we register runtime aliases
    in ``sys.modules`` so that the bare module names resolve to the
    namespaced ones:

        models   -> eomt.models
        datasets -> eomt.datasets
        utils    -> eomt.utils

    Already-imported names are left untouched, and missing sub-packages are
    silently ignored so the wrapper degrades gracefully if some pieces of
    EoMT are not vendored.
    """
    aliases = {
        "models":   "eomt.models",
        "datasets": "eomt.datasets",
        "utils":    "eomt.utils",
    }

    for src_name, target_name in aliases.items():
        if src_name in sys.modules:
            continue
        try:
            mod = importlib.import_module(target_name)
            sys.modules[src_name] = mod
        except Exception:
            # Sub-package not present — fine, just skip the alias.
            pass


# ---------------------------------------------------------------------------
# State-dict utilities
# ---------------------------------------------------------------------------

def _unwrap_state_dict(raw: Dict) -> Dict[str, torch.Tensor]:
    """
    Return the actual tensor dictionary from a checkpoint payload.

    Lightning checkpoints wrap the tensors inside a top-level ``"state_dict"``
    entry; plain ``torch.save(model.state_dict(), ...)`` checkpoints do not.
    This helper handles both cases.
    """
    if "state_dict" in raw and isinstance(raw["state_dict"], dict):
        return raw["state_dict"]
    return raw


def _strip_prefixes(k: str, prefixes: Tuple[str, ...]) -> str:
    """
    Repeatedly strip any of the given prefixes from the start of ``k``.

    Iterates until a fixed point is reached so that nested prefixes such as
    ``"module.network.encoder..."`` are fully unwrapped.
    """
    changed = True
    k2 = k
    while changed:
        changed = False
        for p in prefixes:
            if k2.startswith(p):
                k2 = k2[len(p):]
                changed = True
    return k2


def _clean_state_dict_keys(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Normalize state-dict keys by removing the common wrapper prefixes:
    ``"network."``, ``"model."``, ``"module."``.

    These prefixes are added by Lightning modules, DataParallel, and custom
    wrappers. Removing them lets us match keys against the plain backbone.
    """
    prefixes = ("network.", "model.", "module.")
    clean = {}
    for k, v in state.items():
        k2 = _strip_prefixes(k, prefixes)
        clean[k2] = v
    return clean


def _interp_pos_embed_to_model(
    state: Dict[str, torch.Tensor],
    model: nn.Module,
) -> Dict[str, torch.Tensor]:
    """
    Bicubically resize the checkpoint's positional embeddings to match the
    model's grid.

    Motivation:
    -----------
    The EoMT positional embedding is a tensor of shape ``[1, G*G, C]``,
    where ``G`` is the patch grid size (= img_size / 16). When a checkpoint
    trained at one resolution is loaded into a model instantiated at a
    different resolution, the two ``pos_embed`` tensors have different
    sequence lengths and the fuzzy loader would silently drop the
    checkpoint's tensor — the model would then keep the random / pretrained
    pos_embed from timm, throwing away whatever was learned during
    fine-tuning.

    Example: a Cityscapes checkpoint trained at 1024 has ``pos_embed`` of
    shape ``[1, 4096, 768]`` (64x64 grid). Loaded into a 640-resolution
    model that expects ``[1, 1600, 768]`` (40x40 grid), it would be
    discarded. With ``interp_pos_embed=True`` the tensor is reshaped to
    ``[1, C, 64, 64]``, bicubically resized to ``[1, C, 40, 40]``, and
    reshaped back to ``[1, 1600, 768]`` before the fuzzy match.

    Only square grids are handled; non-square ones are left untouched and a
    note is printed.
    """
    import torch.nn.functional as F

    own = model.state_dict()
    out = dict(state)

    for k, v in state.items():
        if "pos_embed" not in k or not isinstance(v, torch.Tensor) or v.dim() != 3:
            continue
        if k not in own:
            continue
        tgt = own[k]
        if tgt.shape == v.shape:
            continue  # No mismatch, nothing to do.

        # Both grids must be square: shape [1, g*g, C].
        g_src = round(v.shape[1] ** 0.5)
        g_tgt = round(tgt.shape[1] ** 0.5)
        if g_src * g_src != v.shape[1] or g_tgt * g_tgt != tgt.shape[1]:
            print(f"[EoMT][interp] {k}: non-square grid "
                  f"(src={v.shape[1]}, tgt={tgt.shape[1]}) — skipped")
            continue

        C = v.shape[2]
        # Reshape [1, g*g, C] -> [1, C, g, g], resize, then reshape back.
        pe = v.reshape(1, g_src, g_src, C).permute(0, 3, 1, 2)
        pe = F.interpolate(pe, size=(g_tgt, g_tgt),
                           mode="bicubic", align_corners=False)
        out[k] = pe.permute(0, 2, 3, 1).reshape(1, g_tgt * g_tgt, C)
        print(f"[EoMT][interp] {k}: {g_src}x{g_src} -> {g_tgt}x{g_tgt} (bicubic)")

    return out


# ---------------------------------------------------------------------------
# Weight loading strategies
# ---------------------------------------------------------------------------

def _load_weights_robust(model: nn.Module, ckpt_path: str, device: torch.device,
                         interp_pos_embed: bool = False) -> Tuple[int, int]:
    """
    Fuzzy weight loader.

    Behaviour:
      * Accepts any wrapper prefix on the checkpoint keys (Lightning,
        DataParallel, etc.) by stripping them via ``_clean_state_dict_keys``.
      * Loads only the parameters whose shapes match the model exactly.
        Mismatched shapes are silently skipped — this is the price for not
        failing on small architectural differences.
      * If no key matches at all after prefix stripping, retries once after
        removing an optional ``"eomt."`` namespace prefix.
      * Optionally bicubically resizes the ``pos_embed`` to match the
        model's grid before the shape filter (see
        ``_interp_pos_embed_to_model``).

    Returns:
        (num_missing, num_unexpected) as reported by ``load_state_dict``.
    """
    raw = torch.load(ckpt_path, map_location="cpu")
    state = _clean_state_dict_keys(_unwrap_state_dict(raw))

    # Optional pos_embed resize. Must happen BEFORE the shape match so the
    # tensor passes the shape filter below.
    if interp_pos_embed:
        state = _interp_pos_embed_to_model(state, model)

    own = model.state_dict()
    loadable = {}

    for k, v in state.items():
        if k in own and own[k].shape == v.shape:
            loadable[k] = v

    # Fallback for checkpoints saved with an extra "eomt." namespace prefix
    # that survives the standard stripping above.
    if len(loadable) == 0:
        for k, v in state.items():
            k2 = re.sub(r"^eomt\.", "", k)
            if k2 in own and own[k2].shape == v.shape:
                loadable[k2] = v

    missing, unexpected = model.load_state_dict(loadable, strict=False)
    model.to(device)
    model.eval()

    return len(missing), len(unexpected)


def _load_weights_prof_exact(model: nn.Module, ckpt_path: str, device: torch.device) -> None:
    """
    Strict weight loader.

    Loads every key from the (prefix-cleaned) state dict and raises on any
    missing, unexpected, or shape-mismatched parameter. Use this as a
    regression check when you want to confirm that the checkpoint matches
    the model architecturally with no fuzz.
    """
    raw = torch.load(ckpt_path, map_location="cpu")
    state = _clean_state_dict_keys(_unwrap_state_dict(raw))

    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()


# ---------------------------------------------------------------------------
# Wrapper module
# ---------------------------------------------------------------------------

class EoMTWrapper(nn.Module):
    """
    A small ``nn.Module`` wrapping the EoMT semantic-segmentation model.

    Responsibilities:
      * Build the encoder (DINOv2 ViT from timm) and the EoMT head with the
        requested hyperparameters.
      * Apply the sub-package import shim before instantiating the model so
        that the upstream EoMT source files import cleanly.
      * Expose a ``load(...)`` method that chooses between the robust and
        the strict weight-loading strategies and, optionally, resizes the
        ``pos_embed``.
      * Provide a thin ``forward_masks_and_classes(...)`` that returns the
        FINAL-layer mask and class logits, which is all the downstream
        inference code needs.
    """

    def __init__(
        self,
        img_size:           Tuple[int, int],
        num_classes:        int  = 19,
        num_q:              int  = 100,
        num_blocks:         int  = 3,
        backbone_name:      str  = "vit_base_patch14_reg4_dinov2",
        masked_attn_enabled: bool = True,
    ):
        super().__init__()

        # Resolve EoMT's bare absolute imports before instantiating any
        # class from it. Calling this in __init__ keeps the side-effect
        # local to the moment we actually need it.
        _alias_eomt_subpackages()

        from eomt.models.vit import ViT
        from eomt.models.eomt import EoMT

        encoder = ViT(img_size=img_size, backbone_name=backbone_name)
        self.net = EoMT(
            encoder=encoder,
            num_classes=num_classes,
            num_q=num_q,
            num_blocks=num_blocks,
            masked_attn_enabled=masked_attn_enabled,
        )

        # Preserve construction hyperparameters as instance attributes for
        # downstream introspection (e.g. by the runners' logging code).
        self.img_size            = img_size
        self.num_classes         = num_classes
        self.num_q               = num_q
        self.num_blocks          = num_blocks
        self.backbone_name       = backbone_name
        self.masked_attn_enabled = masked_attn_enabled

    def load(self, ckpt_path: str, device: torch.device, mode: str = "robust",
             interp_pos_embed: bool = False) -> None:
        """
        Load weights from a checkpoint file.

        Args:
            ckpt_path:        Path to the ``.bin`` / ``.ckpt`` file.
            device:           Destination device (``cpu`` or ``cuda``).
            mode:             ``"robust"`` (default) or ``"prof-exact"``.
                              See ``_load_weights_robust`` and
                              ``_load_weights_prof_exact`` for behaviour.
            interp_pos_embed: When ``True`` and ``mode == "robust"``, the
                              checkpoint's ``pos_embed`` is bicubically
                              resized to the model's grid instead of being
                              dropped by the shape filter.
        """
        mode = mode.lower()
        if mode == "prof-exact":
            _load_weights_prof_exact(self.net, ckpt_path, device)
            print(f"[EoMT][prof-exact] Loaded STRICT weights from: {ckpt_path}")
        else:
            miss, unexp = _load_weights_robust(
                self.net, ckpt_path, device,
                interp_pos_embed=interp_pos_embed,
            )
            tag = "robust+interp" if interp_pos_embed else "robust"
            print(
                f"[EoMT][{tag}] Loaded fuzzy weights from: {ckpt_path} "
                f"| missing={miss} unexpected={unexp}"
            )

        # Both loading helpers call ``.eval()`` on ``self.net`` (the inner
        # EoMT module) but not on the wrapper itself. If any submodule with
        # BatchNorm or Dropout were ever instantiated directly on the
        # wrapper (not inside ``self.net``), it would remain in training
        # mode after load. Calling ``.eval()`` on the whole wrapper here
        # propagates the change to every submodule.
        self.to(device)
        self.eval()

    @torch.no_grad()
    def forward_masks_and_classes(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Inference helper returning the FINAL decoder layer's outputs.

        The underlying EoMT network returns lists of intermediate logits
        from each decoder block. For inference we only care about the last
        one, which is what the rest of the pipeline consumes.

        Args:
            x: input image batch of shape ``[B, 3, H, W]``.

        Returns:
            mask_logits:  ``[B, Q, H, W]`` per-query binary mask logits.
            class_logits: ``[B, Q, C+1]`` per-query class logits (``+1``
                          for the no-object label).
        """
        mask_list, class_list = self.net(x)
        mask_logits  = mask_list[-1]
        class_logits = class_list[-1]
        return mask_logits, class_logits
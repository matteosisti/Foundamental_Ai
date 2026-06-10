import re
import sys
import importlib
from typing import Dict, Tuple, Optional

import torch
import torch.nn as nn


def _alias_eomt_subpackages():
    """
    Fix broken absolute imports within the EoMT repository.
    Original course files often use:
        from models.xxx import ...
    instead of the correct:
        from eomt.models.xxx import ...

    To avoid patching original source files, we create runtime aliases:
        models   -> eomt.models
        datasets -> eomt.datasets
        utils    -> eomt.utils
    """
    aliases = {
        "models": "eomt.models",
        "datasets": "eomt.datasets",
        "utils": "eomt.utils",
    }

    for src_name, target_name in aliases.items():
        if src_name in sys.modules:
            continue
        try:
            mod = importlib.import_module(target_name)
            sys.modules[src_name] = mod
        except Exception:
            # Silently skip if subpackage is missing
            pass


def _unwrap_state_dict(raw: Dict) -> Dict[str, torch.Tensor]:
    """
    Extracts the state dictionary from various checkpoint formats.
    Handles both raw dictionaries and Lightning-style {state_dict: {...}} structures.
    """
    if "state_dict" in raw and isinstance(raw["state_dict"], dict):
        return raw["state_dict"]
    return raw


def _strip_prefixes(k: str, prefixes: Tuple[str, ...]) -> str:
    """Recursively removes specific prefixes from state_dict keys."""
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
    Normalizes common state_dict key prefixes resulting from 
    Lightning, DataParallel, or custom wrappers.
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
    BRANCH A-interp (da fossanetti, evalAnomalyStep8.ipynb):
    interpola bicubicamente il pos_embed del checkpoint alla griglia del modello,
    invece di scartarlo per shape mismatch.

    Esempio: checkpoint Cityscapes trainato a 1024x1024 → pos_embed [1, 4096, 768]
    (griglia 64x64). Modello istanziato a 640x640 → atteso [1, 1600, 768] (40x40).
    Senza interpolazione il load fuzzy scarta il pos_embed trainato e il modello
    usa quello DINOv2-pretrained di timm (non finetuned su Cityscapes).
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
            continue  # nessun mismatch, niente da fare

        # entrambe devono essere griglie quadrate [1, g*g, C]
        g_src = round(v.shape[1] ** 0.5)
        g_tgt = round(tgt.shape[1] ** 0.5)
        if g_src * g_src != v.shape[1] or g_tgt * g_tgt != tgt.shape[1]:
            print(f"[EoMT][interp] {k}: griglia non quadrata "
                  f"(src={v.shape[1]}, tgt={tgt.shape[1]}) — skip")
            continue

        C = v.shape[2]
        pe = v.reshape(1, g_src, g_src, C).permute(0, 3, 1, 2)          # [1,C,g,g]
        pe = F.interpolate(pe, size=(g_tgt, g_tgt),
                           mode="bicubic", align_corners=False)
        out[k] = pe.permute(0, 2, 3, 1).reshape(1, g_tgt * g_tgt, C)
        print(f"[EoMT][interp] {k}: {g_src}x{g_src} -> {g_tgt}x{g_tgt} (bicubic)")

    return out


def _load_weights_robust(model: nn.Module, ckpt_path: str, device: torch.device,
                         interp_pos_embed: bool = False) -> Tuple[int, int]:
    """
    Fuzzy weight loading:
    - Accepts checkpoints with mismatched keys.
    - Loads only parameters with compatible shapes.
    - Does not fail on missing or unexpected keys.
    Returns: (num_missing, num_unexpected)
    """
    raw = torch.load(ckpt_path, map_location="cpu")
    state = _clean_state_dict_keys(_unwrap_state_dict(raw))

    # BRANCH A-interp: porta il pos_embed del checkpoint alla griglia del modello
    # PRIMA del matching per shape, così non viene scartato.
    if interp_pos_embed:
        state = _interp_pos_embed_to_model(state, model)

    own = model.state_dict()
    loadable = {}

    for k, v in state.items():
        if k in own and own[k].shape == v.shape:
            loadable[k] = v

    # Fallback: attempt to remove optional "eomt." prefix if no matches found
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
    Strict weight loading:
    - Attempts to load all keys after prefix cleaning.
    - Raises clear errors on architecture mismatches (strict=True).
    """
    raw = torch.load(ckpt_path, map_location="cpu")
    state = _clean_state_dict_keys(_unwrap_state_dict(raw))

    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()


class EoMTWrapper(nn.Module):
    """
    A robust wrapper for the Everything on Mask Transformer (EoMT) model.
    Handles dynamic aliasing of subpackages and offers flexible weight loading modes.
    """
    def __init__(
        self,
        img_size: Tuple[int, int],
        num_classes: int = 19,
        num_q: int = 100,
        num_blocks: int = 3,
        backbone_name: str = "vit_base_patch14_reg4_dinov2",
        masked_attn_enabled: bool = True,
    ):
        super().__init__()

        # Fix internal EoMT absolute imports before model instantiation
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

        self.img_size = img_size
        self.num_classes = num_classes
        self.num_q = num_q
        self.num_blocks = num_blocks
        self.backbone_name = backbone_name
        self.masked_attn_enabled = masked_attn_enabled

    def load(self, ckpt_path: str, device: torch.device, mode: str = "robust",
             interp_pos_embed: bool = False) -> None:
        """
        Loads model weights.
        'mode' can be 'prof-exact' for strict loading or 'robust' for fuzzy matching.
        'interp_pos_embed=True' (BRANCH A-interp): interpola bicubicamente il
        pos_embed del checkpoint alla griglia del modello invece di scartarlo.
        Default False → comportamento storico invariato.
        """
        mode = mode.lower()
        if mode == "prof-exact":
            _load_weights_prof_exact(self.net, ckpt_path, device)
            print(f"[EoMT][prof-exact] Loaded STRICT weights from: {ckpt_path}")
        else:
            miss, unexp = _load_weights_robust(self.net, ckpt_path, device,
                                               interp_pos_embed=interp_pos_embed)
            tag = "robust+interp" if interp_pos_embed else "robust"
            print(f"[EoMT][{tag}] Loaded fuzzy weights from: {ckpt_path} | missing={miss} unexpected={unexp}")

        # PATCH — eval mode sul wrapper
        # _load_weights_robust / _load_weights_prof_exact chiamano .eval() solo su
        # self.net (il modulo EoMT interno), ma lasciano il wrapper EoMTWrapper
        # stesso in training=True. Qualsiasi submodulo con BatchNorm o Dropout
        # istanziato fuori da self.net rimarrebbe in training mode.
        # Questa chiamata mette in eval l'intero albero partendo dal wrapper.
        self.to(device)
        self.eval()

    @torch.no_grad()
    def forward_masks_and_classes(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Inference hook that returns logits from the FINAL decoder layer:
        - mask_logits: [B, Q, H, W]
        - class_logits: [B, Q, C(+1)]
        """
        mask_list, class_list = self.net(x)
        mask_logits = mask_list[-1]
        class_logits = class_list[-1]
        return mask_logits, class_logits
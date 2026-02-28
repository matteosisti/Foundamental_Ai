import re
import sys
import importlib
from pathlib import Path
from typing import Dict, Tuple, Any, Optional

import torch
import torch.nn as nn


def _alias_eomt_subpackages():
    """
    Fix broken absolute imports within the course EoMT codebase.
    Some files use 'from models.xxx import ...' instead of 'from eomt.models.xxx'.
    This creates runtime aliases to avoid modifying original source files:
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
            # Silently ignore if the target package doesn't exist
            pass


def _safe_read_yaml(path: str) -> Dict[str, Any]:
    """Reads a YAML configuration file safely using PyYAML."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {path}")

    try:
        import yaml
    except Exception as e:
        raise ImportError("Missing dependency: PyYAML. Install it with `pip install pyyaml`.") from e

    with open(p, "r", encoding="utf-8") as f:
        obj = yaml.safe_load(f)
    if not isinstance(obj, dict):
        raise ValueError(f"Invalid YAML structure in {path}")
    return obj


def _yaml_get(d: Dict[str, Any], keys: Tuple[str, ...], default=None):
    """Deep-get helper for nested dictionaries."""
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _clean_state_dict_keys(state: Dict[str, Any], allowed_prefixes=("network.", "model.", "module.")) -> Dict[str, torch.Tensor]:
    """
    Standardizes state_dict keys by stripping common prefixes (e.g., from Lightning or DataParallel).
    Iteratively removes prefixes like 'model.network.' to find the raw weight keys.
    """
    if "state_dict" in state and isinstance(state["state_dict"], dict):
        state = state["state_dict"]

    if not isinstance(state, dict):
        raise ValueError("Checkpoint format not recognized (expected dict or state_dict).")

    clean: Dict[str, torch.Tensor] = {}
    for k, v in state.items():
        if not torch.is_tensor(v):
            continue
        k2 = k
        changed = True
        while changed:
            changed = False
            for p in allowed_prefixes:
                if k2.startswith(p):
                    k2 = k2[len(p):]
                    changed = True
        clean[k2] = v
    return clean


def _load_weights_fuzzy(
    model: nn.Module,
    ckpt_path: str,
    device: torch.device,
    strict_load: bool = False,
    min_loaded_ratio: float = 0.70,
) -> None:
    """
    Loads weights from a checkpoint using shape-matching and key-normalization.
    If strict_load is enabled, it raises an error if the percentage of loaded keys 
    falls below min_loaded_ratio, preventing silent architecture mismatches.
    """
    raw = torch.load(ckpt_path, map_location="cpu")
    state = _clean_state_dict_keys(raw)

    own = model.state_dict()
    loadable: Dict[str, torch.Tensor] = {}

    for k, v in state.items():
        if k in own and own[k].shape == v.shape:
            loadable[k] = v

    # Fallback: attempt to strip optional leading "eomt." prefix
    if len(loadable) == 0:
        for k, v in state.items():
            k2 = re.sub(r"^eomt\.", "", k)
            if k2 in own and own[k2].shape == v.shape:
                loadable[k2] = v

    missing, unexpected = model.load_state_dict(loadable, strict=False)

    model.to(device)
    model.eval()

    loaded_ratio = len(loadable) / max(1, len(own))

    print(f"[EoMT] Loaded weights from: {ckpt_path}")
    print(f"[EoMT] Match ratio: {len(loadable)} / {len(own)} ({loaded_ratio*100:.1f}%)")
    
    if len(missing) > 0:
        print(f"[EoMT] Missing keys (first 20): {missing[:20]}")

    if strict_load and loaded_ratio < min_loaded_ratio:
        raise RuntimeError(
            f"[EoMT] Critical weight mismatch! Only {loaded_ratio*100:.1f}% weights loaded. "
            "Verify your YAML config and checkpoint compatibility."
        )


class EoMTWrapper(nn.Module):
    """
    Robust wrapper for the EoMT (Everything on Mask Transformer) model.
    Automatically parses architecture parameters from the professor's YAML config
    to ensure the model structure perfectly matches the checkpoint.
    """
    def __init__(
        self,
        config_path: str,
        img_size: Tuple[int, int],
        num_classes: int = 19,
        masked_attn_enabled: bool = True,
    ):
        super().__init__()

        # Fix internal EoMT absolute imports
        _alias_eomt_subpackages()
        
        # Load configuration to extract architectural hyperparameters
        cfg = _safe_read_yaml(config_path)

        num_q = _yaml_get(cfg, ("model", "init_args", "network", "init_args", "num_q"), default=100)
        num_blocks = _yaml_get(cfg, ("model", "init_args", "network", "init_args", "num_blocks"), default=3)
        backbone_name = _yaml_get(
            cfg,
            ("model", "init_args", "network", "init_args", "encoder", "init_args", "backbone_name"),
            default="vit_base_patch14_reg4_dinov2",
        )

        self.meta = {
            "config_path": str(config_path),
            "img_size": tuple(img_size),
            "num_classes": int(num_classes),
            "num_q": int(num_q),
            "num_blocks": int(num_blocks),
            "backbone_name": str(backbone_name),
            "masked_attn_enabled": bool(masked_attn_enabled),
        }

        # Imports must occur after aliasing
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

    def load(self, ckpt_path: str, device: torch.device, strict_load: bool = False) -> None:
        """Loads weights into the wrapped EoMT model."""
        _load_weights_fuzzy(self.net, ckpt_path, device, strict_load=strict_load)

    @torch.no_grad()
    def forward_masks_and_classes(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Runs inference and returns masks and class logits from the final decoder layer.
        Returns: (mask_logits [B, Q, H, W], class_logits [B, Q, C])
        """
        mask_list, class_list = self.net(x)
        return mask_list[-1], class_list[-1]
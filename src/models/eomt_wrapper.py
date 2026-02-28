
import re
import sys
import importlib
from typing import Dict, Tuple

import torch
import torch.nn as nn


def _alias_eomt_subpackages():
	"""
	EoMT code inside this course repo sometimes uses broken absolute imports:
		from models.xxx import ...
	instead of:
		from eomt.models.xxx import ...

	We avoid editing professor files by aliasing:
		models -> eomt.models
		datasets -> eomt.datasets (if needed)
		utils -> eomt.utils (if needed)
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
			# not all targets exist; ignore silently
			pass


def _clean_state_dict_keys(state: Dict[str, torch.Tensor], allowed_prefixes=("network.", "model.", "module.")) -> Dict[str, torch.Tensor]:
	# If it's a Lightning checkpoint, sometimes it's {"state_dict": {...}}
	if "state_dict" in state and isinstance(state["state_dict"], dict):
		state = state["state_dict"]

	# Strip known prefixes iteratively
	clean = {}
	for k, v in state.items():
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


def _load_weights_fuzzy(model: nn.Module, ckpt_path: str, device: torch.device) -> None:
	raw = torch.load(ckpt_path, map_location="cpu")
	state = _clean_state_dict_keys(raw)

	own = model.state_dict()
	loadable = {}

	for k, v in state.items():
		if k in own and own[k].shape == v.shape:
			loadable[k] = v

	if len(loadable) == 0:
		for k, v in state.items():
			k2 = re.sub(r"^eomt\.", "", k)
			if k2 in own and own[k2].shape == v.shape:
				loadable[k2] = v

	missing, unexpected = model.load_state_dict(loadable, strict=False)

	model.to(device)
	model.eval()

	print(f"[EoMT] Loaded weights from: {ckpt_path}")
	print(f"[EoMT] loadable keys: {len(loadable)} / model keys: {len(own)}")
	if len(missing) > 0:
		print(f"[EoMT] missing (up to 20): {missing[:20]}")
	if len(unexpected) > 0:
		print(f"[EoMT] unexpected (up to 20): {unexpected[:20]}")


class EoMTWrapper(nn.Module):
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

		#  fix broken absolute imports inside EoMT code
		_alias_eomt_subpackages()

		# Now imports should work even if EoMT used "from models..."
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

	def load(self, ckpt_path: str, device: torch.device) -> None:
		_load_weights_fuzzy(self.net, ckpt_path, device)

	@torch.no_grad()
	def forward_masks_and_classes(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
		mask_list, class_list = self.net(x)
		# We took last layer decoder outputs
		mask_logits = mask_list[-1]
		class_logits = class_list[-1]
		return mask_logits, class_logits

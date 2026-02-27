
import re
from typing import Dict, Any, Tuple, Optional

import torch
import torch.nn as nn


def _clean_state_dict_keys(state: Dict[str, torch.Tensor], allowed_prefixes=("network.", "model.", "module.")) -> Dict[str, torch.Tensor]:
	"""
	Try to normalize checkpoint keys to match the bare EoMT network keys.
	This supports common wrappers:
	- Lightning: state_dict with "network."
	- DataParallel: "module."
	- custom: "model."
	"""
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
	"""
	Load a checkpoint with best-effort key matching (strict=False),
	and also tries to handle common key-prefix differences.
	"""
	raw = torch.load(ckpt_path, map_location="cpu")
	state = _clean_state_dict_keys(raw)

	own = model.state_dict()
	loadable = {}

	for k, v in state.items():
		if k in own and own[k].shape == v.shape:
			loadable[k] = v

	# fallback: try removing "encoder." / "network." kinds of mismatches if needed
	if len(loadable) == 0:
		for k, v in state.items():
			k2 = re.sub(r"^eomt\.", "", k)
			if k2 in own and own[k2].shape == v.shape:
				loadable[k2] = v

	missing, unexpected = model.load_state_dict(loadable, strict=False)

	# Move to device after loading
	model.to(device)
	model.eval()

	# Helpful prints
	print(f"[EoMT] Loaded weights from: {ckpt_path}")
	print(f"[EoMT] loadable keys: {len(loadable)} / model keys: {len(own)}")
	if len(missing) > 0:
		print(f"[EoMT] missing (showing up to 20): {missing[:20]}")
	if len(unexpected) > 0:
		print(f"[EoMT] unexpected (showing up to 20): {unexpected[:20]}")


class EoMTWrapper(nn.Module):
	"""
	Thin wrapper around the professor EoMT code.
	Assumes eomt/ package is available in PYTHONPATH (repo root in sys.path).
	"""

	def __init__(
		self,
		num_classes: int = 19,
		num_q: int = 100,
		num_blocks: int = 3,
		backbone_name: str = "vit_base_patch14_reg4_dinov2",
		masked_attn_enabled: bool = True,
	):
		super().__init__()

		# Import from the provided EoMT package
		from eomt.models.vit import ViT
		from eomt.models.eomt import EoMT

		encoder = ViT(backbone_name=backbone_name)
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
		"""
		Returns last-layer (mask_logits, class_logits)
		mask_logits: [B,Q,H,W]
		class_logits: [B,Q,C(+1)]
		"""
		mask_list, class_list = self.net(x)
		mask_logits = mask_list[-1]
		class_logits = class_list[-1]
		return mask_logits, class_logits

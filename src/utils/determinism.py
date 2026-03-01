# src/utils/determinism.py
import os
import random
from typing import Optional

import numpy as np
import torch


def apply_determinism(
	mode: str,
	seed: int = 0,
	deterministic: bool = False,
) -> None:
	"""
	mode:
	- robust: default deterministic-friendly
	- prof-exact: default speed/compat (benchmark on) unless deterministic=True

	deterministic=True forces determinism even in prof-exact.
	"""

	# 1) Seeds
	os.environ["PYTHONHASHSEED"] = str(seed)
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(seed)

	# 2) TF32 off (avoid tiny numeric differences on Ampere+)
	# This helps matching logits between runs.
	try:
		torch.backends.cuda.matmul.allow_tf32 = False
		torch.backends.cudnn.allow_tf32 = False
	except Exception:
		pass

	# 3) cuDNN behavior
	mode = (mode or "robust").lower().strip()
	force_det = bool(deterministic)

	if mode == "prof-exact" and not force_det:
		# mimic typical professor scripts: fast but not deterministic
		torch.backends.cudnn.benchmark = True
		torch.backends.cudnn.deterministic = False
	else:
		# deterministic-friendly
		torch.backends.cudnn.benchmark = False
		torch.backends.cudnn.deterministic = True

		# 4) PyTorch deterministic algorithms (can throw if an op is not supported)
		# Keep warn_only=True to not crash in Colab if some op is nondet.
		try:
			torch.use_deterministic_algorithms(True, warn_only=True)
		except Exception:
			pass

		# 5) cuBLAS workspace config (needed for strict determinism in matmul)
		# Must be set BEFORE CUDA context sometimes; still ok to set here.
		os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
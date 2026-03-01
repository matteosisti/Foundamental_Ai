# src/utils/determinism.py
def set_determinism(seed: int = 0, deterministic: bool = True):
	import os
	import random
	import numpy as np
	import torch

	# python/numpy
	os.environ["PYTHONHASHSEED"] = str(seed)
	random.seed(seed)
	np.random.seed(seed)

	# torch
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(seed)

	if not deterministic:
		# allow fastest algos
		torch.backends.cudnn.deterministic = False
		torch.backends.cudnn.benchmark = True
		return

	# deterministic path
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True

	# PyTorch 2.x: make ops deterministic if possible
	try:
		torch.use_deterministic_algorithms(True)
	except Exception:
		pass

	# recommended for determinism on recent CUDA/cuBLAS
	os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
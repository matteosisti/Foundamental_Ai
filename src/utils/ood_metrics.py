import numpy as np

from sklearn.metrics import roc_curve


def fpr_at_95_tpr_sweep(scores: np.ndarray, labels: np.ndarray) -> float:
	"""
	FPR when TPR is at least 95% (pixel-wise), computed by explicit sweep.
	scores: higher => more OOD
	labels: 1=OOD, 0=InD
	"""
	scores = scores.astype(np.float64)
	labels = labels.astype(np.int64)

	order = np.argsort(-scores)
	scores = scores[order]
	labels = labels[order]

	P = int((labels == 1).sum())
	N = int((labels == 0).sum())
	if P == 0 or N == 0:
		return float("nan")

	tp = 0
	fp = 0
	tpr_target = 0.95

	for i in range(len(scores)):
		if labels[i] == 1:
			tp += 1
		else:
			fp += 1

		tpr = tp / P
		if tpr >= tpr_target:
			return float(fp / N)

	return 1.0


def fpr_at_95_tpr_roc(scores: np.ndarray, labels: np.ndarray) -> float:
	"""
	FPR@95TPR using ROC curve interpolation (sklearn).
	This is often what people mean by 'FPR@95TPR' in papers.
	"""
	scores = scores.astype(np.float64)
	labels = labels.astype(np.int64)

	P = int((labels == 1).sum())
	N = int((labels == 0).sum())
	if P == 0 or N == 0:
		return float("nan")

	fpr, tpr, _thr = roc_curve(labels, scores, pos_label=1)

	idx = np.where(tpr >= 0.95)[0]
	if len(idx) == 0:
		return 1.0
	return float(fpr[idx[0]])


def fpr_at_95_tpr(scores: np.ndarray, labels: np.ndarray, mode: str = "robust") -> float:
	"""
	mode:
	- robust: sweep exact
	- prof-exact: ROC-based
	"""
	mode = mode.lower()
	if mode == "prof-exact":
		return fpr_at_95_tpr_roc(scores, labels)
	return fpr_at_95_tpr_sweep(scores, labels)
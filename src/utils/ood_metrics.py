import numpy as np
from sklearn.metrics import roc_curve

def fpr_at_95_tpr_sweep(scores: np.ndarray, labels: np.ndarray) -> float:
    """
    Computes False Positive Rate at 95% True Positive Rate using a vectorized sweep.
    
    This implementation replaces slow Python loops with NumPy cumulative operations,
    providing exact discrete thresholding results at pixel-level scale.
    
    Complexity: O(N log N) due to sorting, where N is the number of pixels.
    Note: If many equal scores exist, this performs a sample-wise sweep 
    rather than a threshold-grouped sweep.
    
    Args:
        scores: Anomaly scores (higher values indicate Out-of-Distribution).
        labels: Ground truth binary masks (1 for OOD, 0 for In-Distribution).
        
    Returns:
        float: FPR value at the 95% TPR threshold.
    """
    # Downcast to float32/int8 to optimize memory footprint during sorting (O(N) space)
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    labels = np.asarray(labels, dtype=np.int8).reshape(-1)

    # Count total Positive (OOD) and Negative (InD) pixels - O(N)
    pos_count = int(np.sum(labels == 1))
    neg_count = int(np.sum(labels == 0))
    
    if pos_count == 0 or neg_count == 0:
        return float("nan")

    # 1. Sort scores in descending order to simulate descending threshold sweep - O(N log N)
    sort_indices = np.argsort(-scores)
    sorted_labels = labels[sort_indices]

    # 2. Vectorized cumulative sum of True Positives and False Positives - O(N)
    # tp_sum[i] represents TP count found up to the i-th highest score
    tp_sum = np.cumsum(sorted_labels == 1)
    fp_sum = np.cumsum(sorted_labels == 0)

    # 3. Locate the first index where TPR meets or exceeds 0.95
    # np.ceil ensures we cover AT LEAST 95% of positive samples (conservative approach)
    target_tp = int(np.ceil(0.95 * pos_count))
    
    # Binary search for the target TP count - O(log N)
    # side='left' picks the smallest index satisfying tp_sum[idx] >= target_tp
    idx_at_95 = np.searchsorted(tp_sum, target_tp, side='left')

    if idx_at_95 >= len(fp_sum):
        return 1.0
        
    # FPR = FP / Total Negatives
    return float(fp_sum[idx_at_95] / neg_count)


def fpr_at_95_tpr_roc(scores: np.ndarray, labels: np.ndarray) -> float:
    """
    Computes FPR@95TPR using Scikit-Learn's ROC curve implementation.
    
    Commonly used for benchmarking against academic papers. Note that interpolation 
    strategies in Scikit-Learn might result in slight differences compared to 
    discrete manual sweeps on pixel-wise data.
    """
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    labels = np.asarray(labels, dtype=np.int8).reshape(-1)

    if np.sum(labels == 1) == 0 or np.sum(labels == 0) == 0:
        return float("nan")

    # Generate the ROC curve points
    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)

    # Find the first point where TPR >= 0.95
    match_indices = np.where(tpr >= 0.95)[0]
    if len(match_indices) == 0:
        return 1.0
	
    return float(fpr[match_indices[0]])


def fpr_at_95_tpr(scores: np.ndarray, labels: np.ndarray, mode: str = "robust") -> float:
    """
    Universal wrapper for FPR@95TPR calculation.
    
    Supported Modes:
        - 'robust' / 'prof-exact' / 'sweep': Exact vectorized discrete sweep (Matches course scripts).
        - 'roc' / 'sklearn': Interpolated ROC curve from Scikit-Learn.
    """
    mode = (mode or "robust").lower().strip()

    if mode in ("roc", "roc-curve", "sklearn"):
        return fpr_at_95_tpr_roc(scores, labels)

    if mode in ("prof-exact", "prof", "exact", "robust", "sweep"):
        return fpr_at_95_tpr_sweep(scores, labels)

    raise ValueError(f"Unsupported metric mode: {mode}")
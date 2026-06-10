"""
src/utils/ood_dataset.py

Central utility for loading and remapping OOD ground-truth masks.
Used by all runners (ERFNet, EoMT) and sweep scripts.

Convention after remapping:
    0   = In-Distribution (InD)
    1   = Out-of-Distribution (OOD)
    255 = ignore / void
"""

import os
from typing import Tuple

import numpy as np
from PIL import Image
from torchvision.transforms import Resize


# ---------------------------------------------------------------------------
# GT path resolution
# ---------------------------------------------------------------------------

def gt_path_from_image(path_img: str) -> str:
    """
    Derives the GT mask path from an image path.
    Convention: swap 'images' -> 'labels_masks', force .png extension
    for all known datasets.
    """
    path_gt = path_img.replace("images", "labels_masks")
    root    = path_gt

    if "RoadObstacle21" in root or "RoadObsticle21" in root:
        return os.path.splitext(root)[0] + ".png"
    if "fs_static" in root:
        return os.path.splitext(root)[0] + ".png"
    if "RoadAnomaly21" in root or "RoadAnomaly" in root:
        return os.path.splitext(root)[0] + ".png"
    if "LostAndFound" in root or "FS_LostFound_full" in root:
        return os.path.splitext(root)[0] + ".png"

    # Fallback: keep existing extension or add .png
    ext = os.path.splitext(root)[1].lower()
    return root if ext else root + ".png"


# ---------------------------------------------------------------------------
# GT mask remapping
# ---------------------------------------------------------------------------

def _is_already_binary_ood_mask(uvals: np.ndarray) -> bool:
    """
    Returns True if the mask is already in the final binary convention
    {0=InD, 1=OOD, 255=ignore} — i.e. no further remapping needed.

    This guard is critical for LostAndFound: some exports are already
    binary while others use the legacy multi-class encoding.
    Applying the legacy remapping to an already-binary mask would corrupt it.
    """
    s = set(int(x) for x in uvals.tolist())
    return s.issubset({0, 1, 255}) and (0 in s or 1 in s)


# PATCH — remap_ood_mask: aggiunto branch esplicito per RoadObstacle21
# ---------------------------------------------------------------------
# BUG ORIGINALE: RoadObstacle21 non aveva nessun branch dedicato in
# remap_ood_mask. Il check "RoadAnomaly" in path_gt NON matcha il path
# di RO21 (che contiene "RoadObstacle21"), quindi le sue maschere
# venivano restituite grezze, senza alcuna rimappatura.
#
# Le maschere di RO21 usano un encoding multi-valore distinto da quello
# delle altre maschere già in formato {0,1}. Senza rimappatura corretta,
# i pixel anomali non vengono identificati come label=1, producendo
# score invertiti e AuPRC vicino al caso casuale (FPR ~100%).
#
# FIX: branch specifico per RoadObstacle21 inserito PRIMA del check
# generico "RoadAnomaly", con guard binario per gestire entrambe le
# versioni delle maschere (già binarie o multi-valore).
#
# ENCODING ATTESO per RO21 (multi-valore):
#   0   -> void/ignore  (rimappato a 255)
#   1   -> InD road     (rimappato a 0)
#   2   -> OOD obstacle (rimappato a 1)
# Se la maschera è già in {0,1,255} il guard la lascia invariata.
#
# IMPATTO STIMATO: +45÷+59 AuPRC su RO21 per tutti i metodi e checkpoint.

def remap_ood_mask(path_gt: str, ood: np.ndarray) -> np.ndarray:
    """
    Remaps dataset-specific GT encodings to the binary convention:
        0 = InD,  1 = OOD,  (255 = ignore/void)

    Dataset rules:
        RoadObstacle21     : 0->255 (void), 1->0 (InD), 2->1 (OOD)
                             (only if not already binary — guard applied)
        RoadAnomaly / RA21 : label 2 -> 1 (OOD)
        LostAndFound /
        FS_LostFound_full  : legacy multi-class -> binary
                             (only if not already binary — see guard below)
    """

    # --- RoadObstacle21 — branch dedicato, controllato prima di "RoadAnomaly" ---
    # Il check è più specifico: "RoadObstacle21" non contiene la stringa
    # "RoadAnomaly", quindi i due branch sono mutuamente esclusivi sul path.
    if "RoadObstacle21" in path_gt or "RoadObsticle21" in path_gt:
        if not _is_already_binary_ood_mask(np.unique(ood)):
            # Encoding multi-valore atteso: 0=void, 1=InD, 2=OOD
            ood = np.where(ood == 2, 1,   ood)   # OOD obstacle -> 1
            ood = np.where(ood == 1, 0,   ood)   # InD road     -> 0
            ood = np.where(ood == 0, 255, ood)   # void         -> ignore
            # Nota: l'ordine è importante — rimappare prima il 2, poi l'1,
            # poi lo 0, per evitare collisioni tra i valori intermedi.
        return ood

    # --- RoadAnomaly21 / RoadAnomaly ---
    if "RoadAnomaly" in path_gt:
        ood = np.where(ood == 2, 1, ood)

    # --- LostAndFound / FS_LostFound_full ---
    if "LostAndFound" in path_gt or "FS_LostFound_full" in path_gt:
        # Guard: skip remapping if mask is already in binary convention.
        # Without this check a double-remap would corrupt the labels.
        if not _is_already_binary_ood_mask(np.unique(ood)):
            ood = np.where(ood == 0, 255, ood)           # void  -> ignore
            ood = np.where(ood == 1, 0,   ood)           # road  -> InD
            ood = np.where((ood > 1) & (ood < 201), 1, ood)  # obstacles -> OOD

    return ood


# ---------------------------------------------------------------------------
# Public loader
# ---------------------------------------------------------------------------

def load_ood_mask(path_img: str, size_hw: Tuple[int, int]) -> np.ndarray:
    """
    Loads, resizes (nearest-neighbour), and remaps the GT mask for a given image.

    Args:
        path_img : path to the RGB image (used to derive GT path)
        size_hw  : (H, W) target resolution — must match the model input size

    Returns:
        np.ndarray uint8 [H, W] with values in {0, 1, 255}
    """
    path_gt = gt_path_from_image(path_img)
    mask    = Image.open(path_gt)
    mask    = Resize(size_hw, Image.NEAREST)(mask)
    ood     = np.array(mask)
    return remap_ood_mask(path_gt, ood)


def has_ood_pixels(ood: np.ndarray) -> bool:
    """Returns True if the mask contains at least one OOD pixel (label == 1)."""
    return bool((ood == 1).any())
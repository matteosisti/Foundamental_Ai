"""
src/utils/artifacts.py

Artifact management: run directory creation, config saving, and
run discovery for offline temperature sweeps.

Public API:
    create_run_dir                  — create timestamped run folder + config.json
    update_run_config               — patch an existing config.json
    resolve_latest_run_dir          — find most recent run for a model/dataset
    resolve_latest_run_dir_filtered — find most recent run matching method/mode,
                                      optionally checking that logit cache files exist
"""

import os
import re
import json
import hashlib
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple


# ---------------------------------------------------------------------------
# Run directory naming — shared regex (used by both ERFNet and EoMT sweeps)
# Format: YYYY-MM-DD_HH-MM-SS__method__T<float>__mode__<hash8>
# ---------------------------------------------------------------------------

RUN_RE = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})"
    r"__"
    r"(?P<method>[a-zA-Z0-9\-]+)"
    r"__T(?P<T>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
    r"__"
    r"(?P<mode>robust|prof-exact)"
    r"__"
    r"(?P<hash>[0-9a-fA-F]+)$"
)


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class ArtifactPaths:
    """
    Run-specific directory layout.
    Ensures consistent path resolution across evaluation and sweep stages.
    """
    root:         Path
    results:      Path
    logits:       Path
    anomaly_maps: Path
    sweep:        Path


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _short_hash(payload: Dict[str, Any], n: int = 8) -> str:
    """Stable short hash from a run config dict."""
    s = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha1(s).hexdigest()[:n]


def _sha256_file_8(path: str, chunk_size: int = 1024 * 1024) -> str:
    """First 8 chars of the SHA-256 hash of a file (for checkpoint lineage)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()[:8]


def _normalize_float(x: Optional[float]) -> Optional[float]:
    return None if x is None else float(x)


def _list_run_dirs(base: Path) -> List[Path]:
    """All subdirectories under base (non-recursive)."""
    if not base.exists():
        return []
    return [p for p in base.iterdir() if p.is_dir()]


def _parse_run_dir_name(name: str) -> Optional[Dict[str, Any]]:
    """
    Parse a run directory name into its components using RUN_RE.
    Returns None if the name does not match the expected format.
    """
    m = RUN_RE.match(name)
    if not m:
        return None
    d = m.groupdict()
    return {
        "ts":     d["ts"],
        "method": d["method"].lower(),
        "T":      float(d["T"]),
        "mode":   d["mode"].lower(),
        "hash":   d["hash"].lower(),
    }


# ---------------------------------------------------------------------------
# Run directory creation
# ---------------------------------------------------------------------------

def create_run_dir(
    artifacts_root: str,
    dataset:        str,
    model:          str,
    method:         str,
    temperature:    Optional[float],
    mode:           str,
    extra:          Optional[Dict[str, Any]] = None,
    timestamp:      Optional[str] = None,
    hash_files:     Optional[Dict[str, str]] = None,
    name_style:     str = "pretty",
) -> ArtifactPaths:
    """
    Creates a unique timestamped directory for an experiment run and saves
    a config.json for full reproducibility.

    Directory name format (name_style='pretty'):
        <timestamp>__<method>__T<temperature>__<mode>__<hash8>

    Args:
        artifacts_root : root folder for all artifacts
        dataset        : dataset identifier (e.g. 'RA21')
        model          : model name (e.g. 'EoMT', 'ERFNet')
        method         : anomaly method (e.g. 'msp', 'rba')
        temperature    : temperature value (None -> 'NA' in folder name)
        mode           : 'robust' or 'prof-exact'
        extra          : additional fields merged into config.json
        timestamp      : override auto-generated timestamp
        hash_files     : dict of {key: filepath} to hash for lineage tracking
        name_style     : 'pretty' (default) or 'compact'

    Returns:
        ArtifactPaths with .root, .results, .logits, .anomaly_maps, .sweep
    """
    root = Path(os.path.expanduser(artifacts_root))

    if timestamp is None:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    meta: Dict[str, Any] = {
        "dataset":     dataset,
        "model":       model,
        "method":      method,
        "temperature": _normalize_float(temperature),
        "mode":        mode,
    }
    if extra:
        meta.update(extra)

    if hash_files:
        hash_block: Dict[str, Any] = {}
        for k, p in hash_files.items():
            if p is None:
                hash_block[k] = None
                continue
            pp = os.path.expanduser(str(p))
            hash_block[f"{k}_basename"] = Path(pp).name
            try:
                hash_block[f"{k}_hash8"] = _sha256_file_8(pp)
            except Exception:
                hash_block[f"{k}_hash8"] = None
        meta.update(hash_block)

    run_id = _short_hash(meta)
    T_str  = "NA" if temperature is None else str(float(temperature))

    if name_style == "pretty":
        run_name = f"{timestamp}__{method}__T{T_str}__{mode}__{run_id}"
    else:
        run_name = f"{timestamp}{method}T{T_str}{mode}__{run_id}"

    run_root = root / dataset / model / run_name

    paths = ArtifactPaths(
        root=        run_root,
        results=     run_root / "results",
        logits=      run_root / "logits",
        anomaly_maps=run_root / "anomaly_maps",
        sweep=       run_root / "sweep",
    )

    for p in [paths.root, paths.results, paths.logits, paths.anomaly_maps, paths.sweep]:
        p.mkdir(parents=True, exist_ok=True)

    with open(paths.root / "config.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    return paths


# ---------------------------------------------------------------------------
# Config patching
# ---------------------------------------------------------------------------

def update_run_config(run_root: Path, patch: Dict[str, Any]) -> None:
    """Merge a patch dict into an existing <run_root>/config.json."""
    run_root = Path(run_root)
    cfg = run_root / "config.json"
    if not cfg.exists():
        raise FileNotFoundError(f"Missing config.json in: {run_root}")
    with open(cfg, "r", encoding="utf-8") as f:
        data = json.load(f)
    data.update(patch)
    with open(cfg, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Run discovery
# ---------------------------------------------------------------------------

def resolve_latest_run_dir(
    artifacts_root: str,
    dataset:        str,
    model:          str,
) -> Path:
    """Return the most recent run directory for a given model/dataset."""
    base = Path(os.path.expanduser(artifacts_root)) / dataset / model
    runs = _list_run_dirs(base)
    if not runs:
        raise FileNotFoundError(f"No runs found in: {base}")
    return sorted(runs, key=lambda p: p.name)[-1]


def resolve_latest_run_dir_filtered(
    artifacts_root: str,
    dataset:        str,
    model:          str,
    method:         Optional[str] = None,
    mode:           Optional[str] = None,
    require_logits: bool = False,
    logit_files:    Optional[List[str]] = None,
) -> Path:
    """
    Find the most recent run directory matching method and mode,
    optionally verifying that required logit cache files exist on disk.

    Args:
        artifacts_root : root artifacts folder
        dataset        : dataset identifier (e.g. 'RA21')
        model          : model name (e.g. 'EoMT', 'ERFNet')
        method         : filter by anomaly method (None = any)
        mode           : filter by mode (None = any)
        require_logits : if True, skip runs whose logits/ folder is missing
                         the files listed in logit_files
        logit_files    : list of filenames to check inside logits/
                         (e.g. ['RA21__logits.npy', 'RA21__gt.npy'])
                         ignored when require_logits=False

    Returns:
        Path to the most recent matching run directory.

    Raises:
        FileNotFoundError if no matching run is found.
    """
    base = Path(os.path.expanduser(artifacts_root)) / dataset / model
    if not base.exists():
        raise FileNotFoundError(f"Artifacts base folder not found: {base}")

    method_f = method.lower().strip() if method else None
    mode_f   = mode.lower().strip()   if mode   else None

    candidates: List[Tuple[str, Path]] = []

    for run_dir in _list_run_dirs(base):
        info = _parse_run_dir_name(run_dir.name)
        if info is None:
            continue
        if method_f is not None and info["method"] != method_f:
            continue
        if mode_f is not None and info["mode"] != mode_f:
            continue

        # Optionally verify that logit cache files are present
        if require_logits and logit_files:
            logits_dir = run_dir / "logits"
            if not all((logits_dir / f).exists() for f in logit_files):
                continue

        candidates.append((info["ts"], run_dir))

    if not candidates:
        raise FileNotFoundError(
            f"No matching run found in {base} "
            f"for method={method}, mode={mode}, require_logits={require_logits}."
        )

    # Most recent by timestamp (lexicographic sort is safe with ISO format)
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]

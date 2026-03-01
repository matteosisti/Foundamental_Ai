import os
import json
import hashlib
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List

@dataclass
class ArtifactPaths:
    """
    Data container for run-specific directories.
    Ensures consistent path resolution across evaluation and sweep stages.
    """
    root: Path
    results: Path
    logits: Path
    anomaly_maps: Path
    sweep: Path

def _short_hash(payload: Dict[str, Any], n: int = 8) -> str:
    """Generates a stable short hash from the run configuration."""
    s = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha1(s).hexdigest()[:n]

def _sha256_file_8(path: str, chunk_size: int = 1024 * 1024) -> str:
    """Returns the first 8 characters of the sha256 hash for a given file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()[:8]

def _normalize_float(x: Optional[float]) -> Optional[float]:
    if x is None:
        return None
    return float(x)

def _list_run_dirs(base: Path) -> List[Path]:
    """Helper to list all subdirectories in a model-specific folder."""
    if not base.exists():
        return []
    return [p for p in base.iterdir() if p.is_dir()]

def create_run_dir(
    artifacts_root: str,
    dataset: str,
    model: str,
    method: str,
    temperature: Optional[float],
    mode: str,
    extra: Optional[Dict[str, Any]] = None,
    timestamp: Optional[str] = None,
    hash_files: Optional[Dict[str, str]] = None,
    name_style: str = "pretty",
) -> ArtifactPaths:
    """
    Creates a unique, timestamped directory for each experiment run.
    Also saves a config.json for full experiment reproducibility.
    """
    root = Path(os.path.expanduser(artifacts_root))

    if timestamp is None:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    meta: Dict[str, Any] = {
        "dataset": dataset,
        "model": model,
        "method": method,
        "temperature": _normalize_float(temperature),
        "mode": mode,
    }

    if extra:
        meta.update(extra)

    # Add file hashes (e.g., checkpoints) for lineage tracking
    if hash_files:
        hash_block = {}
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
    T_str = "NA" if temperature is None else str(float(temperature))

    # Define the folder name using a consistent naming convention
    if name_style == "pretty":
        run_name = f"{timestamp}__{method}__T{T_str}__{mode}__{run_id}"
    else:
        run_name = f"{timestamp}{method}T{T_str}{mode}__{run_id}"

    run_root = root / dataset / model / run_name

    paths = ArtifactPaths(
        root=run_root,
        results=run_root / "results",
        logits=run_root / "logits",
        anomaly_maps=run_root / "anomaly_maps",
        sweep=run_root / "sweep",
    )

    # Create directories physically on disk
    for p in [paths.root, paths.results, paths.logits, paths.anomaly_maps, paths.sweep]:
        p.mkdir(parents=True, exist_ok=True)

    # Save the configuration for auditing
    with open(paths.root / "config.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    return paths

def update_run_config(run_root: Path, patch: Dict[str, Any]) -> None:
    """Updates <run_root>/config.json by merging a patch dictionary."""
    run_root = Path(run_root)
    cfg = run_root / "config.json"
    if not cfg.exists():
        raise FileNotFoundError(f"Missing config.json in: {run_root}")

    with open(cfg, "r", encoding="utf-8") as f:
        data = json.load(f)

    data.update(patch)

    with open(cfg, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def resolve_latest_run_dir(artifacts_root: str, dataset: str, model: str) -> Path:
    """Automated discovery of the most recent experiment based on timestamp."""
    base = Path(os.path.expanduser(artifacts_root)) / dataset / model
    runs = _list_run_dirs(base)

    if len(runs) == 0:
        raise FileNotFoundError(f"No runs found in: {base}")

    # Lexicographical sort on ISO timestamps ensures the last run is selected
    runs = sorted(runs, key=lambda p: p.name)
    return runs[-1]

def resolve_latest_run_dir_filtered(
    artifacts_root: str,
    dataset: str,
    model: str,
    method: Optional[str] = None,
    mode: Optional[str] = None,
) -> Path:
    """
    Discovery tool that filters by method/mode from the folder name.
    Critical for matching sweep logic to the correct cached logits.
    """
    base = Path(os.path.expanduser(artifacts_root)) / dataset / model
    runs = _list_run_dirs(base)
    
    if len(runs) == 0:
        raise FileNotFoundError(f"No runs found in: {base}")

    def matches(p: Path) -> bool:
        name = p.name
        if method is not None and f"__{method}__" not in name:
            return False
        if mode is not None and f"__{mode}__" not in name:
            return False
        return True

    filtered_runs = [p for p in runs if matches(p)]
    
    if len(filtered_runs) == 0:
        raise FileNotFoundError(f"No runs matching method={method} mode={mode} in: {base}")

    # Return the most recent run among the filtered ones
    filtered_runs = sorted(filtered_runs, key=lambda p: p.name)
    return filtered_runs[-1]
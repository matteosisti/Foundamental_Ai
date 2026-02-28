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
    """
    Generates a stable short hash from the run configuration.
    Used to differentiate runs with identical timestamps but different parameters.
    """
    s = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha1(s).hexdigest()[:n]


def create_run_dir(
    artifacts_root: str,
    dataset: str,
    model: str,
    method: str,
    temperature: Optional[float],
    mode: str,
    extra: Optional[Dict[str, Any]] = None,
    timestamp: Optional[str] = None,
) -> ArtifactPaths:
    """
    Creates a unique, timestamped directory for each experiment run.
    Structure: <root>/<dataset>/<model>/<timestamp>__<method>__T<T>__<mode>__<hash>
    Also saves a config.json for full experiment reproducibility.
    """
    root = Path(os.path.expanduser(artifacts_root))

    if timestamp is None:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Metadata for hashing and reproducibility
    meta = {
        "dataset": dataset,
        "model": model,
        "method": method,
        "temperature": None if temperature is None else float(temperature),
        "mode": mode,
    }
    if extra:
        meta.update(extra)

    run_id = _short_hash(meta)
    T_str = "NA" if temperature is None else str(float(temperature))
    run_name = f"{timestamp}__{method}__T{T_str}__{mode}__{run_id}"

    run_root = root / dataset / model / run_name

    paths = ArtifactPaths(
        root=run_root,
        results=run_root / "results",
        logits=run_root / "logits",
        anomaly_maps=run_root / "anomaly_maps",
        sweep=run_root / "sweep",
    )

    # Initialize physical directories
    for p in [paths.root, paths.results, paths.logits, paths.anomaly_maps, paths.sweep]:
        p.mkdir(parents=True, exist_ok=True)

    # Save run configuration (The 'Black Box' record)
    with open(paths.root / "config.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    return paths


def _list_run_dirs(base: Path) -> List[Path]:
    """Helper to list all subdirectories in a model-specific folder."""
    if not base.exists():
        return []
    return [p for p in base.iterdir() if p.is_dir()]


def resolve_latest_run_dir(artifacts_root: str, dataset: str, model: str) -> Path:
    """
    Automated discovery of the most recent experiment.
    Lexicographical sort on ISO timestamps ensures the last run is selected.
    """
    base = Path(os.path.expanduser(artifacts_root)) / dataset / model
    runs = _list_run_dirs(base)

    if len(runs) == 0:
        raise FileNotFoundError(f"No runs found in: {base}")

    # Lexicographical sort works because folder names start with YYYY-MM-DD
    runs = sorted(runs, key=lambda p: p.name)
    return runs[-1]
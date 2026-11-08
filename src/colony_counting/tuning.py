"""Validation-only Optuna search for configurable classical CV parameters."""

from __future__ import annotations

import csv
import hashlib
import json
import platform
import sys
import time
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import yaml

from colony_counting.classical import count_colonies, load_classical_config


def tune_classical(
    *,
    manifest_path: Path,
    images_root: Path,
    config_path: Path,
    output_dir: Path,
    trials: int = 50,
    seed: int = 42,
    required_split: str = "val",
) -> dict[str, Any]:
    """Minimize validation MAE and persist the search history and best config."""

    try:
        import optuna
    except ImportError as exc:
        raise RuntimeError("Classical tuning requires Optuna; install the analysis extra.") from exc
    if trials < 1:
        raise ValueError("trials must be positive")
    rows = _read_validation_rows(Path(manifest_path), required_split)
    image_root = Path(images_root)
    cases: list[tuple[str, Path, int]] = []
    for row in rows:
        path = Path(str(row["image_path"]))
        path = path if path.is_absolute() else image_root / path
        if not path.is_file():
            raise FileNotFoundError(f"Validation image is missing: {path}")
        ground_truth = int(row["count"])
        if ground_truth < 0:
            raise ValueError(f"Negative GT count in manifest for {row['image_id']}.")
        cases.append((str(row["image_id"]), path, ground_truth))
    source_config = load_classical_config(config_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    source_hash = _fingerprint(cases, Path(manifest_path))
    started_at = datetime.now(timezone.utc).isoformat()
    started = time.perf_counter()
    trial_rows: list[dict[str, Any]] = []

    def objective(trial: Any) -> float:
        config = replace(
            source_config,
            channel=trial.suggest_categorical("channel", ["gray", "blue", "green", "h"]),
            illumination_correction=trial.suggest_categorical(
                "illumination_correction", [False, True]
            ),
            background_sigma=trial.suggest_float("background_sigma", 8.0, 60.0),
            clahe_clip_limit=trial.suggest_float("clahe_clip_limit", 1.0, 4.0),
            threshold_method=trial.suggest_categorical("threshold_method", ["otsu", "adaptive"]),
            polarity=trial.suggest_categorical("polarity", ["dark", "bright"]),
            morphology_kernel=trial.suggest_categorical("morphology_kernel", [1, 3, 5, 7]),
            opening_iterations=trial.suggest_int("opening_iterations", 0, 2),
            closing_iterations=trial.suggest_int("closing_iterations", 0, 2),
            peak_min_distance=trial.suggest_int("peak_min_distance", 3, 15),
            peak_min_fraction=trial.suggest_float("peak_min_fraction", 0.08, 0.45),
            min_colony_area=trial.suggest_int("min_colony_area", 3, 30),
            cluster_area_correction=trial.suggest_categorical(
                "cluster_area_correction", [False, True]
            ),
        )
        absolute_errors: list[float] = []
        predictions: list[int] = []
        for _, path, ground_truth in cases:
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"Cannot decode validation image: {path}")
            predicted = count_colonies(image, config).count
            predictions.append(predicted)
            absolute_errors.append(float(abs(predicted - ground_truth)))
        trial.set_user_attr("all_zero_predictions", bool(predictions and not any(predictions)))
        trial.set_user_attr("prediction_unique_counts", len(set(predictions)))
        score = sum(absolute_errors) / len(absolute_errors)
        trial_rows.append(
            {
                "trial": trial.number,
                "state": "COMPLETE",
                "objective_validation_mae": score,
                "all_zero_predictions": bool(predictions and not any(predictions)),
                "prediction_unique_counts": len(set(predictions)),
                **trial.params,
            }
        )
        return score

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=trials, n_jobs=1, show_progress_bar=False)
    completed = [trial for trial in study.trials if trial.value is not None]
    if not completed:
        raise RuntimeError("Optuna completed no valid trials.")
    best_config = replace(source_config, **study.best_params)
    config_output = output_dir / "best_classical.yaml"
    temp_config = config_output.with_suffix(".yaml.tmp")
    temp_config.write_text(yaml.safe_dump(asdict(best_config), sort_keys=True), encoding="utf-8")
    temp_config.replace(config_output)
    history_path = output_dir / "objective_history.csv"
    fields = sorted({key for row in trial_rows for key in row})
    with history_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(trial_rows)
    metadata = {
        "experiment_id": f"classical-optuna-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}",
        "status": "completed",
        "objective": "mean absolute colony-count error on the supplied frozen validation split",
        "required_split": required_split,
        "search_space": {
            "channel": ["gray", "blue", "green", "h"],
            "illumination_correction": [False, True],
            "background_sigma": [8.0, 60.0],
            "clahe_clip_limit": [1.0, 4.0],
            "threshold_method": ["otsu", "adaptive"],
            "polarity": ["dark", "bright"],
            "morphology_kernel": [1, 3, 5, 7],
            "opening_iterations": [0, 2],
            "closing_iterations": [0, 2],
            "peak_min_distance": [3, 15],
            "peak_min_fraction": [0.08, 0.45],
            "min_colony_area": [3, 30],
            "cluster_area_correction": [False, True],
        },
        "seed": seed,
        "requested_trials": trials,
        "completed_trials": len(completed),
        "validation_images": len(cases),
        "manifest_sha256": source_hash,
        "base_config": str(config_path),
        "base_config_sha256": _sha256(Path(config_path)),
        "source_sha256": _source_fingerprint(Path(__file__).parents[0]),
        "best_validation_mae": float(study.best_value),
        "best_parameters": study.best_params,
        "best_config": str(config_output),
        "objective_history_csv": str(history_path),
        "contains_all_zero_trials": any(bool(row["all_zero_predictions"]) for row in trial_rows),
        "started_at_utc": started_at,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "wall_seconds": time.perf_counter() - started,
        "host_platform": platform.platform(),
        "software": {
            "python": platform.python_version(),
            "opencv": cv2.__version__,
            "optuna": optuna.__version__,
            "executable": sys.executable,
        },
        "hardware": _hardware_report(),
        "selection_note": "Final test data must not be supplied to this command.",
    }
    metadata_path = output_dir / "experiment.json"
    _write_json_atomic(metadata, metadata_path)
    return metadata


def _read_validation_rows(manifest_path: Path, split: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(manifest_path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid manifest JSON at {manifest_path}:{line_number}") from exc
        if not isinstance(row, dict) or row.get("image_id") is None:
            raise ValueError(f"Invalid manifest row {line_number}: expected image_id object.")
        if row.get("split") == split:
            if "count" not in row or "image_path" not in row:
                raise ValueError(
                    f"Validation manifest row {line_number} lacks count or image_path."
                )
            rows.append(row)
    if not rows:
        raise ValueError(f"No images are assigned to required tuning split {split!r}.")
    return rows


def _fingerprint(cases: list[tuple[str, Path, int]], manifest: Path) -> str:
    digest = hashlib.sha256()
    digest.update(manifest.read_bytes())
    for image_id, path, count in cases:
        digest.update(image_id.encode())
        digest.update(path.read_bytes())
        digest.update(str(count).encode())
    return digest.hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_fingerprint(source_dir: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(source_dir.rglob("*.py")):
        digest.update(path.relative_to(source_dir).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _hardware_report() -> dict[str, str | bool | None]:
    try:
        import torch

        cuda_available = bool(torch.cuda.is_available())
        gpu = torch.cuda.get_device_name(0) if cuda_available else None
    except (ImportError, OSError, RuntimeError):
        cuda_available, gpu = False, None
    return {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cuda_available": cuda_available,
        "gpu": gpu,
    }


def _write_json_atomic(payload: dict[str, Any], destination: Path) -> None:
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(destination)

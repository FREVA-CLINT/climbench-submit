"""Runtime metadata ingestion from local evaluation logs."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import re
from pathlib import Path
from typing import Any, Literal

import yaml

from climbench_submit.schema import RuntimeMetadata

RuntimeMetadataSourceKind = Literal["custom_yaml", "wandb_local", "mlflow_local"]

_WANDB_METADATA_FILENAME = "wandb-metadata.json"
_WANDB_SUMMARY_FILENAME = "wandb-summary.json"
_MLFLOW_META_FILENAME = "meta.yaml"
_METADATA_ATTR_NAME = "climbench_runtime_metadata"


def load_runtime_metadata(
    source: str | Path, source_kind: RuntimeMetadataSourceKind | None = None
) -> RuntimeMetadata:
    """Load normalized runtime metadata from a local source."""

    source_path = Path(source)
    if not source_path.exists():
        raise FileNotFoundError(
            f"Runtime metadata source does not exist: {source_path}"
        )

    kind = source_kind or detect_runtime_metadata_source_kind(source_path)
    if kind == "custom_yaml":
        return _load_custom_yaml_runtime_metadata(source_path)
    if kind == "wandb_local":
        return _load_wandb_local_runtime_metadata(source_path)
    if kind == "mlflow_local":
        return _load_mlflow_local_runtime_metadata(source_path)
    raise ValueError(f"Unsupported runtime metadata source kind: {kind!r}")


def detect_runtime_metadata_source_kind(source_path: Path) -> RuntimeMetadataSourceKind:
    """Infer the runtime metadata source kind from the local path."""

    if source_path.is_file() and source_path.suffix.lower() in {".yaml", ".yml"}:
        return "custom_yaml"
    wandb = _find_wandb_metadata_file(source_path) is not None
    mlflow = _find_mlflow_meta_file(source_path) is not None
    if wandb and mlflow:
        raise ValueError("Ambiguous tracking directory; specify --source-kind.")
    if wandb:
        return "wandb_local"
    if mlflow:
        return "mlflow_local"
    raise ValueError(
        "Could not infer runtime metadata source kind from the provided path. "
        "Pass --runtime-metadata-source-kind explicitly."
    )


def _load_custom_yaml_runtime_metadata(source_path: Path) -> RuntimeMetadata:
    with source_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(
            f"Custom runtime metadata YAML must contain a mapping: {source_path}"
        )
    if "runtime_metadata" in payload and isinstance(payload["runtime_metadata"], dict):
        payload = payload["runtime_metadata"]
    if not isinstance(payload, dict):
        raise ValueError(
            f"Custom runtime metadata YAML must contain a mapping: {source_path}"
        )
    if payload.get("source_kind") not in (None, "custom_yaml"):
        raise ValueError(
            "Custom runtime metadata YAML must not declare a different source_kind."
        )
    payload = dict(payload)
    payload.setdefault("source_kind", "custom_yaml")
    payload.setdefault("source_path", str(source_path))
    payload.setdefault("framework", "custom")
    return RuntimeMetadata.model_validate(payload)


def _load_wandb_local_runtime_metadata(source_path: Path) -> RuntimeMetadata:
    run_dir = source_path if source_path.is_dir() else source_path.parent
    metadata_path = _find_wandb_metadata_file(run_dir)
    if metadata_path is None:
        raise FileNotFoundError(
            f"Could not find W&B metadata file under: {source_path}"
        )

    metadata = _read_json(metadata_path)
    summary_path = _find_wandb_summary_file(run_dir)
    summary = _read_json(summary_path) if summary_path is not None else {}
    log_text = _collect_text_from_candidates(
        [
            run_dir / "files" / "output.log",
            run_dir / "logs" / "debug.log",
            run_dir / "logs" / "debug-internal.log",
            run_dir / "logs" / "debug-core.log",
        ]
    )

    gpu_details = metadata.get("gpu_nvidia") or []
    gpu_memory_total_bytes = _first_int_from_mapping_list(gpu_details, "memoryTotal")
    wallclock_seconds = _float_or_none(
        summary.get("_runtime") or summary.get("_wandb", {}).get("runtime")
    )
    if wallclock_seconds is None and metadata.get("startedAt"):
        wallclock_seconds = _duration_from_timestamps(
            metadata.get("startedAt"), metadata.get("endedAt")
        )

    runtime_metadata = RuntimeMetadata(
        source_kind="wandb_local",
        source_path=str(run_dir),
        framework="wandb",
        run_id=_infer_wandb_run_id(run_dir),
        run_name=metadata.get("run_name") or run_dir.name,
        host=metadata.get("host"),
        python=metadata.get("python"),
        executable=metadata.get("executable"),
        gpu_count=_int_or_none(metadata.get("gpu_count")),
        gpu_name=metadata.get("gpu"),
        gpu_memory_total_bytes=gpu_memory_total_bytes,
        host_memory_total_bytes=_int_or_none(
            _nested_value(metadata, ("memory", "total"))
        ),
        wallclock_seconds=wallclock_seconds,
        started_at=metadata.get("startedAt"),
        ended_at=metadata.get("endedAt"),
        hardware={
            "os": metadata.get("os"),
            "cuda_version": metadata.get("cudaVersion"),
            "cpu_count": metadata.get("cpu_count"),
            "cpu_count_logical": metadata.get("cpu_count_logical"),
            "gpu_nvidia": gpu_details,
        },
        provenance={
            "args": metadata.get("args", []),
            "program": metadata.get("program"),
            "email": metadata.get("email"),
            "root": metadata.get("root"),
            "slurm": metadata.get("slurm", {}),
            "writer_id": metadata.get("writerId"),
            "summary": summary,
            "log_excerpt": _tail_excerpt(log_text),
        },
    )

    iterations_total = _infer_iterations_total(summary, log_text)
    if iterations_total is not None:
        runtime_metadata.iterations_total = iterations_total
    iterations_per_sec = _infer_iterations_per_sec(
        runtime_metadata.iterations_total, runtime_metadata.wallclock_seconds, log_text
    )
    if iterations_per_sec is not None:
        runtime_metadata.iterations_per_sec = iterations_per_sec
    peak_ram_bytes = _infer_peak_memory_bytes(log_text, ("ram", "memory", "rss"))
    if peak_ram_bytes is not None:
        runtime_metadata.peak_ram_bytes = peak_ram_bytes
    peak_vram_bytes = _infer_peak_memory_bytes(
        log_text, ("vram", "gpu memory", "cuda memory")
    )
    if peak_vram_bytes is not None:
        runtime_metadata.peak_vram_bytes = peak_vram_bytes
    return runtime_metadata


def _load_mlflow_local_runtime_metadata(source_path: Path) -> RuntimeMetadata:
    run_dir = source_path if source_path.is_dir() else source_path.parent
    meta_path = _find_mlflow_meta_file(run_dir)
    if meta_path is None:
        raise FileNotFoundError(f"Could not find MLflow meta.yaml under: {source_path}")

    meta = yaml.safe_load(meta_path.read_text(encoding="utf-8")) or {}
    if not isinstance(meta, dict):
        raise ValueError(f"MLflow meta.yaml must contain a mapping: {meta_path}")

    tags = _read_text_mapping(run_dir / "tags")
    params = _read_text_mapping(run_dir / "params")
    metrics = _read_mlflow_metrics(run_dir / "metrics")
    wallclock_seconds = _duration_seconds_from_ms(
        meta.get("start_time"), meta.get("end_time")
    )

    run_id = meta.get("run_id") or meta.get("run_uuid")
    runtime_metadata = RuntimeMetadata(
        source_kind="mlflow_local",
        source_path=str(run_dir),
        framework="mlflow",
        run_id=str(run_id) if run_id is not None else None,
        run_name=tags.get("mlflow.runName") or meta.get("run_name"),
        host=meta.get("host"),
        python=None,
        executable=meta.get("entry_point_name"),
        wallclock_seconds=wallclock_seconds,
        started_at=_timestamp_ms_to_iso(meta.get("start_time")),
        ended_at=_timestamp_ms_to_iso(meta.get("end_time")),
        hardware={
            "artifact_uri": meta.get("artifact_uri"),
            "experiment_id": meta.get("experiment_id"),
            "lifecycle_stage": meta.get("lifecycle_stage"),
            "status": meta.get("status"),
            "user_id": meta.get("user_id"),
        },
        provenance={
            "source_name": tags.get("mlflow.source.name"),
            "source_type": tags.get("mlflow.source.type"),
            "tags": tags,
            "params": params,
            "metrics": metrics,
            "meta": meta,
        },
    )

    iterations_total = _infer_iterations_total_from_mlflow_metrics(metrics)
    if iterations_total is not None:
        runtime_metadata.iterations_total = iterations_total
    iterations_per_sec = _infer_iterations_per_sec(
        runtime_metadata.iterations_total, runtime_metadata.wallclock_seconds, ""
    )
    if iterations_per_sec is not None:
        runtime_metadata.iterations_per_sec = iterations_per_sec
    peak_ram_bytes = _find_metric_value(
        metrics, {"peak_ram_bytes", "max_ram_bytes", "peak_rss_bytes", "max_rss_bytes"}
    )
    if peak_ram_bytes is not None:
        runtime_metadata.peak_ram_bytes = int(peak_ram_bytes)
    peak_vram_bytes = _find_metric_value(
        metrics,
        {
            "peak_vram_bytes",
            "max_vram_bytes",
            "peak_gpu_memory_bytes",
            "max_gpu_memory_bytes",
        },
    )
    if peak_vram_bytes is not None:
        runtime_metadata.peak_vram_bytes = int(peak_vram_bytes)
    gpu_count = _find_metric_value(metrics, {"gpu_count", "num_gpus"})
    if gpu_count is not None:
        runtime_metadata.gpu_count = int(gpu_count)
    return runtime_metadata


def _find_wandb_metadata_file(source_path: Path) -> Path | None:
    candidates = [
        source_path / "files" / _WANDB_METADATA_FILENAME,
        source_path / _WANDB_METADATA_FILENAME,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    matches = sorted(source_path.rglob(_WANDB_METADATA_FILENAME))
    if len(matches) > 1:
        raise ValueError("Multiple W&B runs found; select one run directory.")
    return matches[0] if matches else None


def _find_wandb_summary_file(source_path: Path) -> Path | None:
    candidates = [
        source_path / "files" / _WANDB_SUMMARY_FILENAME,
        source_path / _WANDB_SUMMARY_FILENAME,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    for candidate in source_path.rglob(_WANDB_SUMMARY_FILENAME):
        return candidate
    return None


def _find_mlflow_meta_file(source_path: Path) -> Path | None:
    candidates = [source_path / _MLFLOW_META_FILENAME]
    if source_path.is_dir():
        for candidate in candidates:
            if candidate.exists():
                return candidate
        matches = sorted(source_path.rglob(_MLFLOW_META_FILENAME))
        if len(matches) > 1:
            raise ValueError("Multiple MLflow runs found; select one run directory.")
        return matches[0] if matches else None
    return None


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _collect_text_from_candidates(paths: list[Path]) -> str:
    chunks: list[str] = []
    for path in paths:
        if path.exists():
            chunks.append(path.read_text(encoding="utf-8", errors="replace"))
    return "\n".join(chunks)


def _tail_excerpt(text: str, max_lines: int = 12) -> str | None:
    if not text.strip():
        return None
    lines = [line.rstrip() for line in text.splitlines() if line.strip()]
    if not lines:
        return None
    return "\n".join(lines[-max_lines:])


def _nested_value(mapping: dict[str, Any], path: tuple[str, ...]) -> Any | None:
    current: Any = mapping
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
        if current is None:
            return None
    return current


def _int_or_none(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _float_or_none(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _first_int_from_mapping_list(items: list[dict[str, Any]], key: str) -> int | None:
    for item in items:
        if not isinstance(item, dict):
            continue
        value = _int_or_none(item.get(key))
        if value is not None:
            return value
    return None


def _infer_wandb_run_id(run_dir: Path) -> str | None:
    match = re.match(r"run-(?:\d+_\d+-)?(?P<run_id>[a-z0-9]+)", run_dir.name)
    if match:
        return match.group("run_id")
    if "-" in run_dir.name:
        return run_dir.name.rsplit("-", 1)[-1]
    return None


def _duration_from_timestamps(
    started_at: str | None, ended_at: str | None
) -> float | None:
    if not started_at or not ended_at:
        return None
    try:
        started = datetime.fromisoformat(started_at.replace("Z", "+00:00"))
        ended = datetime.fromisoformat(ended_at.replace("Z", "+00:00"))
    except ValueError:
        return None
    return max((ended - started).total_seconds(), 0.0)


def _duration_seconds_from_ms(start_time: Any, end_time: Any) -> float | None:
    start = _float_or_none(start_time)
    end = _float_or_none(end_time)
    if start is None or end is None:
        return None
    return max((end - start) / 1000.0, 0.0)


def _timestamp_ms_to_iso(value: Any) -> str | None:
    millis = _float_or_none(value)
    if millis is None:
        return None
    dt = datetime.fromtimestamp(millis / 1000.0, tz=timezone.utc)
    return dt.isoformat().replace("+00:00", "Z")


def _infer_iterations_total(summary: dict[str, Any], log_text: str) -> int | None:
    for key in ("_step", "step", "steps", "iteration", "iterations", "train_steps"):
        value = _int_or_none(summary.get(key))
        if value is not None:
            return value
    patterns = [
        r"(?:iterations?|iters?|steps?)\s*[:=]\s*(\d+)",
        r"epoch\s+\d+\s*/\s*(\d+)",
    ]
    for text in (log_text,):
        for pattern in patterns:
            matches = [
                int(match) for match in re.findall(pattern, text, flags=re.IGNORECASE)
            ]
            if matches:
                return max(matches)
    return None


def _infer_iterations_total_from_mlflow_metrics(
    metrics: dict[str, list[dict[str, float]]],
) -> int | None:
    for key in ("iterations_total", "iterations", "iters", "num_iterations"):
        metric_entries = metrics.get(key)
        if metric_entries:
            value = metric_entries[-1].get("value")
            int_value = _int_or_none(value)
            if int_value is not None:
                return int_value
    for key in ("step", "steps", "train_steps"):
        metric_entries = metrics.get(key)
        if metric_entries:
            return max(_int_or_none(entry.get("step")) or 0 for entry in metric_entries)
    return None


def _infer_iterations_per_sec(
    iterations_total: int | None, wallclock_seconds: float | None, log_text: str
) -> float | None:
    if iterations_total is not None and wallclock_seconds and wallclock_seconds > 0:
        return iterations_total / wallclock_seconds
    match = re.search(
        r"(\d+(?:\.\d+)?)\s*(?:it/s|iter/s|iterations/s|steps/s)",
        log_text,
        flags=re.IGNORECASE,
    )
    if match:
        return float(match.group(1))
    return None


def _infer_peak_memory_bytes(log_text: str, tokens: tuple[str, ...]) -> int | None:
    patterns = [
        r"(?:peak|max(?:imum)?)\s+(?:%s)[^\d]*(\d+)" % re.escape(token)
        for token in tokens
    ]
    patterns.extend(
        [
            r"(\d+)\s*bytes\s*(?:peak|max)?",
            r"(\d+(?:\.\d+)?)\s*(?:gb|gib|mb|mib|kb|kib)",
        ]
    )
    for pattern in patterns:
        match = re.search(pattern, log_text, flags=re.IGNORECASE)
        if not match:
            continue
        value = match.group(1)
        if "." in value:
            return int(float(value))
        return int(value)
    return None


def _read_text_mapping(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    mapping: dict[str, str] = {}
    if path.is_file():
        return mapping
    for item in path.iterdir():
        if item.is_file():
            mapping[item.name] = item.read_text(
                encoding="utf-8", errors="replace"
            ).strip()
    return mapping


def _read_mlflow_metrics(path: Path) -> dict[str, list[dict[str, float]]]:
    metrics: dict[str, list[dict[str, float]]] = {}
    if not path.exists():
        return metrics
    for metric_file in path.iterdir():
        if not metric_file.is_file():
            continue
        entries: list[dict[str, float]] = []
        for line in metric_file.read_text(
            encoding="utf-8", errors="replace"
        ).splitlines():
            parts = line.split()
            if len(parts) < 3:
                continue
            timestamp, value, step = parts[:3]
            try:
                entries.append(
                    {
                        "timestamp": float(timestamp),
                        "value": float(value),
                        "step": float(step),
                    }
                )
            except ValueError:
                continue
        if entries:
            metrics[metric_file.name] = entries
    return metrics


def _find_metric_value(
    metrics: dict[str, list[dict[str, float]]], candidate_names: set[str]
) -> float | None:
    for name in candidate_names:
        entries = metrics.get(name)
        if entries:
            return entries[-1].get("value")
    return None

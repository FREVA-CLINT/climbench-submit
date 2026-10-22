from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from climbench_submit.metadata import load_runtime_metadata


def test_custom_yaml_runtime_metadata_parser(tmp_path: Path) -> None:
    path = tmp_path / "runtime_metadata.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "source_kind": "custom_yaml",
                "framework": "my-runner",
                "run_id": "run-123",
                "host": "test-host",
                "gpu_count": 2,
                "peak_ram_bytes": 123,
                "iterations_total": 456,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    metadata = load_runtime_metadata(path)

    assert metadata.source_kind == "custom_yaml"
    assert metadata.framework == "my-runner"
    assert metadata.run_id == "run-123"
    assert metadata.gpu_count == 2
    assert metadata.peak_ram_bytes == 123
    assert metadata.iterations_total == 456


def test_wandb_local_runtime_metadata_parser(tmp_path: Path) -> None:
    run_dir = tmp_path / "wandb" / "run-20260529_092616-yohru1io"
    (run_dir / "files").mkdir(parents=True)
    (run_dir / "logs").mkdir()
    (run_dir / "files" / "wandb-metadata.json").write_text(
        json.dumps(
            {
                "host": "test-host",
                "python": "CPython 3.11.14",
                "startedAt": "2026-05-29T07:26:16.764137Z",
                "program": "-m climbench.train",
                "gpu": "NVIDIA A100-SXM4-80GB",
                "gpu_count": 4,
                "memory": {"total": "540759543808"},
                "gpu_nvidia": [{"memoryTotal": "85899345920"}],
                "slurm": {"job_id": "25225968"},
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "files" / "wandb-summary.json").write_text(
        json.dumps({"_wandb": {"runtime": 12}, "_runtime": 12}), encoding="utf-8"
    )
    (run_dir / "files" / "output.log").write_text(
        "training... 12 it/s peak ram 987654321 bytes\n", encoding="utf-8"
    )

    metadata = load_runtime_metadata(run_dir)

    assert metadata.source_kind == "wandb_local"
    assert metadata.framework == "wandb"
    assert metadata.run_id == "yohru1io"
    assert metadata.host == "test-host"
    assert metadata.gpu_count == 4
    assert metadata.gpu_memory_total_bytes == 85899345920
    assert metadata.host_memory_total_bytes == 540759543808
    assert metadata.wallclock_seconds == 12
    assert metadata.provenance["slurm"]["job_id"] == "25225968"


def test_mlflow_local_runtime_metadata_parser(tmp_path: Path) -> None:
    run_dir = tmp_path / "mlruns" / "0" / "abc123"
    (run_dir / "metrics").mkdir(parents=True)
    (run_dir / "tags").mkdir()
    (run_dir / "params").mkdir()
    (run_dir / "meta.yaml").write_text(
        yaml.safe_dump(
            {
                "artifact_uri": "file:///tmp/artifacts",
                "experiment_id": "0",
                "run_id": "abc123",
                "run_name": "demo-run",
                "start_time": 1_000,
                "end_time": 7_000,
                "status": 3,
                "user_id": "joe",
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    (run_dir / "tags" / "mlflow.runName").write_text("demo-run", encoding="utf-8")
    (run_dir / "metrics" / "iterations_total").write_text(
        "1000 42 42\n", encoding="utf-8"
    )
    (run_dir / "metrics" / "iterations_per_sec").write_text(
        "1000 7.0 42\n", encoding="utf-8"
    )
    (run_dir / "metrics" / "peak_ram_bytes").write_text(
        "1000 123456 42\n", encoding="utf-8"
    )
    (run_dir / "metrics" / "gpu_count").write_text("1000 8 42\n", encoding="utf-8")

    metadata = load_runtime_metadata(run_dir)

    assert metadata.source_kind == "mlflow_local"
    assert metadata.framework == "mlflow"
    assert metadata.run_id == "abc123"
    assert metadata.run_name == "demo-run"
    assert metadata.wallclock_seconds == 6
    assert metadata.iterations_total == 42
    assert metadata.iterations_per_sec == 7.0
    assert metadata.peak_ram_bytes == 123456
    assert metadata.gpu_count == 8


def test_runtime_metadata_parser_errors_are_clear(tmp_path: Path) -> None:
    bad_yaml = tmp_path / "bad.yaml"
    bad_yaml.write_text("- not-a-mapping\n", encoding="utf-8")

    with pytest.raises(ValueError, match="must contain a mapping"):
        load_runtime_metadata(bad_yaml)


def test_wandb_runtime_metadata_missing_files_fails_cleanly(tmp_path: Path) -> None:
    (tmp_path / "wandb-run").mkdir()
    with pytest.raises(FileNotFoundError, match="Could not find W&B metadata file"):
        load_runtime_metadata(tmp_path / "wandb-run", "wandb_local")


def test_mlflow_runtime_metadata_missing_files_fails_cleanly(tmp_path: Path) -> None:
    (tmp_path / "mlflow-run").mkdir()
    with pytest.raises(FileNotFoundError, match="Could not find MLflow meta.yaml"):
        load_runtime_metadata(tmp_path / "mlflow-run", "mlflow_local")

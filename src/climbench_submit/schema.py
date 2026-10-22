"""Runtime metadata extracted from evaluation tracking logs."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ClimbenchModel(BaseModel):
    model_config = ConfigDict(extra="allow")


class RuntimeMetadata(ClimbenchModel):
    """Normalized runtime metadata uploaded with predictions."""

    schema_version: str = "1.0"
    source_kind: str
    source_path: str | None = None
    framework: str | None = None
    run_id: str | None = None
    run_name: str | None = None
    host: str | None = None
    python: str | None = None
    executable: str | None = None
    gpu_count: int | None = None
    gpu_name: str | None = None
    gpu_memory_total_bytes: int | None = None
    host_memory_total_bytes: int | None = None
    peak_ram_bytes: int | None = None
    peak_vram_bytes: int | None = None
    iterations_total: int | None = None
    iterations_per_sec: float | None = None
    wallclock_seconds: float | None = None
    started_at: str | None = None
    ended_at: str | None = None
    hardware: dict[str, Any] = Field(default_factory=dict)
    provenance: dict[str, Any] = Field(default_factory=dict)

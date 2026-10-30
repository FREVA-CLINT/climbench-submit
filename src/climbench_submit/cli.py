"""Public CLIs; scientific checks are performed by the platform."""

from __future__ import annotations

from contextlib import ExitStack, contextmanager
from pathlib import Path
import os
import sys
import tempfile
import zipfile

import httpx
import typer
import yaml

from climbench_submit.client import _api_error, _optional_prompt, _submission_metadata
from climbench_submit.metadata import load_runtime_metadata

submit_app = typer.Typer(help="Submit model predictions to ClimBench.")
metadata_app = typer.Typer(help="Export local W&B/MLflow training metadata as YAML.")


@contextmanager
def prediction_upload(path: Path):
    path = path.expanduser()
    if path.is_dir():
        if path.suffix.lower() != ".zarr":
            raise ValueError("Prediction directories must end in .zarr.")
        with tempfile.TemporaryDirectory() as tmp:
            archive = Path(tmp) / f"{path.name}.zip"
            with zipfile.ZipFile(
                archive, "w", compression=zipfile.ZIP_STORED
            ) as output:
                for item in sorted(path.rglob("*")):
                    if item.is_symlink():
                        raise ValueError(
                            "Zarr directories must not contain symbolic links."
                        )
                    if item.is_file():
                        output.write(item, item.relative_to(path.parent))
            yield archive
    else:
        if not path.is_file() or path.suffix.lower() not in {".nc", ".netcdf", ".zip"}:
            raise ValueError(
                "Choose an existing NetCDF file, Zarr ZIP, or .zarr directory."
            )
        yield path


def _size(count: int) -> str:
    if count < 1024:
        return f"{count} B"
    value = count / 1024
    for unit in ("KB", "MB"):
        if value < 1024:
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} GB"


class _ProgressReader:
    """Pass an upload file to httpx and report every chunk it reads."""

    def __init__(self, file, advance):
        self._file = file
        self._advance = advance

    def read(self, size: int = -1) -> bytes:
        chunk = self._file.read(size)
        self._advance(len(chunk))
        return chunk

    def __getattr__(self, name):
        # httpx sizes the request through fileno() and rewinds with seek().
        return getattr(self._file, name)


@contextmanager
def upload_progress(total: int):
    """Yield a callback that advances a byte progress bar for the request body."""

    sent = 0

    def status(_item) -> str:
        text = f"{_size(sent)}/{_size(total)}"
        # The platform validates synchronously, so the response lags the last byte.
        return f"{text}, waiting for platform validation" if sent >= total else text

    with typer.progressbar(
        length=max(total, 1), label="Uploading", item_show_func=status, file=sys.stderr
    ) as bar:

        def advance(count: int) -> None:
            nonlocal sent
            sent += count
            bar.update(count, sent)

        yield advance


@submit_app.command()
def submit(platform_url: str | None = typer.Option(None, "--platform-url")):
    base = (
        platform_url
        or os.environ.get("CLIMBENCH_PLATFORM_URL")
        or "http://localhost:8000"
    ).rstrip("/")
    email = typer.prompt("Email address").strip()
    password = typer.prompt("Password", hide_input=True)
    try:
        # Large uploads and synchronous validation can exceed the old five-minute timeout.
        with httpx.Client(timeout=httpx.Timeout(None, connect=15.0)) as client:
            response = client.post(
                f"{base}/api/v1/auth/login/",
                json={"email": email, "password": password},
            )
            if response.status_code != 200:
                raise ValueError(f"Login failed: {_api_error(response)}")
            login = response.json()
            token = login.get("access")
            if not token:
                raise ValueError(
                    "Login failed: platform did not return an access token."
                )
            headers = {"Authorization": f"Bearer {token}"}
            response = client.get(f"{base}/api/v1/tasks/", headers=headers)
            if response.status_code != 200:
                raise ValueError(_api_error(response))
            tasks = response.json()
            if not isinstance(tasks, list) or not tasks:
                raise ValueError("No active benchmark tasks are available.")
            for number, task in enumerate(tasks, 1):
                typer.echo(
                    f"{number}. {task['name']} ({task['task_id']}:{task['version']})"
                )
            choice = typer.prompt("Task number", type=int)
            if not 1 <= choice <= len(tasks):
                raise ValueError("Task number is outside the displayed list.")
            task = tasks[choice - 1]
            prediction = Path(typer.prompt("Path to model output"))
            compressed = None
            if task["task_family"] == "compression":
                compressed = Path(
                    typer.prompt("Path to compressed artifact")
                ).expanduser()
            runtime_path = _optional_prompt("Path to training/runtime YAML")
            metadata = _submission_metadata(login.get("user") or {})
            metadata["selected_task"] = task["id"]
            if not typer.confirm("Submit predictions for evaluation?", default=True):
                typer.echo("Submission cancelled.")
                return
            with ExitStack() as stack:
                uploads = {
                    "prediction_file": stack.enter_context(prediction_upload(prediction))
                }
                for field, path in (
                    ("compressed_file", compressed),
                    (
                        "runtime_metadata_file",
                        Path(runtime_path).expanduser() if runtime_path else None,
                    ),
                ):
                    if path is not None:
                        if not path.is_file():
                            raise ValueError(f"File does not exist: {path}")
                        uploads[field] = path
                advance = stack.enter_context(
                    upload_progress(sum(path.stat().st_size for path in uploads.values()))
                )
                files = {
                    field: (
                        path.name,
                        _ProgressReader(stack.enter_context(path.open("rb")), advance),
                        "application/octet-stream",
                    )
                    for field, path in uploads.items()
                }
                response = client.post(
                    f"{base}/api/v1/submissions/",
                    headers=headers,
                    data=metadata,
                    files=files,
                )
            if response.status_code != 201:
                raise ValueError(f"Submission failed: {_api_error(response)}")
            payload = response.json()
            typer.echo(f"Submission created: {payload['id']} ({payload['status']}).")
            for warning in payload.get("validation_warnings", []):
                typer.echo(f"Deferred: {warning}")
            if payload.get("detail_url"):
                typer.echo(f"View submission: {payload['detail_url']}")
    except (httpx.RequestError, ValueError, OSError, KeyError, AttributeError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc


@metadata_app.command()
def export(
    source: Path = typer.Option(..., "--source", exists=True, file_okay=False),
    output: Path = typer.Option(..., "--output"),
    source_kind: str | None = typer.Option(
        None,
        "--source-kind",
        help="wandb_local or mlflow_local; inferred when omitted.",
    ),
):
    try:
        if source_kind not in {None, "wandb_local", "mlflow_local"}:
            raise ValueError("Source kind must be wandb_local or mlflow_local.")
        metadata = load_runtime_metadata(source, source_kind)
        output = output.expanduser()
        # Exclusive creation avoids destroying an existing hand-edited export.
        with output.open("x", encoding="utf-8") as handle:
            yaml.safe_dump(
                metadata.model_dump(mode="json", exclude_none=True),
                handle,
                sort_keys=False,
            )
    except (ValueError, OSError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc
    typer.echo(f"Wrote training metadata: {output}")

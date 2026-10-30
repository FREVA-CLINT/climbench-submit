"""Public CLIs; scientific checks are performed by the platform."""

from __future__ import annotations

from contextlib import ExitStack, contextmanager
from pathlib import Path
import os
import tempfile
import zipfile

import httpx
import typer
import yaml

from climbench_submit.client import _api_error, _optional_prompt, _submission_metadata
from climbench_submit.metadata import load_runtime_metadata

submit_app = typer.Typer(help="Submit model predictions to ClimBench.")
metadata_app = typer.Typer(help="Export local W&B/MLflow training metadata as YAML.")
SIGN_BATCH = 200
PUT_ATTEMPTS = 3


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


def direct_upload(
    client: httpx.Client, base: str, headers: dict, task_id: str, files: dict[str, Path]
) -> str | None:
    """Upload files straight to object storage, as the browser does.

    Large bodies never pass the platform's reverse proxy. Returns the completed
    ``direct_upload`` token, or None when the platform has no object storage.
    """

    def post(step: str, payload: dict) -> dict:
        response = client.post(
            f"{base}/api/v1/uploads/{step}/", headers=headers, json=payload
        )
        if response.status_code != 200:
            raise ValueError(f"Upload failed: {_api_error(response)}")
        return response.json()

    response = client.post(
        f"{base}/api/v1/uploads/start/",
        headers=headers,
        json={
            "task": task_id,
            "items": {
                role: {"name": path.name, "size": path.stat().st_size}
                for role, path in files.items()
            },
        },
    )
    # Older platforms (404) or local artifact storage (409) take multipart uploads.
    if response.status_code in (404, 409):
        return None
    if response.status_code != 200:
        raise ValueError(f"Upload failed: {_api_error(response)}")
    started = response.json()
    token, part_size = started["token"], started["part_size"]
    jobs = [
        (role, part)
        for role in files
        for part in range(1, started["items"][role]["parts"] + 1)
    ]
    etags: dict[str, list[dict]] = {}
    for offset in range(0, len(jobs), SIGN_BATCH):
        batch = jobs[offset : offset + SIGN_BATCH]
        # Sign lazily per batch so presigned URLs do not expire during long uploads.
        urls = post(
            "sign",
            {"token": token, "requests": [{"role": r, "part": p} for r, p in batch]},
        )["urls"]
        for index, ((role, part), url) in enumerate(zip(batch, urls), offset + 1):
            with files[role].open("rb") as handle:
                handle.seek((part - 1) * part_size)
                chunk = handle.read(part_size)
            for attempt in range(1, PUT_ATTEMPTS + 1):
                try:
                    put = client.put(url, content=chunk)
                    if put.status_code < 300:
                        break
                    error = f"Storage rejected an upload part (HTTP {put.status_code})."
                except httpx.RequestError as exc:
                    error = f"The connection to the storage service failed: {exc}"
                if attempt == PUT_ATTEMPTS:
                    raise ValueError(error)
                url = post(
                    "sign", {"token": token, "requests": [{"role": role, "part": part}]}
                )["urls"][0]
            etags.setdefault(role, []).append(
                {"part": part, "etag": put.headers.get("ETag")}
            )
            typer.echo(f"\rUploading to storage: part {index}/{len(jobs)}", nl=False)
    typer.echo()
    return post("complete", {"token": token, "parts": etags})["token"]


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
            runtime = Path(runtime_path).expanduser() if runtime_path else None
            for path in (compressed, runtime):
                if path is not None and not path.is_file():
                    raise ValueError(f"File does not exist: {path}")
            with ExitStack() as stack:
                prediction = stack.enter_context(prediction_upload(prediction))
                uploads = {"prediction": prediction}
                if compressed is not None:
                    uploads["compressed"] = compressed
                token = direct_upload(client, base, headers, task["id"], uploads)
                if token:
                    metadata["direct_upload"] = token
                    uploads = {}
                if runtime is not None:
                    uploads["runtime_metadata"] = runtime
                files = {
                    f"{role}_file": (
                        path.name,
                        stack.enter_context(path.open("rb")),
                        "application/octet-stream",
                    )
                    for role, path in uploads.items()
                }
                response = client.post(
                    f"{base}/api/v1/submissions/",
                    headers=headers,
                    data=metadata,
                    files=files or None,
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

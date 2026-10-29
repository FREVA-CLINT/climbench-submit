import json
import zipfile

import httpx
import pytest
import yaml
from typer.testing import CliRunner

from climbench_submit import cli, client

runner = CliRunner()


def test_submission_auth_task_selection_and_upload(tmp_path, monkeypatch):
    prediction = tmp_path / "predictions.nc"
    prediction.write_bytes(b"model output")
    calls = []

    def handle(request):
        calls.append(request)
        if request.url.path.endswith("/auth/login/"):
            assert json.loads(request.content) == {
                "email": "user@example.org",
                "password": "secret",
            }
            return httpx.Response(200, json={"access": "private-token", "user": {}})
        assert request.headers["Authorization"] == "Bearer private-token"
        if request.method == "GET":
            return httpx.Response(
                200,
                json=[
                    {
                        "id": "task-uuid",
                        "task_id": "demo",
                        "version": "1",
                        "name": "Demo",
                        "task_family": "infilling",
                    }
                ],
            )
        body = request.read()
        assert b'name="selected_task"' in body and b"task-uuid" in body
        assert b'name="prediction_file"' in body and b"model output" in body
        assert b"evaluation_result_netcdf_file" not in body
        return httpx.Response(
            201,
            json={
                "id": "upload-uuid",
                "status": "submitted",
                "validation_warnings": ["Exact levels deferred."],
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handle))
    monkeypatch.setattr(cli.httpx, "Client", lambda **kwargs: client)
    monkeypatch.setattr(
        cli,
        "_submission_metadata",
        lambda user: {
            "method_name": "Demo",
            "method_type": "baseline",
            "leakage_statement": "No leakage",
        },
    )
    result = runner.invoke(
        cli.submit_app,
        ["--platform-url", "https://example.org"],
        input=f"user@example.org\nsecret\n1\n{prediction}\n\ny\n",
    )
    assert result.exit_code == 0, result.output
    assert "upload-uuid" in result.output and "Exact levels deferred" in result.output
    assert "private-token" not in result.output
    assert len(calls) == 3


def test_zarr_directory_packaging_preserves_source(tmp_path):
    source = tmp_path / "prediction.zarr"
    source.mkdir()
    (source / "zarr.json").write_text('{"zarr_format":3}')
    with cli.prediction_upload(source) as archive:
        with zipfile.ZipFile(archive) as contents:
            assert contents.read("prediction.zarr/zarr.json") == b'{"zarr_format":3}'
    assert (source / "zarr.json").exists()
    assert not archive.exists()


def test_zarr_symlink_rejected(tmp_path):
    source = tmp_path / "prediction.zarr"
    source.mkdir()
    (source / "link").symlink_to(tmp_path / "outside")
    with pytest.raises(ValueError, match="symbolic links"):
        with cli.prediction_upload(source):
            pass


def test_metadata_export_and_no_overwrite(tmp_path):
    source = tmp_path / "run"
    source.mkdir()
    (source / "wandb-metadata.json").write_text(
        json.dumps({"gpu_count": 2, "gpu": "A100"})
    )
    output = tmp_path / "training.yaml"
    result = runner.invoke(
        cli.metadata_app, ["--source", str(source), "--output", str(output)]
    )
    assert result.exit_code == 0, result.output
    assert yaml.safe_load(output.read_text())["gpu_count"] == 2
    before = output.read_bytes()
    result = runner.invoke(
        cli.metadata_app, ["--source", str(source), "--output", str(output)]
    )
    assert result.exit_code == 1 and output.read_bytes() == before


def test_ambiguous_metadata_requires_source_kind(tmp_path):
    (tmp_path / "wandb-metadata.json").write_text("{}")
    (tmp_path / "meta.yaml").write_text("{}")
    result = runner.invoke(
        cli.metadata_app,
        ["--source", str(tmp_path), "--output", str(tmp_path / "out.yaml")],
    )
    assert result.exit_code == 1
    assert "Ambiguous" in result.output


def test_multiple_tracker_runs_require_one_directory(tmp_path):
    for name in ("run1", "run2"):
        run = tmp_path / name
        run.mkdir()
        (run / "wandb-metadata.json").write_text("{}")
    result = runner.invoke(
        cli.metadata_app,
        ["--source", str(tmp_path), "--output", str(tmp_path / "out.yaml")],
    )
    assert result.exit_code == 1
    assert "Multiple W&B runs" in result.output


def test_submission_metadata_leaves_prediction_type_to_platform(monkeypatch):
    prompts = []

    def prompt(text, default="", **kwargs):
        del kwargs
        prompts.append(text)
        return {"Method name": "Ensemble model"}.get(text, default)

    def confirm(text, default=False):
        prompts.append(text)
        return default

    monkeypatch.setattr(client.typer, "prompt", prompt)
    monkeypatch.setattr(client.typer, "confirm", confirm)

    metadata = client._submission_metadata({"first_name": "Ada", "email": "ada@example.org"})

    assert metadata["method_name"] == "Ensemble model"
    assert metadata["authors"] == "Ada"
    assert "deterministic_run" not in metadata
    assert not any("deterministic" in text.lower() and "run" in text.lower() for text in prompts)

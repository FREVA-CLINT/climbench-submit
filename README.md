# climbench-submit

Public Python 3.11+ tools for submitting model predictions to ClimBench and exporting optional training metadata. This project does not depend on private climbench-core, xarray, NetCDF/Zarr libraries, W&B, or MLflow SDKs.

## Install

```bash
python -m pip install -e .
# Development
python -m pip install -e '.[test]'
```

The only direct runtime dependencies are Typer (prompts), HTTPX (authenticated uploads), PyYAML (metadata export), and Pydantic (the normalized metadata schema).

## Submit predictions

First register and verify your email address on the platform, then run:

```bash
export CLIMBENCH_PLATFORM_URL=https://climbench.example.org
climbench-submit
# Or select a server for one invocation:
climbench-submit --platform-url http://localhost:8000
```

The command logs in, lists active task versions, asks for model output and the same metadata as the web form, and confirms the upload. Credentials and short-lived bearer tokens stay in memory. The URL option takes precedence over the environment variable; the development default is `http://localhost:8000`.

Accepted prediction inputs are `.nc`/`.netcdf`, a ZIP containing one Zarr v2/v3 dataset, or a local `.zarr` directory. Directories are temporarily packaged with the standard library, leaving source files unchanged. When the platform stores artifacts in S3, the CLI uploads model output in parts straight to the bucket with presigned URLs (like the web form), so large files are not limited by the platform's proxy; otherwise it falls back to a multipart upload to the platform. Compression tasks additionally require the compressed artifact file.

The platform validates declared task requirements before reporting success. It stamps its private prediction copy with the submission UUID and selected task identity. Admins evaluate the stored predictions and attach results; staff must separately accept those results before leaderboard publication. Exact target-coordinate comparisons and scientific data-value checks happen during admin evaluation.

Model output may be deterministic or a probabilistic ensemble with one extra dimension named `ensemble` on every required variable, e.g. `(time, ensemble, lat, lon)`, for any task. The CLI does not inspect the data: the platform detects the prediction type and sets the submission's `deterministic_run` accordingly, so there is no prompt for it.

The CLI prints validation errors or a receipt containing the upload UUID and detail URL. Update and deletion requests are available through the web interface.

## Export optional training metadata

```bash
climbench-metadata --source path/to/wandb/run-directory --output training.yaml
climbench-metadata --source path/to/mlflow/run-directory --source-kind mlflow_local --output training.yaml
```

Source kinds are `wandb_local` and `mlflow_local`. Detection fails clearly if both are found. No network access or tracker SDK is needed. The output must not already exist. The exporter preserves the extraction behavior previously provided by core.

Supply the resulting `.yaml`/`.yml` file when prompted by `climbench-submit`, or upload it in the browser. It is optional, stored with the submission, and is the sole source for the platform's hardware/training panel. It is not passed to evaluation. Manual YAML can use the same schema, for example:

```yaml
schema_version: '1.0'
source_kind: custom_yaml
framework: pytorch
gpu_count: 2
gpu_name: NVIDIA A100
gpu_memory_total_bytes: 85899345920
iterations_total: 10000
wallclock_seconds: 1200
```

Defaults: 5 GiB per prediction/compressed upload, 20 GiB expanded Zarr, and 1 MiB training YAML. Server administrators can configure different limits. Downloads and artifact locations are staff-only.

## Tests

```bash
python -m pytest
```

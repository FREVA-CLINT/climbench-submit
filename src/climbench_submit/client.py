from __future__ import annotations
import httpx
import typer


def _api_error(response: httpx.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        return f"Platform returned HTTP {response.status_code}."
    if isinstance(payload, dict):
        detail = payload.get("detail")
        if detail:
            return str(detail)
        errors = payload.get("errors")
        if isinstance(errors, dict):
            messages = []
            for field, field_errors in errors.items():
                if isinstance(field_errors, list):
                    for error in field_errors:
                        message = (
                            error.get("message") if isinstance(error, dict) else error
                        )
                        messages.append(f"{field}: {message}")
            if messages:
                return "; ".join(messages)
    return f"Platform returned HTTP {response.status_code}."


def _optional_prompt(label: str, default: str = "") -> str:
    return typer.prompt(
        f"{label} [OPTIONAL]", default=default, show_default=bool(default)
    ).strip()


def _submission_metadata(user: dict[str, object]) -> dict[str, str]:
    full_name = " ".join(
        value
        for value in (
            str(user.get("first_name") or "").strip(),
            str(user.get("last_name") or "").strip(),
        )
        if value
    )
    metadata = {
        "method_name": typer.prompt("Method name").strip(),
        "method_type": typer.prompt(
            "Method type", default="deterministic_baseline"
        ).strip(),
        "method_description": _optional_prompt("Method description"),
        "authors": _optional_prompt("Authors (comma-separated)", full_name),
        "team": _optional_prompt("Team"),
        "organization": _optional_prompt(
            "Organization", str(user.get("organization") or "")
        ),
        "contact": _optional_prompt("Contact email", str(user.get("email") or "")),
        "repository": _optional_prompt("Code repository"),
        "commit": _optional_prompt("Commit"),
        "license": _optional_prompt("License", "MIT"),
        "container_image": _optional_prompt("Container image"),
        "entrypoint": _optional_prompt("Entrypoint"),
        "publication_doi": _optional_prompt("Publication DOI"),
        "architecture": _optional_prompt("Architecture name"),
    }
    parameter_count = _optional_prompt("Parameter count (Million)")
    if parameter_count:
        try:
            if int(parameter_count) < 0:
                raise ValueError
        except ValueError as exc:
            raise typer.BadParameter(
                "Parameter count must be a non-negative integer."
            ) from exc
    metadata.update(
        {
            "parameter_count": parameter_count,
            "pretrained": "on"
            if typer.confirm("Is the model pretrained? [OPTIONAL]", default=False)
            else "",
            "leakage_statement": typer.prompt(
                "Leakage statement",
                default="No test-period target data or forbidden data were used.",
            ).strip(),
            "run_command": _optional_prompt("Run command"),
            "notes": _optional_prompt("Notes"),
        }
    )
    if (
        not metadata["method_name"]
        or not metadata["method_type"]
        or not metadata["leakage_statement"]
    ):
        raise typer.BadParameter(
            "Method name, method type, and leakage statement are required."
        )
    return metadata

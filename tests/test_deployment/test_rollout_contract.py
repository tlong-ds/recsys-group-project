from __future__ import annotations

from pathlib import Path

import yaml


_PIN_KEYS = (
    "RECSYS_DEPLOY_MODEL_NAME",
    "RECSYS_DEPLOY_MODEL_VERSION",
    "RECSYS_DEPLOY_RUN_ID",
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _env_values(container: dict) -> dict[str, list[str]]:
    values: dict[str, list[str]] = {}
    for item in container.get("env", []):
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        if not name:
            continue
        values.setdefault(str(name), []).append(str(item.get("value", "")))
    return values


def test_pinned_model_triplet_exists_exactly_once_in_api_and_init_containers() -> None:
    path = _repo_root() / "deployment/kubernetes/api-deployment.yaml"
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    pod_spec = payload["spec"]["template"]["spec"]

    init_container = next(
        container
        for container in pod_spec.get("initContainers", [])
        if container.get("name") == "model-downloader"
    )
    api_container = next(
        container
        for container in pod_spec.get("containers", [])
        if container.get("name") == "api"
    )

    init_env = _env_values(init_container)
    api_env = _env_values(api_container)

    for key in _PIN_KEYS:
        assert len(init_env.get(key, [])) == 1
        assert len(api_env.get(key, [])) == 1
        assert init_env[key][0] == api_env[key][0]


def test_prewarm_job_uses_same_pinned_model_triplet_placeholders() -> None:
    path = _repo_root() / "deployment/kubernetes/recsys-model-prewarm-job.yaml"
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    container = payload["spec"]["template"]["spec"]["containers"][0]
    env = _env_values(container)

    expected = {
        "RECSYS_DEPLOY_MODEL_NAME": "__RECSYS_DEPLOY_MODEL_NAME__",
        "RECSYS_DEPLOY_MODEL_VERSION": "__RECSYS_DEPLOY_MODEL_VERSION__",
        "RECSYS_DEPLOY_RUN_ID": "__RECSYS_DEPLOY_RUN_ID__",
    }
    for key, value in expected.items():
        assert env.get(key) == [value]


def test_deploy_workflow_consumes_promotion_result_contract() -> None:
    path = _repo_root() / ".github/workflows/deploy-eks.yml"
    workflow_text = path.read_text(encoding="utf-8")

    assert "metrics/promotion_result.json" in workflow_text
    assert "model_name={model_name}" in workflow_text
    assert "model_version={model_version}" in workflow_text
    assert "run_id={run_id}" in workflow_text

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


def test_api_container_has_startup_probe() -> None:
    path = _repo_root() / "deployment/kubernetes/api-deployment.yaml"
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    pod_spec = payload["spec"]["template"]["spec"]

    api_container = next(
        c for c in pod_spec.get("containers", []) if c.get("name") == "api"
    )

    startup_probe = api_container.get("startupProbe")
    assert startup_probe is not None, "api container must have a startupProbe"
    assert startup_probe["httpGet"]["path"] == "/ready"
    # Budget should allow at least 2 minutes for model loading
    budget_seconds = startup_probe.get("failureThreshold", 1) * startup_probe.get(
        "periodSeconds", 10
    )
    assert budget_seconds >= 120, f"startup budget {budget_seconds}s is too short"


def test_termination_grace_period_is_sufficient() -> None:
    path = _repo_root() / "deployment/kubernetes/api-deployment.yaml"
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    pod_spec = payload["spec"]["template"]["spec"]

    grace = pod_spec.get("terminationGracePeriodSeconds", 30)
    assert grace >= 30, f"terminationGracePeriodSeconds={grace} is too low"


def test_rolling_update_strategy_zero_unavailable() -> None:
    path = _repo_root() / "deployment/kubernetes/api-deployment.yaml"
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    strategy = payload["spec"].get("strategy", {})

    assert strategy.get("type") == "RollingUpdate"
    rolling = strategy.get("rollingUpdate", {})
    assert rolling.get("maxUnavailable") == 0


def test_hpa_min_replicas_at_least_two() -> None:
    path = _repo_root() / "deployment/kubernetes/api-hpa.yaml"
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))

    min_replicas = payload["spec"].get("minReplicas", 1)
    assert min_replicas >= 2, f"HPA minReplicas={min_replicas} is too low for HA"


def test_api_container_has_prestop_hook() -> None:
    path = _repo_root() / "deployment/kubernetes/api-deployment.yaml"
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    pod_spec = payload["spec"]["template"]["spec"]

    api_container = next(
        c for c in pod_spec.get("containers", []) if c.get("name") == "api"
    )

    lifecycle = api_container.get("lifecycle", {})
    pre_stop = lifecycle.get("preStop")
    assert pre_stop is not None, "api container must have a preStop lifecycle hook"

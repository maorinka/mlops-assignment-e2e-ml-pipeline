import hashlib
import json
import os
import shutil
import subprocess
import tarfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from airflow.sdk import Param, dag, get_current_context, task


PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parents[1])).expanduser()
RUNS_ROOT = Path(os.environ.get("RUNS_ROOT", PROJECT_ROOT / "runs")).expanduser()
UV_BIN = os.environ.get("UV_BIN") or shutil.which("uv") or str(Path.home() / ".local/bin" / "uv")
MINI_SWE_CONFIG = (
    Path(
        os.environ.get(
            "MINI_SWE_CONFIG",
            PROJECT_ROOT.parent
            / "mini-swe-agent"
            / "src"
            / "minisweagent"
            / "config"
            / "benchmarks"
            / "swebench.yaml",
        )
    )
    .expanduser()
    .resolve()
)
DATASETS = {
    "verified": "princeton-nlp/SWE-bench_Verified",
    "lite": "princeton-nlp/SWE-bench_Lite",
    "full": "princeton-nlp/SWE-bench",
}
SMALL_ARTIFACTS = ("config.json", "metrics.json", "manifest.json")


def load_dotenv(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def package_version(package: str) -> str:
    try:
        from importlib.metadata import PackageNotFoundError, version

        return version(package)
    except PackageNotFoundError:
        return "not-installed"


def git_sha() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError:
        return "unknown"


def build_run_config(params: dict[str, Any]) -> dict[str, Any]:
    task_slice = str(params["task_slice"])
    model = str(params["model"])
    run_id = str(params.get("run_id") or "").strip()
    if not run_id:
        run_id = (
            f"{time.strftime('%Y%m%d-%H%M%S')}-"
            f"{model.replace('/', '_')}-{task_slice.replace(':', '-')}"
        )
    return {
        "split": str(params["split"]),
        "subset": str(params["subset"]).lower(),
        "workers": int(params["workers"]),
        "model": model,
        "task_slice": task_slice,
        "run_id": run_id,
        "cost_limit": float(params["cost_limit"]),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_sha": git_sha(),
        "package_versions": {
            "mini-swe-agent": package_version("mini-swe-agent"),
            "swebench": package_version("swebench"),
        },
    }


def prepare_run_dir(run_config: dict[str, Any]) -> Path:
    run_dir = RUNS_ROOT / run_config["run_id"]
    (run_dir / "run-agent").mkdir(parents=True, exist_ok=True)
    (run_dir / "run-eval").mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(run_config, indent=2, sort_keys=True))
    return run_dir


def run_agent_batch(run_config: dict[str, Any]) -> Path:
    run_dir = RUNS_ROOT / run_config["run_id"]
    output_dir = run_dir / "run-agent"
    env = {
        **os.environ,
        **load_dotenv(PROJECT_ROOT / ".env"),
        "MSWEA_COST_TRACKING": "ignore_errors",
    }
    cmd = [
        UV_BIN,
        "run",
        "mini-extra",
        "swebench",
        "--subset",
        run_config["subset"],
        "--split",
        run_config["split"],
        "--model",
        run_config["model"],
        "--slice",
        run_config["task_slice"],
        "--workers",
        str(run_config["workers"]),
        "--cost-limit",
        str(run_config["cost_limit"]),
        "--config",
        str(MINI_SWE_CONFIG),
        "-o",
        str(output_dir),
    ]
    subprocess.run(cmd, cwd=PROJECT_ROOT, env=env, check=True)
    preds_path = output_dir / "preds.json"
    if not preds_path.exists() or preds_path.stat().st_size == 0:
        raise FileNotFoundError(f"mini-swe-agent did not write predictions: {preds_path}")
    preds = json.loads(preds_path.read_text())
    if not isinstance(preds, dict) or not preds:
        raise ValueError(f"preds.json is empty or malformed: {preds_path}")
    return preds_path


def run_swebench_eval(run_config: dict[str, Any], preds_path: str) -> Path:
    subset = run_config["subset"]
    if subset not in DATASETS:
        raise ValueError(f"Unsupported subset {subset!r}; expected one of {sorted(DATASETS)}")
    eval_dir = RUNS_ROOT / run_config["run_id"] / "run-eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        UV_BIN,
        "run",
        "python",
        "-m",
        "swebench.harness.run_evaluation",
        "--dataset_name",
        DATASETS[subset],
        "--predictions_path",
        str(Path(preds_path).resolve()),
        "--max_workers",
        str(run_config["workers"]),
        "--run_id",
        run_config["run_id"],
    ]
    subprocess.run(cmd, cwd=eval_dir, check=True)
    return eval_dir


def find_summary_report(eval_dir: Path, run_id: str) -> Path:
    matches = sorted(eval_dir.glob(f"*.{run_id}.json"))
    if not matches:
        raise FileNotFoundError(f"No SWE-bench summary report found in {eval_dir}")
    return matches[0]


def collect_metrics(run_config: dict[str, Any], eval_dir: str | Path) -> dict[str, Any]:
    eval_path = Path(eval_dir)
    summary_path = find_summary_report(eval_path, run_config["run_id"])
    summary = json.loads(summary_path.read_text())
    submitted = int(summary.get("submitted_instances", 0))
    resolved = int(summary.get("resolved_instances", 0))
    metrics = {
        "total_instances": int(summary.get("total_instances", 0)),
        "submitted_instances": submitted,
        "completed_instances": int(summary.get("completed_instances", 0)),
        "resolved_instances": resolved,
        "unresolved_instances": int(summary.get("unresolved_instances", 0)),
        "empty_patch_instances": int(summary.get("empty_patch_instances", 0)),
        "error_instances": int(summary.get("error_instances", 0)),
        "resolved_rate": resolved / submitted if submitted else 0.0,
    }
    created_at = datetime.fromisoformat(run_config["created_at"])
    metrics["duration_s"] = max(
        0.0,
        (datetime.now(timezone.utc) - created_at).total_seconds(),
    )
    return metrics


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest(run_config: dict[str, Any], metrics: dict[str, Any], s3_uri: str | None) -> dict[str, Any]:
    run_dir = RUNS_ROOT / run_config["run_id"]
    eval_dir = run_dir / "run-eval"
    summary_path = find_summary_report(eval_dir, run_config["run_id"])
    preds_path = run_dir / "run-agent" / "preds.json"
    required = [run_dir / "config.json", preds_path, run_dir / "metrics.json", summary_path]
    for path in required:
        if not path.exists() or path.stat().st_size == 0:
            raise FileNotFoundError(f"Required run artifact missing or empty: {path}")
    if not (eval_dir / "logs" / "run_evaluation" / run_config["run_id"]).exists():
        raise FileNotFoundError("SWE-bench evaluation logs are missing from the run tree")

    files = []
    for path in sorted(run_dir.rglob("*")):
        if path.is_file():
            files.append(
                {
                    "path": str(path.relative_to(run_dir)),
                    "size": path.stat().st_size,
                }
            )
    return {
        "run_id": run_config["run_id"],
        "run_dir": str(run_dir),
        "summary_report": str(summary_path.relative_to(run_dir)),
        "preds": {
            "path": str(preds_path.relative_to(run_dir)),
            "size": preds_path.stat().st_size,
            "sha256": sha256(preds_path),
        },
        "metrics": {
            "path": "metrics.json",
            "size": (run_dir / "metrics.json").stat().st_size,
            "sha256": sha256(run_dir / "metrics.json"),
            "values": metrics,
        },
        "s3_uri": s3_uri,
        "files": files,
    }


def make_run_archive(run_dir: Path) -> Path:
    archive_path = run_dir.with_suffix(".tar.gz")
    with tarfile.open(archive_path, "w:gz") as tar:
        tar.add(run_dir, arcname=run_dir.name)
    return archive_path


def upload_artifacts_if_configured(run_config: dict[str, Any]) -> str | None:
    env = {**os.environ, **load_dotenv(PROJECT_ROOT / ".env")}
    bucket = env.get("S3_BUCKET")
    endpoint = env.get("S3_ENDPOINT_URL")
    access_key = env.get("AWS_ACCESS_KEY_ID")
    secret_key = env.get("AWS_SECRET_ACCESS_KEY")
    if not all([bucket, endpoint, access_key, secret_key]):
        print("S3 upload skipped: S3_BUCKET/S3_ENDPOINT_URL/AWS credentials are not fully configured")
        return None

    import boto3

    run_dir = RUNS_ROOT / run_config["run_id"]
    archive_path = make_run_archive(run_dir)
    key = f"runs/{run_config['run_id']}/{archive_path.name}"
    client = boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
    )
    client.upload_file(str(archive_path), bucket, key)
    return f"s3://{bucket}/{key}"


def log_mlflow_run(run_config: dict[str, Any], metrics: dict[str, Any], manifest: dict[str, Any]) -> None:
    import mlflow

    run_dir = RUNS_ROOT / run_config["run_id"]
    eval_dir = run_dir / "run-eval"
    summary_path = find_summary_report(eval_dir, run_config["run_id"])
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    mlflow.set_experiment("swe-bench-eval")
    with mlflow.start_run(run_name=run_config["run_id"]):
        flat_params = {
            key: value
            for key, value in run_config.items()
            if isinstance(value, (str, int, float, bool))
        }
        mlflow.log_params(flat_params)
        mlflow.log_metrics({key: float(value) for key, value in metrics.items()})
        mlflow.set_tag("artifact_path", str(run_dir))
        if manifest.get("s3_uri"):
            mlflow.set_tag("s3_uri", manifest["s3_uri"])
        for name in SMALL_ARTIFACTS:
            mlflow.log_artifact(str(run_dir / name))
        mlflow.log_artifact(str(summary_path))


def summarize_run(run_config: dict[str, Any], eval_dir: str | Path) -> dict[str, Any]:
    run_dir = RUNS_ROOT / run_config["run_id"]
    metrics = collect_metrics(run_config, eval_dir)
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True))
    s3_uri = upload_artifacts_if_configured(run_config)
    manifest = build_manifest(run_config, metrics, s3_uri)
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    log_mlflow_run(run_config, metrics, manifest)
    return {"metrics": metrics, "manifest": manifest}


default_args = {"retries": 2, "retry_delay": timedelta(minutes=2)}


@dag(
    dag_id="evaluate_agent",
    start_date=datetime(2024, 1, 1),
    schedule=None,
    catchup=False,
    default_args=default_args,
    params={
        "split": Param("test", type="string"),
        "subset": Param("verified", type="string"),
        "workers": Param(4, type="integer", minimum=1),
        "model": Param("nebius/moonshotai/Kimi-K2.6", type="string"),
        "task_slice": Param("0:3", type="string"),
        "run_id": Param("", type="string"),
        "cost_limit": Param(0, type="number"),
    },
)
def evaluate_agent_dag():
    @task(execution_timeout=timedelta(minutes=5))
    def prepare_run() -> dict[str, Any]:
        context = get_current_context()
        run_config = build_run_config(dict(context["params"]))
        prepare_run_dir(run_config)
        return run_config

    @task(execution_timeout=timedelta(minutes=60))
    def run_agent(run_config: dict[str, Any]) -> str:
        return str(run_agent_batch(run_config))

    @task(execution_timeout=timedelta(minutes=60))
    def run_eval(run_config: dict[str, Any], preds_path: str) -> str:
        return str(run_swebench_eval(run_config, preds_path))

    @task(execution_timeout=timedelta(minutes=5))
    def summarize_and_log(run_config: dict[str, Any], eval_dir: str) -> dict[str, Any]:
        return summarize_run(run_config, eval_dir)

    config = prepare_run()
    predictions = run_agent(config)
    evaluation = run_eval(config, predictions)
    summarize_and_log(config, evaluation)


evaluate_agent_dag()

# Nebius Academy MLOps HW3 Report

## Architecture

The implemented Airflow DAG is `evaluate_agent`:

```text
prepare_run -> run_agent -> run_eval -> summarize_and_log
```

- `prepare_run` resolves runtime params, writes `runs/<run-id>/config.json`, and records git/package provenance.
- `run_agent` runs `mini-extra swebench` with Airflow params and writes trajectories plus `preds.json`.
- `run_eval` runs SWE-bench evaluation for the prediction instance ids and writes reports/logs under the same run tree.
- `summarize_and_log` parses the SWE-bench summary, writes `metrics.json` and `manifest.json`, optionally uploads artifacts to S3, and logs params/metrics/artifact references to MLflow.

This replaces script-chain failure modes with: Airflow params instead of hard-coded scripts, per-task retries instead of restarting from zero, a DAG instead of manual sequential execution, and MLflow plus a durable run tree instead of no experiment history.

## Triggering

Standalone path used on the VM:

```bash
bash run-airflow-standalone.sh
.venv/bin/mlflow server --host 0.0.0.0 --port 5000 \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root /home/maorhadad/mlops-assignment-e2e-ml-pipeline/mlflow-artifacts
```

Development run:

```json
{"task_slice":"0:1","workers":1,"run_id":"dev-20260708-211749"}
```

Final report run:

```json
{"task_slice":"0:3","workers":3,"run_id":"report-20260708-212813"}
```

`docker-compose.yaml` is included and validated with `docker compose config` on the VM. The verified execution path for this submission is Airflow standalone plus MLflow server.

## Artifacts

Every run writes:

```text
runs/<run-id>/
  config.json
  run-agent/
    preds.json
    <instance>/<instance>.traj.json
    minisweagent.log
    exit_statuses_*.yaml
  run-eval/
    logs/run_evaluation/<run-id>/<model-slug>/<instance>/
    <model-slug>.<run-id>.json
  metrics.json
  manifest.json
```

The committed sample run is `runs/report-20260708-212813/` with only small files committed: config, metrics, manifest, and the summary report. Full logs/trajectories stay ignored.

To reconstruct a run, read `config.json`, re-trigger the DAG with the same params, and compare outputs using `manifest.json` hashes and the SWE-bench summary report.

## Evidence

Airflow runs:

```text
dag_id         | run_id                 | state   | start_date                       | end_date
===============+========================+=========+==================================+=================================
evaluate_agent | report-20260708-212813 | success | 2026-07-08T21:28:15.960966+00:00 | 2026-07-08T21:34:53.622927+00:00
evaluate_agent | dev-20260708-211749    | success | 2026-07-08T21:17:51.413684+00:00 | 2026-07-08T21:26:45.886990+00:00
```

Final task states:

```text
prepare_run       success
run_agent         success
run_eval          success
summarize_and_log success
```

MLflow runs:

```text
params.run_id             params.task_slice params.workers metrics.resolved_rate metrics.resolved_instances metrics.submitted_instances
report-20260708-212813    0:3               3              0.666667              2.0                        3.0
dev-20260708-211749       0:1               1              1.000000              1.0                        1.0
```

Final SWE-bench summary:

```json
{
  "submitted_instances": 3,
  "completed_instances": 3,
  "resolved_instances": 2,
  "unresolved_instances": 1,
  "resolved_ids": ["astropy__astropy-12907", "astropy__astropy-13236"],
  "unresolved_ids": ["astropy__astropy-13033"]
}
```

S3/Object Storage: no `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `S3_ENDPOINT_URL`, or `S3_BUCKET` were configured in the VM `.env`, so upload skipped gracefully and `manifest.json` records `"s3_uri": null`.

Screenshots to capture later:

- `screenshots/airflow_dag.png`
- `screenshots/mlflow_runs.png`
- `screenshots/object_storage_artifacts.png` or CLI skip evidence

## Notes

The installed `mini-extra swebench` command in mini-swe-agent 2.4.1 does not support `--cost-limit`; the DAG records `cost_limit` in config but does not forward it to the batch command. SWE-bench evaluation requires `--instance_ids` for the prediction slice to avoid report generation over unrelated dataset rows.

Trajectory observation: the unresolved instance `astropy__astropy-13033` took 102 trajectory messages, compared with 22 for resolved `astropy__astropy-12907`; the unresolved case required substantially more agent/tool turns before producing a patch.

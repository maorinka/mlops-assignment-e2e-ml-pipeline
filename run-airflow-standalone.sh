set -euo pipefail

export AIRFLOW_HOME=~/airflow
export AIRFLOW__CORE__DAGS_FOLDER=$(pwd)/dags
export AIRFLOW__CORE__LOAD_EXAMPLES=false

mkdir -p $AIRFLOW_HOME

echo '{"admin": "admin"}' > $AIRFLOW_HOME/simple_auth_manager_passwords.json.generated

UV_BIN=${UV_BIN:-$(command -v uv || true)}
if [ -z "$UV_BIN" ] && [ -x "$HOME/.local/bin/uv" ]; then
  UV_BIN="$HOME/.local/bin/uv"
fi

"$UV_BIN" tool run apache-airflow standalone

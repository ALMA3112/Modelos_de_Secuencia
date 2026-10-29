#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/Modelos_de_Secuencia"
source .venv/bin/activate
exec mlflow server \
  --host 0.0.0.0 --port 5000 \
  --backend-store-uri "sqlite:///$HOME/mlflow/mlflow.db" \
  --artifacts-destination "$HOME/mlflow/artifacts" \
  --serve-artifacts \
  --allowed-hosts "*" \
  --cors-allowed-origins "*"

#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/Modelos_de_Secuencia"
source .venv/bin/activate
export MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-http://127.0.0.1:5000}"
python scripts/run_experiment.py --name E1 --arch lstm
python scripts/run_experiment.py --name E2 --arch gru
python scripts/run_experiment.py --name E3 --arch gru_attn

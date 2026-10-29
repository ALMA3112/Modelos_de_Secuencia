import json
import sys
from pathlib import Path

import mlflow
import mlflow.pytorch
from mlflow import MlflowClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bach.train import CKPT_DIR, load_checkpoint

NAME = "bach-chorales-alma"
mlflow.set_tracking_uri("http://127.0.0.1:5000")

best = json.loads((ROOT / "artifacts" / "selected.json").read_text())
model, ck = load_checkpoint(CKPT_DIR / f"{best['name']}.pt")
assert ck["epoch"] == best["best_epoch"]

with mlflow.start_run(run_id=best["run_id"]):
    info = mlflow.pytorch.log_model(
        model, name="model", registered_model_name=NAME,
        code_paths=[str(ROOT / "bach")], serialization_format="pickle",
        pip_requirements=["torch", "numpy", "pandas", "mlflow"])

c = MlflowClient()
v = max(c.search_model_versions(f"name='{NAME}'"), key=lambda x: int(x.version)).version
c.set_registered_model_alias(NAME, "champion", v)
res = {"model": NAME, "version": str(v), "alias": "champion",
       "run_id": best["run_id"], "source_run": best["name"], "epoch": ck["epoch"]}
(ROOT / "artifacts" / "registered_model.json").write_text(json.dumps(res, indent=2))
print(json.dumps(res, indent=2))

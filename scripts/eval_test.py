import json
import sys
from pathlib import Path

import mlflow
from mlflow import MlflowClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bach.data import get_loader
from bach.metrics import evaluate
from bach.train import CKPT_DIR, load_checkpoint

mlflow.set_tracking_uri("http://127.0.0.1:5000")
out = ROOT / "artifacts" / "test_result.json"
if out.exists():
    sys.exit("test ya se evaluo: no se repite.")

best = json.loads((ROOT / "artifacts" / "selected.json").read_text())
model, ck = load_checkpoint(CKPT_DIR / f"{best['name']}.pt")
assert ck["epoch"] == best["best_epoch"]
t = evaluate(model, get_loader("test", 64))
res = {"selected": best["name"], "run_id": best["run_id"], "epoch": ck["epoch"],
       "val_loss": best["val_loss"], "test_loss": t["loss"],
       "test_note_acc": t["note_acc"], "test_step_acc": t["step_acc"]}
out.write_text(json.dumps(res, indent=2))
print(json.dumps(res, indent=2))

c = MlflowClient()
for k in ("test_loss", "test_note_acc", "test_step_acc"):
    c.log_metric(best["run_id"], k, res[k])
c.set_tag(best["run_id"], "selected", "true")
c.log_artifact(best["run_id"], str(out))
print("Registrado en el run", best["name"])

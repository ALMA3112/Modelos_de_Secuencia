import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bach.train import DEFAULTS, train_run

p = argparse.ArgumentParser()
p.add_argument("--name", required=True, help="E1, E2, E3, E4...")
p.add_argument("--arch", required=True, choices=["lstm", "gru", "gru_attn"])
p.add_argument("--hidden", type=int, default=DEFAULTS["hidden"])
p.add_argument("--emb", type=int, default=DEFAULTS["emb"])
p.add_argument("--layers", type=int, default=DEFAULTS["layers"])
p.add_argument("--lr", type=float, default=DEFAULTS["lr"])
p.add_argument("--batch", type=int, default=DEFAULTS["batch"])
p.add_argument("--max-epochs", type=int, default=DEFAULTS["max_epochs"])
p.add_argument("--patience", type=int, default=DEFAULTS["patience"])
p.add_argument("--seed", type=int, default=DEFAULTS["seed"])
p.add_argument("--experiment", default=os.environ.get("EXPERIMENT", "nlp-bach-equipo01"))
p.add_argument("--tracking-uri", default=os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
a = p.parse_args()

cfg = dict(arch=a.arch, hidden=a.hidden, emb=a.emb, layers=a.layers, lr=a.lr,
           batch=a.batch, max_epochs=a.max_epochs, patience=a.patience, seed=a.seed)
train_run(a.name, cfg, a.experiment, a.tracking_uri)

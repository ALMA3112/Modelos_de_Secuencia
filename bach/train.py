"""Entrenamiento con early stopping, mejor checkpoint y registro en MLflow."""
import json
import os
import platform
import random
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import torch
import torch.nn.functional as F

from bach.data import CONTEXT, HORIZON, get_loader
from bach.metrics import evaluate
from bach.models import BachSeq2Seq, build_model, count_parameters

ROOT = Path(__file__).resolve().parents[1]
CKPT_DIR = ROOT / "artifacts" / "ckpt"
RES_DIR = ROOT / "artifacts" / "results"

DEFAULTS = dict(arch="gru", hidden=128, emb=16, layers=1, lr=1e-3,
                batch=64, max_epochs=20, patience=4, seed=42)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_checkpoint(path, device="cpu"):
    ck = torch.load(path, map_location=device, weights_only=True)
    model = BachSeq2Seq.from_config(ck["config"])
    model.load_state_dict(ck["state_dict"])
    return model.to(device).eval(), ck


def train_run(name, cfg, experiment, tracking_uri):
    cfg = {**DEFAULTS, **cfg}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(cfg["seed"])                                   # reinicio al comenzar el run
    train_dl = get_loader("train", cfg["batch"], cfg["seed"])
    val_dl = get_loader("valid", cfg["batch"], cfg["seed"])
    model = build_model(cfg["arch"], cfg["hidden"], cfg["emb"], cfg["layers"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    n_params = count_parameters(model)

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    RES_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = CKPT_DIR / f"{name}.pt"

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=name) as run:
        mlflow.set_tag("run_id_lab", name)
        mlflow.log_params({
            "architecture": cfg["arch"], "attention": model.attention,
            "layers": cfg["layers"], "hidden_dim": cfg["hidden"],
            "embedding_dim": cfg["emb"], "learning_rate": cfg["lr"],
            "batch_size": cfg["batch"], "max_epochs": cfg["max_epochs"],
            "patience": cfg["patience"], "seed": cfg["seed"],
            "context": CONTEXT, "horizon": HORIZON, "optimizer": "Adam",
            "n_trainable_params": n_params})

        best, best_epoch, bad = float("inf"), 0, 0
        hist = {"train": [], "val": []}
        t0 = time.time()
        for epoch in range(1, cfg["max_epochs"] + 1):
            model.train()
            tot, n = 0.0, 0
            for x, y in train_dl:
                x, y = x.to(device), y.to(device)
                opt.zero_grad()
                logits = model(x, y)                        # (B,16,4,128), sin softmax
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
                loss.backward()
                opt.step()
                tot += loss.item() * x.size(0)
                n += x.size(0)
            tr = tot / n
            va = evaluate(model, val_dl, device)["loss"]
            hist["train"].append(tr)
            hist["val"].append(va)
            mlflow.log_metric("train_loss", tr, step=epoch)
            mlflow.log_metric("val_loss", va, step=epoch)
            improved = va < best
            if improved:
                best, best_epoch, bad = va, epoch, 0
                torch.save({"state_dict": model.state_dict(), "config": model.get_config(),
                            "epoch": epoch, "val_loss": va}, ckpt_path)
            else:
                bad += 1
            print(f"[{name}] ep {epoch:2d} train {tr:.4f} val {va:.4f}"
                  f"{' *' if improved else ''}", flush=True)
            if bad >= cfg["patience"]:
                print(f"[{name}] early stopping en la epoca {epoch}", flush=True)
                break
        train_time = time.time() - t0

        # Recargar el MEJOR checkpoint y medir todo con esos pesos
        best_model, ck = load_checkpoint(ckpt_path, device)
        final = evaluate(best_model, val_dl, device)
        assert abs(final["loss"] - best) < 1e-4, "el checkpoint no reproduce la mejor perdida"
        assert ck["epoch"] == best_epoch

        mlflow.log_metrics({
            "best_epoch": best_epoch, "best_val_loss": final["loss"],
            "val_note_acc": final["note_acc"], "val_step_acc": final["step_acc"],
            "epochs_run": len(hist["val"]), "train_time_s": train_time})

        # Artefactos
        art = ROOT / "artifacts" / "results" / name
        art.mkdir(parents=True, exist_ok=True)
        (art / "config.json").write_text(json.dumps(
            {**cfg, **model.get_config(), "n_trainable_params": n_params}, indent=2))
        compute = {"device": str(device), "platform": platform.platform(),
                   "processor": platform.processor(), "cpu_count": os.cpu_count(),
                   "torch": torch.__version__, "python": platform.python_version(),
                   "mlflow": mlflow.__version__}
        (art / "compute.json").write_text(json.dumps(compute, indent=2))
        (art / "n_params.txt").write_text(str(n_params))
        fig, ax = plt.subplots(figsize=(6, 4))
        ep = range(1, len(hist["val"]) + 1)
        ax.plot(ep, hist["train"], label="train")
        ax.plot(ep, hist["val"], label="valid")
        ax.axvline(best_epoch, ls="--", c="gray", label=f"mejor epoca ({best_epoch})")
        ax.set_xlabel("epoca"); ax.set_ylabel("entropia cruzada"); ax.set_title(name)
        ax.legend(); fig.tight_layout()
        fig.savefig(art / "loss_curve.png", dpi=130); plt.close(fig)
        for f in ["config.json", "compute.json", "n_params.txt", "loss_curve.png"]:
            mlflow.log_artifact(str(art / f))
        mlflow.log_artifact(str(ckpt_path), artifact_path="checkpoint")

        summary = {"name": name, "run_id": run.info.run_id, "experiment": experiment,
                   "arch": cfg["arch"], "hidden": cfg["hidden"], "lr": cfg["lr"],
                   "batch": cfg["batch"], "best_epoch": best_epoch,
                   "epochs_run": len(hist["val"]), "val_loss": final["loss"],
                   "val_note_acc": final["note_acc"], "val_step_acc": final["step_acc"],
                   "n_params": n_params, "train_time_s": round(train_time, 1),
                   "device": str(device)}
        (RES_DIR / f"{name}.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary, indent=2))
        return summary

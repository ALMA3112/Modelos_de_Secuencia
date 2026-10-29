import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import torch
from mlflow import MlflowClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bach.data import load_chorales
from bach.train import CKPT_DIR, load_checkpoint

SEED, TEMP = 123, 1.0
OUT = ROOT / "artifacts" / "generation"
OUT.mkdir(parents=True, exist_ok=True)
mlflow.set_tracking_uri("http://127.0.0.1:5000")

e3 = json.loads((ROOT / "artifacts" / "results" / "E3.json").read_text())
model, ck = load_checkpoint(CKPT_DIR / "E3.pt")
name, arr = load_chorales("valid")[0]
ctx = torch.from_numpy(arr[:32]).long()
gen, attn = model.generate(ctx, 32, TEMP, torch.Generator().manual_seed(SEED),
                           return_attention=True)
A = attn[0].numpy()                      # (16 pasos generados, 32 pasos de contexto)
assert A.shape == (16, 32) and np.allclose(A.sum(1), 1, atol=1e-4)

fig, ax = plt.subplots(figsize=(9, 5))
im = ax.imshow(A, aspect="auto", cmap="viridis")
ax.set_xticks(range(0, 32, 2)); ax.set_xticklabels(range(1, 33, 2))
ax.set_yticks(range(16)); ax.set_yticklabels(range(1, 17))
ax.set_xlabel("paso de contexto (1 = mas antiguo, 32 = ultimo)")
ax.set_ylabel("paso generado (1..16)")
ax.set_title(f"Atencion E3 (epoca {ck['epoch']}), {name}, T={TEMP}, semilla {SEED}")
fig.colorbar(im, label="peso de atencion"); fig.tight_layout()
png = OUT / "attention_E3.png"
fig.savefig(png, dpi=140); plt.close(fig)

resumen = {"coral": name, "epoca_checkpoint": ck["epoch"], "semilla": SEED, "temperatura": TEMP,
           "posicion_maxima_por_paso_generado": (A.argmax(1) + 1).tolist(),
           "peso_medio_ultimos_4_pasos_de_contexto": round(float(A[:, -4:].sum(1).mean()), 3),
           "peso_medio_primeros_16_pasos_de_contexto": round(float(A[:, :16].sum(1).mean()), 3),
           "cinco_posiciones_con_mas_peso_medio": (np.argsort(-A.mean(0))[:5] + 1).tolist()}
(OUT / "attention_E3.json").write_text(json.dumps(resumen, indent=2))
print(json.dumps(resumen, indent=2))
c = MlflowClient()
for f in (png, OUT / "attention_E3.json"):
    c.log_artifact(e3["run_id"], str(f), artifact_path="atencion")
print("Registrado en el run E3")

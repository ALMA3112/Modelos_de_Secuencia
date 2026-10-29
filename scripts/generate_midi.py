import json
import sys
from pathlib import Path

import numpy as np
import torch
from mlflow import MlflowClient
import mlflow

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bach.data import load_chorales
from bach.midi import step_seconds, to_midi, voice_runs
from bach.train import CKPT_DIR, load_checkpoint

TEMPO, TEMP, SEED, STEPS = 120, 1.0, 123, 32
OUT = ROOT / "artifacts" / "generation"
OUT.mkdir(parents=True, exist_ok=True)
mlflow.set_tracking_uri("http://127.0.0.1:5000")


def stats(a):
    a = np.asarray(a)
    pos = a[a > 0]
    return {"pasos": int(len(a)),
            "silencios": int((a == 0).sum()),
            "notas_distintas_por_voz": [len(set(a[:, v].tolist())) for v in range(4)],
            "frac_paso_igual_al_anterior": round(float((a[1:] == a[:-1]).all(1).mean()), 3),
            "frac_nota_igual_a_la_anterior": round(float((a[1:] == a[:-1]).mean()), 3),
            "rango_notas": [int(pos.min()), int(pos.max())] if len(pos) else [0, 0],
            "notas_midi_tras_fusion": sum(len(voice_runs(a[:, v])) for v in range(4))}


best = json.loads((ROOT / "artifacts" / "selected.json").read_text())
model, ck = load_checkpoint(CKPT_DIR / f"{best['name']}.pt")
chorales = load_chorales("valid")
info = {"modelo": best["name"], "run_id": best["run_id"], "epoca_checkpoint": ck["epoch"],
        "semilla_generacion": SEED, "temperatura": TEMP, "tempo_negras_por_minuto": TEMPO,
        "segundos_por_paso": step_seconds(TEMPO), "pasos_generados": STEPS,
        "fragmentos": []}
files = []
for k, idx in enumerate([0, 1], 1):
    name, arr = chorales[idx]
    gen = model.generate(torch.from_numpy(arr[:32]).long(), STEPS, TEMP,
                         torch.Generator().manual_seed(SEED)).numpy()
    assert gen.shape == (STEPS, 4) and gen.min() >= 0 and gen.max() < 128
    path = OUT / f"coral{k}_{name[:-4]}.mid"
    to_midi(np.concatenate([arr[:32], gen]), TEMPO, path)
    files.append(path)
    info["fragmentos"].append({"coral": name, "archivo_midi": path.name,
                               "contexto": stats(arr[:32]), "generado": stats(gen),
                               "continuacion_real": stats(arr[32:32 + STEPS])})
    print(name, "generado:", json.dumps(info["fragmentos"][-1]["generado"]))
    print(name, "real:    ", json.dumps(info["fragmentos"][-1]["continuacion_real"]))

(OUT / "info.json").write_text(json.dumps(info, indent=2))
c = MlflowClient()
for f in files + [OUT / "info.json"]:
    c.log_artifact(best["run_id"], str(f), artifact_path="generacion")
print("Listo:", [f.name for f in files])

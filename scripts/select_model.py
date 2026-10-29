import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "artifacts" / "results"

rows = [json.loads(p.read_text()) for p in sorted(RES.glob("E*.json"))]
rows.sort(key=lambda r: r["val_loss"])
best = rows[0]
print("Modelo elegido:", best["name"], best["arch"], round(best["val_loss"], 4))

L = ["| Run | Arq. | Hidden | LR | Mejor epoca | Val loss | Acc nota | Acc paso | Params | Tiempo (s) |",
     "|---|---|---|---|---|---|---|---|---|---|"]
for r in sorted(rows, key=lambda r: r["name"]):
    L.append(f"| {r['name']} | {r['arch']} | {r['hidden']} | {r['lr']} | "
             f"{r['best_epoch']}/{r['epochs_run']} | {r['val_loss']:.4f} | "
             f"{r['val_note_acc']:.4f} | {r['val_step_acc']:.4f} | "
             f"{r['n_params']:,} | {r['train_time_s']} |")
table = "\n".join(L)
(ROOT / "artifacts" / "results_table.md").write_text(table + "\n")
(ROOT / "artifacts" / "selected.json").write_text(json.dumps(best, indent=2))
print(table)

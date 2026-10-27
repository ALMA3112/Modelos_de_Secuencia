import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bach.data import build_windows

report = build_windows()
out = ROOT / "artifacts" / "data_report.json"
out.write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
print("Guardado:", out)

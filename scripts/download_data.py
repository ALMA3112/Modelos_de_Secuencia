import sys
import tarfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
TGZ = RAW / "jsb_chorales.tgz"
URLS = [
    "https://github.com/ageron/handson-ml2/raw/master/datasets/jsb_chorales/jsb_chorales.tgz",
    "https://github.com/ageron/data/raw/main/jsb_chorales.tgz",
]

RAW.mkdir(parents=True, exist_ok=True)
if not TGZ.exists():
    for url in URLS:
        try:
            print("Descargando", url)
            urllib.request.urlretrieve(url, TGZ)
            break
        except Exception as e:
            print("Fallo:", e)
    else:
        sys.exit("No se pudo descargar. Revisa la URL en el repositorio de Geron.")

with tarfile.open(TGZ) as t:
    t.extractall(RAW, filter="data")
print("Listo en", RAW)

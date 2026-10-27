"""Carga de JSB Chorales, ventanas 32->16 y DataLoaders."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

VOICES = ["note0", "note1", "note2", "note3"]
CONTEXT = 32
HORIZON = 16
WINDOW = CONTEXT + HORIZON
VOCAB = 128
STRIDES = {"train": 8, "valid": 16, "test": 16}
EXPECTED = {"train": 229, "valid": 76, "test": 77}

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
WIN_PATH = ROOT / "data" / "windows.npz"
META_PATH = ROOT / "data" / "windows_meta.json"


def _split_dir(raw_dir, split):
    for p in sorted(Path(raw_dir).rglob(split)):
        if p.is_dir() and any(p.glob("*.csv")):
            return p
    raise FileNotFoundError(f"No se encontro la particion '{split}' en {raw_dir}")


def load_chorales(split, raw_dir=RAW_DIR):
    """Devuelve [(nombre, array int64 de forma (L,4))] ordenado por nombre."""
    out = []
    for f in sorted(_split_dir(raw_dir, split).glob("*.csv")):
        df = pd.read_csv(f)
        assert list(df.columns) == VOICES, f"{f.name}: columnas {list(df.columns)}"
        out.append((f.name, df[VOICES].to_numpy(dtype=np.int64)))
    return out


def make_windows(chorales, stride):
    """Ventanas completas de 48 pasos dentro de cada coral (nunca cruzan corales)."""
    X, Y, cid, start = [], [], [], []
    for i, (_, arr) in enumerate(chorales):
        for t in range(0, len(arr) - WINDOW + 1, stride):
            X.append(arr[t:t + CONTEXT])
            Y.append(arr[t + CONTEXT:t + WINDOW])
            cid.append(i)
            start.append(t)
    return (np.stack(X).astype(np.int64), np.stack(Y).astype(np.int64),
            np.array(cid, dtype=np.int64), np.array(start, dtype=np.int64))


def build_windows(raw_dir=RAW_DIR, out_path=WIN_PATH, meta_path=META_PATH):
    """Genera las ventanas UNA vez y las guarda en .npz (+ nombres de corales en json)."""
    arrays, meta, report = {}, {}, {}
    for split, stride in STRIDES.items():
        ch = load_chorales(split, raw_dir)
        X, Y, cid, start = make_windows(ch, stride)
        arrays.update({f"{split}_X": X, f"{split}_Y": Y,
                       f"{split}_cid": cid, f"{split}_start": start})
        meta[split] = [name for name, _ in ch]
        lens = [len(a) for _, a in ch]
        allv = np.concatenate([a for _, a in ch])
        report[split] = {
            "corales": len(ch), "esperados": EXPECTED[split],
            "len_min": int(min(lens)), "len_max": int(max(lens)),
            "desplazamiento": stride, "ventanas": int(len(X)),
            "valor_min_por_voz": allv.min(axis=0).tolist(),
            "valor_max_por_voz": allv.max(axis=0).tolist(),
            "fraccion_silencio_por_voz": (allv == 0).mean(axis=0).round(4).tolist(),
        }
    np.savez_compressed(out_path, **arrays)
    Path(meta_path).write_text(json.dumps(meta, indent=1))
    return report


class BachWindows(Dataset):
    def __init__(self, split, path=WIN_PATH):
        z = np.load(path)
        self.X = torch.from_numpy(z[f"{split}_X"]).long()
        self.Y = torch.from_numpy(z[f"{split}_Y"]).long()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.Y[i]


def get_loader(split, batch_size=64, seed=42, path=WIN_PATH):
    """Lotes (B,32,4) y (B,16,4). Solo train se baraja, con generador sembrado."""
    g = torch.Generator().manual_seed(seed)
    return DataLoader(BachWindows(split, path), batch_size=batch_size,
                      shuffle=(split == "train"), generator=g, drop_last=False)

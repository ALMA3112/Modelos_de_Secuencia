import numpy as np
import pytest
import torch

from bach.data import (CONTEXT, EXPECTED, HORIZON, STRIDES, VOCAB, WINDOW,
                       WIN_PATH, get_loader, load_chorales)


@pytest.mark.parametrize("split", ["train", "valid", "test"])
def test_particiones_y_ventanas(split):
    ch = load_chorales(split)
    assert len(ch) == EXPECTED[split]
    z = np.load(WIN_PATH)
    X, Y = z[f"{split}_X"], z[f"{split}_Y"]
    cid, st = z[f"{split}_cid"], z[f"{split}_start"]
    assert X.shape[1:] == (CONTEXT, 4) and Y.shape[1:] == (HORIZON, 4)
    assert X.min() >= 0 and max(X.max(), Y.max()) < VOCAB
    # cada ventana es un trozo contiguo de UN solo coral
    for k in np.random.default_rng(0).choice(len(X), 200, replace=False):
        arr = ch[cid[k]][1]
        assert (arr[st[k]:st[k] + CONTEXT] == X[k]).all()
        assert (arr[st[k] + CONTEXT:st[k] + WINDOW] == Y[k]).all()
    # conteo exacto esperado: sin fragmentos incompletos
    esperado = sum(max(0, (len(a) - WINDOW) // STRIDES[split] + 1) for _, a in ch)
    assert len(X) == esperado
    assert (st % STRIDES[split] == 0).all()


def test_lotes():
    xb, yb = next(iter(get_loader("train", 64)))
    assert xb.shape == (64, 32, 4) and yb.shape == (64, 16, 4)
    assert xb.dtype == torch.long and yb.dtype == torch.long


def test_silencio_presente():
    assert (np.load(WIN_PATH)["train_X"] == 0).any()

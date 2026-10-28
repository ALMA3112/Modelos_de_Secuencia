import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from bach.metrics import evaluate


class Fake(nn.Module):
    def __init__(self, mode):
        super().__init__()
        self.mode = mode

    def forward(self, x, y):
        if self.mode == "perfect":
            return F.one_hot(y, 128).float() * 50
        return torch.zeros(*y.shape, 128)       # argmax = 0 (silencio)


def batches():
    g = torch.Generator().manual_seed(0)
    out = []
    for b in (3, 5):                             # tamanos distintos: prueba la acumulacion
        y = torch.randint(0, 3, (b, 16, 4), generator=g)
        out.append((torch.zeros(b, 32, 4, dtype=torch.long), y))
    return out


def test_prediccion_perfecta():
    r = evaluate(Fake("perfect"), batches())
    assert r["loss"] < 1e-6 and r["note_acc"] == 1.0 and r["step_acc"] == 1.0


def test_uniforme_cuenta_el_silencio():
    data = batches()
    r = evaluate(Fake("zeros"), data)
    y = torch.cat([b[1] for b in data])
    assert math.isclose(r["loss"], math.log(128), rel_tol=1e-6)
    assert math.isclose(r["note_acc"], (y == 0).float().mean().item(), rel_tol=1e-9)
    assert math.isclose(r["step_acc"], (y == 0).all(-1).float().mean().item(), rel_tol=1e-9)

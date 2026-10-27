import pickle

import pytest
import torch
import torch.nn.functional as F

from bach.models import BachSeq2Seq, build_model, count_parameters

ARCHS = ["lstm", "gru", "gru_attn"]
PARAMS = {"lstm": 266752, "gru": 217088, "gru_attn": 364800}


def batch(B=5, seed=0):
    g = torch.Generator().manual_seed(seed)
    return (torch.randint(0, 128, (B, 32, 4), generator=g),
            torch.randint(0, 128, (B, 16, 4), generator=g))


@pytest.mark.parametrize("arch", ARCHS)
def test_formas_y_parametros(arch):
    m = build_model(arch)
    x, y = batch()
    assert m(x, y).shape == (5, 16, 4, 128)
    assert count_parameters(m) == PARAMS[arch]


def test_atencion_E3():
    x, y = batch()
    logits, attn = build_model("gru_attn")(x, y, return_attention=True)
    assert attn.shape == (5, 16, 32)
    assert torch.allclose(attn.sum(-1), torch.ones(5, 16), atol=1e-5)
    assert build_model("gru")(x, y, return_attention=True)[1] is None


def test_lstm_transfiere_h_y_c():
    m = build_model("lstm").eval()
    x, _ = batch()
    enc_out, (h, c) = m.encode(x)
    assert h.shape == (1, 5, 128) and c.shape == (1, 5, 128)
    l1, _, _ = m._step(x[:, -1], (h, c), enc_out)
    l2, _, _ = m._step(x[:, -1], (h, torch.zeros_like(c)), enc_out)
    assert not torch.allclose(l1, l2)  # el estado de memoria c importa


@pytest.mark.parametrize("arch", ARCHS)
def test_teacher_forcing_sin_fuga(arch):
    m = build_model(arch).eval()
    x, y = batch()
    base = m(x, y)
    y2 = y.clone(); y2[:, 15] = (y2[:, 15] + 1) % 128
    assert torch.allclose(base, m(x, y2), atol=1e-6)        # Y[15] nunca es entrada
    y3 = y.clone(); y3[:, 0] = (y3[:, 0] + 1) % 128
    out3 = m(x, y3)
    assert torch.allclose(base[:, 0], out3[:, 0], atol=1e-6)  # paso 0 solo usa x[:, -1]
    assert not torch.allclose(base[:, 1:], out3[:, 1:])


@pytest.mark.parametrize("arch", ARCHS)
def test_paso_a_paso_igual_a_forward(arch):
    m = build_model(arch).eval()
    x, y = batch()
    with torch.no_grad():
        ref = m(x, y)
        enc_out, state = m.encode(x)
        dec_in = torch.cat([x[:, -1:], y[:, :-1]], 1)
        outs = []
        for k in range(16):
            lg, state, _ = m._step(dec_in[:, k], state, enc_out)
            outs.append(lg)
    assert torch.allclose(ref, torch.stack(outs, 1), atol=1e-5)


@pytest.mark.parametrize("arch", ARCHS)
@pytest.mark.parametrize("steps", [1, 16, 17, 33, 64])
def test_generate_formas(arch, steps):
    m = build_model(arch)
    x, _ = batch(2)
    out = m.generate(x, steps, 1.0, torch.Generator().manual_seed(1))
    assert out.shape == (2, steps, 4) and out.dtype == torch.long
    assert out.min() >= 0 and out.max() < 128


def test_generate_reproducible():
    m = build_model("gru")
    x, _ = batch(1)
    a = m.generate(x, 32, 1.0, torch.Generator().manual_seed(7))
    b = m.generate(x, 32, 1.0, torch.Generator().manual_seed(7))
    c = m.generate(x, 32, 1.0, torch.Generator().manual_seed(8))
    assert torch.equal(a, b) and not torch.equal(a, c)


def test_bloques_reinicializan_con_ultimos_32():
    m = build_model("gru_attn")
    x, _ = batch(1)
    full = m.generate(x, 32, 1.0, torch.Generator().manual_seed(3))
    g = torch.Generator().manual_seed(3)
    b1 = m.generate(x, 16, 1.0, g)
    ctx2 = torch.cat([x, b1], 1)[:, -32:]
    b2 = m.generate(ctx2, 16, 1.0, g)
    assert torch.equal(full, torch.cat([b1, b2], 1))


def test_generate_restaura_modo_y_no_modifica_contexto():
    m = build_model("gru").train()
    x, _ = batch(1)
    x0 = x.clone()
    m.generate(x, 5)
    assert m.training and torch.equal(x, x0)


def test_generate_atencion():
    m = build_model("gru_attn")
    x, _ = batch(1)
    gen, attn = m.generate(x, 20, return_attention=True)
    assert len(attn) == 2 and attn[0].shape == (1, 16, 32) and attn[1].shape == (1, 4, 32)


def test_argumentos_invalidos():
    m = build_model("gru")
    x, _ = batch(1)
    with pytest.raises(ValueError):
        m.generate(x, 5, temperature=0)
    with pytest.raises(ValueError):
        m.generate(x, 0)
    with pytest.raises(ValueError):
        m.generate(x[:, :10], 5)


@pytest.mark.parametrize("arch", ARCHS)
def test_pickle(arch):
    m = build_model(arch)
    x, _ = batch(1)
    m2 = pickle.loads(pickle.dumps(m))
    a = m.generate(x, 16, 1.0, torch.Generator().manual_seed(0))
    b = m2.generate(x, 16, 1.0, torch.Generator().manual_seed(0))
    assert torch.equal(a, b)


@pytest.mark.parametrize("arch", ARCHS)
def test_perdida_y_gradientes(arch):
    torch.manual_seed(0)
    m = build_model(arch)
    x, y = batch(8)
    loss = F.cross_entropy(m(x, y).reshape(-1, 128), y.reshape(-1))
    assert 4.5 < loss.item() < 5.3  # ~ ln(128) = 4.85 al inicio
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())


def test_config_roundtrip():
    m = build_model("gru_attn", hidden=64)
    m2 = BachSeq2Seq.from_config(m.get_config())
    assert count_parameters(m) == count_parameters(m2)

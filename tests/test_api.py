import pytest
from fastapi.testclient import TestClient

from api.main import app
from bach.data import load_chorales

ROW = [60, 55, 52, 48]
SEED = [ROW] * 32


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    d = r.json()
    assert d["status"] == "ok" and d["model"] == "bach-chorales-alma"
    assert isinstance(d["version"], str)


def test_solicitud_valida(client):
    name, arr = load_chorales("valid")[0]
    r = client.post("/generate", json={"seed": arr[:32].tolist(), "steps": 32,
                                       "temperature": 1.0})
    assert r.status_code == 200
    d = r.json()
    assert len(d["generated"]) == 32 and all(len(f) == 4 for f in d["generated"])
    assert all(0 <= n <= 127 for f in d["generated"] for n in f)
    assert d["model"] == "bach-chorales-alma" and isinstance(d["version"], str)


def test_steps_33_no_multiplo_de_16(client):
    r = client.post("/generate", json={"seed": SEED, "steps": 33})
    assert r.status_code == 200 and len(r.json()["generated"]) == 33


def test_temperatura_por_defecto(client):
    assert client.post("/generate", json={"seed": SEED, "steps": 1}).status_code == 200


@pytest.mark.parametrize("payload", [
    {"seed": [ROW] * 31, "steps": 5},
    {"seed": [ROW] * 33, "steps": 5},
    {"seed": [[60, 55, 52]] * 32, "steps": 5},
    {"seed": [[60, 55, 52, 128]] * 32, "steps": 5},
    {"seed": [[60, 55, 52, -1]] * 32, "steps": 5},
    {"seed": SEED, "steps": 0},
    {"seed": SEED, "steps": 65},
    {"seed": SEED, "steps": 5, "temperature": 0.4},
    {"seed": SEED, "steps": 5, "temperature": 1.6},
    {"steps": 5},
])
def test_entradas_invalidas(client, payload):
    assert client.post("/generate", json=payload).status_code == 422

"""API FastAPI: resuelve el alias champion una sola vez al iniciar."""
import os
from contextlib import asynccontextmanager

import mlflow
import mlflow.pytorch
import torch
from fastapi import FastAPI, HTTPException
from mlflow import MlflowClient
from pydantic import BaseModel, Field, field_validator

MODEL_NAME = os.environ.get("MODEL_NAME", "bach-chorales-alma")
TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
STATE = {}


@asynccontextmanager
async def lifespan(app):
    mlflow.set_tracking_uri(TRACKING_URI)
    version = MlflowClient().get_model_version_by_alias(MODEL_NAME, "champion").version
    model = mlflow.pytorch.load_model(f"models:/{MODEL_NAME}/{version}", map_location="cpu")
    model.eval()
    STATE.update(model=model, name=MODEL_NAME, version=str(version))
    yield
    STATE.clear()


app = FastAPI(title="Continuacion de corales de Bach", lifespan=lifespan)


class GenerateRequest(BaseModel):
    seed: list[list[int]] = Field(..., description="32 filas x 4 enteros (0-127)")
    steps: int = Field(..., ge=1, le=64)
    temperature: float = Field(1.0, ge=0.5, le=1.5)

    @field_validator("seed")
    @classmethod
    def check_seed(cls, v):
        if len(v) != 32:
            raise ValueError("seed debe tener exactamente 32 filas")
        for row in v:
            if len(row) != 4:
                raise ValueError("cada fila de seed debe tener exactamente 4 enteros")
            if any(n < 0 or n > 127 for n in row):
                raise ValueError("las notas deben estar entre 0 y 127")
        return v


class GenerateResponse(BaseModel):
    generated: list[list[int]]
    model: str
    version: str


@app.get("/health")
def health():
    if "model" not in STATE:
        raise HTTPException(status_code=503, detail="modelo no cargado")
    return {"status": "ok", "model": STATE["name"], "version": STATE["version"]}


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    if "model" not in STATE:
        raise HTTPException(status_code=503, detail="modelo no cargado")
    ctx = torch.tensor(req.seed, dtype=torch.long)
    with torch.inference_mode():
        out = STATE["model"].generate(ctx, req.steps, req.temperature)
    return GenerateResponse(generated=out.tolist(), model=STATE["name"],
                            version=STATE["version"])

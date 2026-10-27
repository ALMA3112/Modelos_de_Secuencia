# Modelos de Secuencia: Coral de Bach

Laboratorio PLN 2026 S02. Encoder-decoder LSTM / GRU / GRU con atención.

## Entorno
- AWS Academy Learner Lab, EC2 t3.large, Ubuntu 24.04
- MLflow 3.17.2 Tracking + Model Registry en la misma instancia (puerto 5000)
- PyTorch 2.14.1 (CPU)
- API FastAPI (puerto 8000)

> La IP pública cambia en cada sesión del lab. La URL vigente se entrega el día de la sustentación.
> Para obtenerla en la instancia: `./scripts/get_urls.sh`

## Instalación
    python3 -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt

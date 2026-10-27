import mlflow.pytorch
from mlflow import MlflowClient

mlflow.set_tracking_uri("http://127.0.0.1:5000")
NAME = "smoke-test-model"
v = MlflowClient().get_model_version_by_alias(NAME, "champion").version
try:
    m = mlflow.pytorch.load_model(f"models:/{NAME}/{v}", map_location="cpu")
    print("carga normal OK")
except Exception as e:
    print("carga normal fallo:", repr(e)[:300])
    m = mlflow.pytorch.load_model(f"models:/{NAME}/{v}", map_location="cpu", weights_only=False)
    print("carga con weights_only=False OK")
m.eval()
print("cargado OK, version", v)

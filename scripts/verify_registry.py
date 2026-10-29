import mlflow
import mlflow.pytorch
import torch
from mlflow import MlflowClient

mlflow.set_tracking_uri("http://127.0.0.1:5000")
NAME = "bach-chorales-alma"
mv = MlflowClient().get_model_version_by_alias(NAME, "champion")
m = mlflow.pytorch.load_model(f"models:/{NAME}/{mv.version}", map_location="cpu")
m.eval()
ctx = torch.randint(0, 60, (32, 4))
out = m.generate(ctx, 33, 1.0, torch.Generator().manual_seed(0))
print("version", mv.version, "run", mv.run_id)
print("salida", tuple(out.shape), int(out.min()), int(out.max()))

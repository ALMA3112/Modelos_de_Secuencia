import inspect
import mlflow, mlflow.pytorch
from mlflow import MlflowClient
from smoke_pkg.dummy import Dummy

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("smoke-test")

sig = inspect.signature(mlflow.pytorch.log_model)
has_fmt = "serialization_format" in sig.parameters
print("serialization_format disponible:", has_fmt)
kwargs = {"serialization_format": "pickle"} if has_fmt else {}

NAME = "smoke-test-model"
with mlflow.start_run(run_name="smoke") as run:
    mlflow.log_param("ok", True)
    mlflow.log_metric("loss", 1.0, step=0)
    with open("hola.txt", "w") as f:
        f.write("artefacto de prueba")
    mlflow.log_artifact("hola.txt")
    mlflow.pytorch.log_model(
        Dummy(),
        name="model",
        registered_model_name=NAME,
        code_paths=["smoke_pkg"],
        **kwargs,
    )
    print("run_id:", run.info.run_id)

c = MlflowClient()
v = max(c.search_model_versions(f"name='{NAME}'"), key=lambda x: int(x.version)).version
c.set_registered_model_alias(NAME, "champion", v)
print("alias champion -> version", v)

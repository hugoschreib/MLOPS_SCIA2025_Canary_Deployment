from fastapi import FastAPI
from pydantic import BaseModel
import mlflow
import pandas as pd


class MyModel(mlflow.pyfunc.PythonModel):
    def predict(self, context, model_input, params=None):
        return [x * 2 for x in model_input]


with mlflow.start_run():
    mlflow.pyfunc.log_model(
        name="model", python_model=MyModel(), pip_requirements=["pandas"]
    )
    run_id = mlflow.active_run().info.run_id

mlflow.set_tracking_uri("http://0.0.0.0:8080")
app = FastAPI()

class Body(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


model = mlflow.pyfunc.load_model(f"models:/tracking-quickstart/2")

@app.post("/predict")
def predict(body: Body):
    x_new = pd.DataFrame([[body.sepal_length, body.sepal_width, body.petal_length, body.petal_width]])
    res = model.predict(x_new)[0]
    print(res)
    return float(res)

class Model(BaseModel):
    version: int

@app.post("/update-model")
def update(body: Model):
    global model
    try:
        model = mlflow.pyfunc.load_model(f"models:/tracking-quickstart/{body.version}")
        return {"status": "SUCCESS"}
    except:
        return {"status": "ERROR"}

from fastapi import FastAPI
from pydantic import BaseModel
import mlflow
import pandas as pd
import random

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

CURRENT_MODEL = 1
NEXT_MODEL = 2
P = 0.2
IS_ACCEPTED = False

model = mlflow.pyfunc.load_model(f"models:/tracking-quickstart/2")

@app.post("/predict")
def predict(body: Body):
    model = mlflow.pyfunc.load_model(f"models:/tracking-quickstart/{NEXT_MODEL if IS_ACCEPTED else (NEXT_MODEL if random.randint(0,100) < int(P * 100) else CURRENT_MODEL)}")
    x_new = pd.DataFrame([[body.sepal_length, body.sepal_width, body.petal_length, body.petal_width]])
    res = model.predict(x_new)[0]
    print(res)
    return float(res)

class Model(BaseModel):
    version: int

@app.post("/update-model")
def update(body: Model):
    global model, IS_ACCEPTED, CURRENT_MODEL, NEXT_MODEL
    try:
        CURRENT_MODEL = body.version
        NEXT_MODEL = body.version + 1
        IS_ACCEPTED = 0
        model = mlflow.pyfunc.load_model(f"models:/tracking-quickstart/{body.version}")
        return {"status": "SUCCESS"}
    except:
        return {"status": "ERROR"}

@app.post("/accept-next-model")
def accept():
    global IS_ACCEPTED, CURRENT_MODEL, NEXT_MODEL
    IS_ACCEPTED = True
    CURRENT_MODEL = NEXT_MODEL
    return {"status": "SUCCESS"}


@app.get("/version")
def version():
    return {"version": CURRENT_MODEL}

# Put the code for your API here.
from fastapi import FastAPI
from pydantic import BaseModel
from starter.ml.model import inference
from starter.ml.data import process_data
import pickle
import joblib
import pandas as pd
app = FastAPI()

with open("model/model.pickle", "rb") as f:
    model = pickle.load(f) 
with open("model/encoder.pickle", "rb") as f:
    encoder = pickle.load(f) 
with open("model/labeler.pickle", "rb") as f:
    lb = pickle.load(f) 

class Data(BaseModel):
    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int
    capital_loss: int
    hours_per_week: int
    native_country: str

    model_config = {
        "json_schema_extra": {
            "examples":[
                    {
                    'age': 39,
                    'workclass':'State-gov', 
                    'fnlgt':77516,
                    'education':'Bachelors',
                    'education_num':13,
                    'marital_status': 'Never-married',
                    'occupation': 'Adm-clerical',
                    'relationship':'Not-in-family',
                    'race':'White',
                    'sex':'Male',
                    'capital_gain': 2174,
                    'capital_loss': 0,
                    'hours_per_week': 40,
                    'native_country':'United-States'
                    },
                    {
                    'age': 31,
                    'workclass':'Private', 
                    'fnlgt':84154,
                    'education':'Some-college',
                    'education_num':10,
                    'marital_status': 'Married-civ-spouse',
                    'occupation': 'Sales',
                    'relationship':'Husband',
                    'race':'White',
                    'sex':'Male',
                    'capital_gain': 0,
                    'capital_loss': 0,
                    'hours_per_week': 38,
                    'native_country':'?'
                    },  
                    ]
            }
        }
    

#greetings
@app.get("/")
async def greetings():
    return "Hello"

#inference
@app.post("/inference")
async def inference_api(input_data : Data):

    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]
    
    new_format = {k.replace('_', '-'): [v] for k, v in input_data.__dict__.items()}
    input_data = pd.DataFrame.from_dict(new_format)
    X, _, _, _ = process_data(
        input_data,
        categorical_features=cat_features,
        label=None,
        training=False,
        encoder=encoder,
        lb=lb
    )
    output = inference(model=model, X=X)[0]
    str_out = '<=50K' if output == 0 else '>50K'
    return {"pred": str_out}
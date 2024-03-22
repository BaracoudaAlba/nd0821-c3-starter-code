# Put the code for your API here.
from fastapi import FastAPI
from pydantic import BaseModel
from starter.ml.model import inference
from starter.ml.data import process_data
import pickle
import joblib
import pandas as pd
import pdb
app = FastAPI()

with open("starter/model/model.pickle", "rb") as f:
    model = pickle.load(f) 
with open("starter/model/encoder.pickle", "rb") as f:
    encoder = pickle.load(f) 
with open("starter/model/labeler.pickle", "rb") as f:
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
    data_dict = {  'age': [input_data.age],
            'workclass': [input_data.workclass], 
            'fnlgt': [input_data.fnlgt],
            'education': [input_data.education],
            'education-num': [input_data.education_num],
            'marital-status': [input_data.marital_status],
            'occupation': [input_data.occupation],
            'relationship': [input_data.relationship],
            'race': [input_data.race],
            'sex': [input_data.sex],
            'capital-gain': [input_data.capital_gain],
            'capital-loss': [input_data.capital_loss],
            'hours-per-week': [input_data.hours_per_week],
            'native-country': [input_data.native_country]
            }

    # pdb.set_trace()
    # new_format = {k.replace('_', '-'): [v] for k, v in input_data.__dict__.items()}
    # print(f"new format {new_format}")
    input_data = pd.DataFrame.from_dict(data_dict)
    X, _, _, _ = process_data(
        input_data,
        categorical_features=cat_features,
        label=None,
        training=False,
        encoder=encoder,
        lb=lb
    )
    output = inference(model=model, X=X)[0]
    if output == 0:
        output_string = '<=50K'
    else: 
        output_string = '>50K'

    return {"predicted": output_string}
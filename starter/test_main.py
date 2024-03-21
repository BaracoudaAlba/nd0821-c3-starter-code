from fastapi.testclient import TestClient
import json
from main import app, Data
import pdb
client = TestClient(app)

def test_api_greetings():

    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == "Hello"

def test_post_data_superior():
    data = {
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
            }
    data_example = json.dumps(data)
    r = client.post("/inference", data=data_example)
    assert r.status_code == 200
    assert r.json() == {"pred": ">50K"}

def test_post_data_inferior():
    data_example = json.dumps(Data.model_config["json_schema_extra"]["examples"][1])
    r = client.post("/inference", data=data_example)
    assert r.status_code == 200
    assert r.json() == {"pred": "<=50K"}
    
if __name__=="__main__":
    # test_api_greetings()
    test_post_data_superior()
    test_post_data_inferior()    
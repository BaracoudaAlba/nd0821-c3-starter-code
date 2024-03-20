from fastapi.testclient import TestClient
import json
from main import app, Data

client = TestClient(app)

def test_api_greetings():

    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == "Hello"

def test_post_data():

    data_example = json.dumps(Data.model_config["json_schema_extra"]["examples"][0])
    r = client.post("/inference", data=data_example)
    assert r.status_code == 200
    assert r.json()["prediction"] == '<=50K'

def test_post_data_V2():
    data_example = json.dumps(Data.model_config["json_schema_extra"]["examples"][1])
    r = client.post("/inference", data=data_example)
    assert r.status_code == 200
    assert r.json()["prediction"] == '>50K'
    
if __name__=="__main__":
    test_api_greetings()
    test_post_data()
from fastapi.testclient import TestClient
import json
from main import app, Data
import pdb
client = TestClient(app)

def test_api_greetings():

    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == "Hello"
# 40,Private,121772,Assoc-voc,11,Married-civ-spouse,Craft-repair,Husband,Asian-Pac-Islander,Male,0,0,40,?,>50K
def test_post_data_superior():
    data = {
            'age': 40,
            'workclass':'Private', 
            'fnlgt':121772,
            'education':'Assoc-voc',
            'education_num':11,
            'marital_status': 'Married-civ-spouse',
            'occupation': 'Craft-repair',
            'relationship':'Husband',
            'race':'Asian-Pac-Islander',
            'sex':'Male',
            'capital_gain': 0,
            'capital_loss': 0,
            'hours_per_week': 40,
            'native_country':'?'
            }
    data_example = json.dumps(data)
    r = client.post("/inference", data=data_example)
    assert r.status_code == 200
    assert r.json() == {"predicted": ">50K"}


# 32,Private,205019,Assoc-acdm,12,Never-married,Sales,Not-in-family,Black,Male,0,0,50,United-States,<=50K
def test_post_data_inferior():

    data = {
            'age': 32,
            'workclass':'Private', 
            'fnlgt':205019,
            'education':'Assoc-acdm',
            'education_num':12,
            'marital_status': 'Never-married',
            'occupation': 'Sales',
            'relationship':'Not-in-family',
            'race':'Black',
            'sex':'Male',
            'capital_gain': 0,
            'capital_loss': 0,
            'hours_per_week': 50,
            'native_country':'United-States'
            }
    
    data_example = json.dumps(data)
    r = client.post("/inference", data=data_example)
    assert r.status_code == 200
    assert r.json() == {"predicted": "<=50K"}
    
if __name__=="__main__":
    test_api_greetings()
    test_post_data_superior()
    test_post_data_inferior()    
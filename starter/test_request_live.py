import json
import requests


if __name__ == "__main__":

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

    resp = requests.post(
        "https://nd0821-c3-starter-code-co39.onrender.com/inference", data=data_example)

    print(f"status code :{resp.status_code}")
    print(resp.json())
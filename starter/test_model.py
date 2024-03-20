
import pytest
import pandas as pd
from starter.ml.model import train_model, compute_model_metrics, inference
from starter.ml.data import process_data 
from sklearn.model_selection import train_test_split as tts
import pickle
from starter.train_model import clean_data
import numpy as np

@pytest.fixture(scope="module")
def data():
    data = pd.read_csv("data/census_dummy.csv")
    data = clean_data(data)
    return data


@pytest.fixture(scope="module")
def cat_features():

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

    return cat_features

@pytest.fixture(scope="module")
def ready_dataset(cat_features, data):

    train, test = tts(data, test_size= 0.2)
    
    X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
    )

    X_test, y_test, _, _ = process_data(
    test, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
    )
    print(type(X_train))
    return [X_train, y_train, X_test, y_test]

@pytest.fixture(scope="module")
def model(ready_dataset):
    X_train = ready_dataset[0]
    y_train = ready_dataset[1]
    model_trained = train_model(X_train, y_train)


    return model_trained

################### tests #################

def test_train_model(ready_dataset):

    X_train = ready_dataset[0]
    y_train = ready_dataset[1]

    model_trained = train_model(X_train, y_train)
    assert model_trained 

def test_inference(ready_dataset, model):
    X_train = ready_dataset[0]
    y_train = ready_dataset[1]

    X_test = ready_dataset[2]

    y_pred = inference(model, X_train)
    
    assert y_pred.shape[0] >0 

def test_compute_model_metrics(ready_dataset, model):

    X_test = ready_dataset[2]
    y_test = ready_dataset[3]

    y_pred = inference(model, X_test)
    precision, recall, fbeta = compute_model_metrics(y_test, y_pred)

    assert type(precision) == np.float64 
    assert type(recall) == np.float64
    assert type(fbeta) == np.float64

if __name__=="__main__":
    data = pd.read_csv("data/census_dummy.csv")

    train, test = tts(data, test_size= 0.2)
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

    X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
    )

    X_test, y_test, _, _ = process_data(
    test, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
    )
    test_train_model([X_train, y_train, X_test, y_test])
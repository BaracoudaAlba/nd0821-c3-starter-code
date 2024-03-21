
import pytest
import pandas as pd
from starter.ml.model import train_model, compute_model_metrics, inference
from starter.ml.data import process_data 
from sklearn.model_selection import train_test_split as tts
import pickle
from starter.train_model import clean_data
import numpy as np


################### tests #################

def test_train_model():

    X_train = np.random.rand(18,10)
    y_train = np.random.randint(low=0, high=2,size=18)
    model_trained = train_model(X_train, y_train)
    assert model_trained is not None

def test_inference():

    X_test = np.random.rand(18,10)
    y_test = np.random.randint(low=0, high=2,size=18)
    
    model_trained = train_model(X_test, y_test)

    y_pred = inference(model_trained, X_test)
    
    assert y_pred.shape[0] == y_test.shape[0]

def test_compute_model_metrics():

    X_train = np.random.rand(18,10)
    y_train = np.random.randint(low=0, high=2,size=18)
    
    model = train_model(X_train, y_train)


    y_pred = np.random.randint(low=0, high=2,size=18)

    y_test = np.random.randint(low=0, high=2,size=18)

    y_pred = inference(model, X_train)
    precision, recall, fbeta = compute_model_metrics(y_test, y_pred)

    assert precision is not None 
    assert recall is not None 
    assert fbeta is not None 


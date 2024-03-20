# Script to train machine learning model.

from sklearn.model_selection import train_test_split
import pandas as pd
# Add the necessary imports for the starter code.
import pdb
# Add code to load in the data.
from starter.ml.data import process_data
from starter.ml.model import train_model, compute_model_metrics, inference

import pickle

def clean_data(data_df):
    """cleaning data
    """
    data_df['salary'].replace('<=50K', 0, inplace = True ) 
    data_df['salary'].replace('>50K', 1, inplace = True )
    return data_df.drop_duplicates()


def save_model(model, encoder, labeler):


    with open(r"model/model.pickle", "wb") as output_file:
        pickle.dump(model, output_file)
    with open(r"model/encoder.pickle", "wb") as output_file:
        pickle.dump(encoder, output_file)
    with open(r"model/labeler.pickle", "wb") as output_file:
        pickle.dump(labeler, output_file)

data_path = "./data/census_dummy.csv"
data_df = pd.read_csv(data_path)
print(data_df.shape)
data_df = clean_data(data_df)


# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data_df, test_size=0.20)

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
# Proces the test data with the process_data function.
X_test, y_test, encoder, lb = process_data(
    test, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
)
# Train and save a model.
print("training model")
trained_model = train_model(X_train, y_train)
save_model(trained_model, encoder, lb)

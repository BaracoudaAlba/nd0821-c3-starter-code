import pandas as pd
import pickle
from starter.ml.model import inference, compute_model_metrics
from starter.ml.data import process_data


def performance_on_slice(model, encoder, lb, data_df, cat_features, features):

    print(f"Sliced for features {features}")
    slice_metrics = {}
    for value in data_df[features].unique():
        X_slice = data_df[data_df[features] == value]


        X_slice, y_slice, _, _ = process_data(
            X_slice, cat_features, label="salary", training=False, encoder=encoder, lb=lb)
        y_preds = inference(model, X_slice)

        prec, recall, fbeta = compute_model_metrics(y_slice, y_preds)

        slice_metrics[value] = {'Precision': prec,
                                'Recall': recall,
                                'Fbeta': fbeta}
        print(
            f"Metrics regarding {features}, column {value}: {slice_metrics[value]}")

    # save results
    with open('slice_output.txt', 'w') as f:
        for k, v in slice_metrics.items():
            f.write(f"{features}  {k}: {v}")
            f.write("\n")
    return slice_metrics


if __name__ == '__main__':
    
    data_df = pd.read_csv( './data/census_dummy.csv')
    with open("./model/model.pickle", "rb") as f:
        model = pickle.load(f) 
    with open("./model/encoder.pickle", "rb") as f:
        encoder = pickle.load(f) 
    with open("./model/labeler.pickle", "rb") as f:
        lb = pickle.load(f) 

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
    performance_on_slice(
        model=model,
        encoder=encoder,
        lb=lb,
        data_df=data_df,
        cat_features=cat_features,
        features='education'
    )
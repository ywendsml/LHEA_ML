import pickle

import pandas as pd

def load_pkl_file_to_pd(file_name):
    """
    This function is to load the features of all alloy candidates from a pkl file.
    """

    with open(f'{file_name}.pkl', 'rb') as f:
        total_features = pickle.load(f)

    return total_features

def load_model_and_scaler(model_filename, scaler_filename):

    with open(f'{model_filename}.pkl', 'rb') as f:
        model = pickle.load(f)
    with open(f'{scaler_filename}.pkl', 'rb') as f:
        scaler = pickle.load(f)

    return model, scaler


def produce_scaled_features(features, scaler):

    return scaler.transform(features)


def predict_phase(features, model, scaler, to_csv=False, csv_name='prediction.csv'):

    scaled_features = produce_scaled_features(features, scaler)
    
    if to_csv:
        pd.DataFrame(model.predict(scaled_features)).to_csv(csv_name)

    return pd.DataFrame(model.predict(scaled_features))


if __name__ == '__main__':
    # Load the features of all alloy candidates
    total_features = load_pkl_file_to_pd('v0_test')
    # Load the trained model and scaler
    model, scaler = load_model_and_scaler('logistic_regression_model', 'scaler')
    # Predict the phases of all alloy candidates
    prediction = predict_phase(total_features[:, 6:], model, scaler)
    
    print(prediction)
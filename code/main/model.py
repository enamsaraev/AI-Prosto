import joblib
import pandas as pd


def load_model(model_path: str):
    loaded_model = joblib.load(model_path)
    return loaded_model

def make_predict(df: pd.DataFrame, model_path: str):
    model = load_model(model_path=model_path)
    predict = model.predict(df)

    return predict
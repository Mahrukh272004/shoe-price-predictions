import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

# Load test dataset
def load_data():
    test_path = "datasets/test.csv"
    df_test = pd.read_csv(test_path)
    return df_test

# Load trained model
def load_model():
    model_path = "models/random_forest_model.pkl"
    model = joblib.load(model_path)
    return model

# Test 1: Check if data is loaded correctly
def test_data_loading():
    df_test = load_data()
    assert not df_test.empty, "Test data is empty."

# Test 2: Check if the model is loaded correctly
def test_model_loading():
    model = load_model()
    assert model is not None, "Model failed to load."

# Test 3: Evaluate model performance
def test_model_performance():
    df_test = load_data()
    X_test = df_test[['brand', 'categories', 'colors']]
    y_test = df_test['price']

    model = load_model()
    y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    assert rmse < 10, f"RMSE is too high: {rmse}"
    assert r2 > 0.7, f"R² score is too low: {r2}"

# Test 4: Check if predictions are within expected range
def test_predictions_range():
    df_test = load_data()
    X_test = df_test[['brand', 'categories', 'colors']]
    model = load_model()
    y_pred = model.predict(X_test)

    assert all(y_pred >= 0), "Some predictions are negative."
    assert all(y_pred <= 1000), "Some predictions are too high."
import joblib
import numpy as np
import os

def load_model():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, '..', 'model', 'xgboost_model.joblib')
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    
    return joblib.load(model_path)

def make_prediction(input_data):
    model = load_model()
    prediction = model.predict(np.array([input_data]))  # Convert input_data to a 2D array
    
    # Convert prediction to standard Python float for JSON serialization
    prediction = prediction.astype(float)  # Convert from float32 to float
    return prediction.tolist()  # Convert to a list if it’s an array

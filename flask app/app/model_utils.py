import os
import pandas as pd
import xgboost as xgb
import joblib
from sklearn.model_selection import train_test_split
from google.cloud import storage
from config import *

# BASE_PATH = r'C:\Users\MUSTAKIM\Desktop\flask app'


def train_model():
    """Train the model and save it to the specified path."""
    # Correct the data file path
    data_file_path = os.path.join(BASE_PATH, 'DATA' ,'PRICEDATA.CSV')
    data = pd.read_csv(data_file_path)

    # Check for required column
    if 'ocean_proximity' not in data.columns:
        raise ValueError("Dataset must contain 'ocean_proximity' column.")
    
    # Convert categorical 'ocean_proximity' to one-hot encoding
    data = pd.get_dummies(data, columns=['ocean_proximity'], drop_first=True)
    
    # Define features and target
    X = data[['longitude', 'latitude', 'median_income', 
              'ocean_proximity_INLAND', 'ocean_proximity_NEAR OCEAN', 
              'housing_median_age', 'total_rooms', 'total_bedrooms']]
    y = data['median_house_value']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize and train the XGBoost model
    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Save the model to the specified directory
    os.makedirs(os.path.dirname(LOCAL_MODEL_PATH), exist_ok=True)  # Ensure the parent directory exists
    joblib.dump(model, LOCAL_MODEL_PATH)
    print(f"Model saved to {LOCAL_MODEL_PATH}")

def download_model_from_gcs():
    """Download the model from Google Cloud Storage if not available locally."""
    client = storage.Client()
    bucket = client.get_bucket(BUCKET_NAME)
    blob = bucket.blob(MODEL_FILE_NAME)
    blob.download_to_filename(LOCAL_MODEL_PATH)
    print(f"Model downloaded to {LOCAL_MODEL_PATH}")

def load_model():
    """Load the model from the local file system or download it from GCS if not available."""
    if not os.path.exists(LOCAL_MODEL_PATH):
        download_model_from_gcs()
    return joblib.load(LOCAL_MODEL_PATH)

def make_prediction(model, input_data):
    """Make a prediction using the loaded model."""
    input_data = [input_data]  # XGBoost expects a 2D array for prediction
    return float(model.predict(input_data)[0])

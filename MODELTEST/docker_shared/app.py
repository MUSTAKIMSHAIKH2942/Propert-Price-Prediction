
import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb
import joblib
import os
from flask import Flask, request, jsonify, render_template
from google.cloud import storage  # Import GCS client

# Create Flask app
app = Flask(__name__)

# Set the environment variable for authentication
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'credentials.json'

# Set the GCS bucket and model details
BUCKET_NAME = 'model-bucket-test-001'
MODEL_FILE_NAME = 'xgboost_model.joblib' 
LOCAL_MODEL_PATH = 'local_xgboost_model.joblib'

# Define base path
# BASE_PATH = r"C:\Users\MUSTAKIM\Desktop\MODELTEST"
BASE_PATH = '/flask_app'
# Load dataset and train the model
def train_model():
    data_file_path = os.path.join(BASE_PATH, 'DATA', 'PRICEDATA.CSV')
    # data_file_path = '/flask_app\DATA\PRICEDATA.CSV'
    print(f"Loading data from: {data_file_path}")
    data = pd.read_csv(data_file_path)
    
    # Ensure 'ocean_proximity' is one of the columns
    if 'ocean_proximity' not in data.columns:
        raise ValueError("The dataset must contain the 'ocean_proximity' column.")
        
    # One-hot encoding for 'ocean_proximity'
    data = pd.get_dummies(data, columns=['ocean_proximity'], drop_first=True)

    # Feature selection
    X = data[['longitude', 'latitude', 'median_income', 
               'ocean_proximity_INLAND', 'ocean_proximity_NEAR OCEAN', 
               'housing_median_age', 'total_rooms', 'total_bedrooms']]
    y = data['median_house_value']

    # Splitting the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train the model
    xg_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
    xg_model.fit(X_train, y_train)

    # Save the trained model
    model_dir = os.path.join(BASE_PATH, 'model')
    os.makedirs(model_dir, exist_ok=True)
    model_filename = os.path.join(model_dir, 'xgboost_model.joblib')
    joblib.dump(xg_model, model_filename)
    print(f"Model saved to {model_filename}")

# Function to download model from GCS
def download_model_from_gcs(bucket_name, model_file_name, local_model_path):
    """Downloads the model from Google Cloud Storage to a local file."""
    client = storage.Client()
    bucket = client.get_bucket(bucket_name)
    blob = bucket.blob(model_file_name)
    blob.download_to_filename(local_model_path)
    print(f"Model downloaded from GCS: {bucket_name}/{model_file_name} to {local_model_path}")

# Load model (will download it if not found locally)
if not os.path.exists(LOCAL_MODEL_PATH):
    download_model_from_gcs(BUCKET_NAME, MODEL_FILE_NAME, LOCAL_MODEL_PATH)

# Load the model
model = joblib.load(LOCAL_MODEL_PATH)

# Make prediction
def make_prediction(input_data):
    """Make a prediction using the trained model."""
    prediction = model.predict([input_data])
    return float(prediction[0])  # Convert to float for JSON serialization

# Route for the main page
@app.route('/')
def home():
    return render_template('index.html')  # Ensure the HTML file is named 'index.html'

# Route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = [
            float(request.form['longitude']),
            float(request.form['latitude']),
            float(request.form['median_income']),
            int(request.form['ocean_proximity_INLAND']),
            int(request.form['ocean_proximity_NEAR_OCEAN']),
            float(request.form['housing_median_age']),
            float(request.form['total_rooms']),
            float(request.form['total_bedrooms'])
        ]
        prediction = make_prediction(input_data)
        return jsonify(prediction=prediction)  # Send back the prediction
    except KeyError as e:
        return jsonify(error=f'Missing key: {str(e)}'), 400  # Handle missing keys
    except ValueError as e:
        return jsonify(error=f'Invalid input: {str(e)}'), 400  # Handle value errors
    except Exception as e:
        return jsonify(error=str(e)), 400  # Return error message with status code 400

if __name__ == '__main__':
    # Uncomment the line below to train the model only once
    train_model()
    
    app.run(host='0.0.0.0', port=8080, debug=True)


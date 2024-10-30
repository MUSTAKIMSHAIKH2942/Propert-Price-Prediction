# Propert-Price-Prediction
XGBoost Price Prediction Flask Application
This project is a Flask-based web application that predicts house prices using an XGBoost model. The app leverages Docker for containerization and Google Cloud Platform (GCP) for storing and downloading model files.

Project Structure
/Flask_APP/
│
├── app/                       # Application source code
│   ├── templates/
│   │   └── index.html         # HTML template for the web UI
│   └── main.py                # Main Flask app script
├── DATA/                      # Directory containing dataset (PRICEDATA.CSV)
├── requirements.txt           # Python dependencies
├── Dockerfile                 # Docker setup file for building the container
├── credentials.json           # GCP credentials file for accessing GCS
└── README.md                  # Project documentation
Requirements


Python 3.9 (used in the Docker image)
Flask for the web framework
XGBoost for the machine learning model
joblib for saving and loading the model
Google Cloud Storage to store the trained model
Docker to containerize the application

Installation
Clone the repository:
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name

Set up environment variables: Configure paths and settings as needed in a .env file or in your Dockerfile.

Install dependencies (if running locally):
pip install -r requirements.txt

Running the App Locally
Train the model: Use the train_model() function in the script to train and save the model locally.

Run the Flask app:
python app/main.py
Open your browser and go to http://localhost:8080 to access the app.

Running with Docker
Build the Docker image:

docker build -t xgboost-price-prediction .
Run the container:

docker run -p 8080:8080 xgboost-price-prediction
Access the app at http://localhost:8080.

Google Cloud Platform (GCP) Services
Google Cloud Storage (GCS): The trained model file is stored on GCS and downloaded to the container at runtime. Ensure that the GCS bucket name and model filename are set correctly in the script.

Required GCP Services:

Google Cloud Storage: to store and retrieve the trained model
Google Cloud IAM: to manage permissions for accessing the storage bucket
Environment Variables for GCP
GOOGLE_APPLICATION_CREDENTIALS: Path to the GCP credentials JSON file in the Docker container (configured in the Dockerfile).
BUCKET_NAME: Name of the GCS bucket where the model is stored.
MODEL_FILE_NAME: Model filename in GCS.
API Endpoints
GET /: Renders the main page with the input form.
POST /predict: Accepts form data from the UI, processes it, and returns a predicted price.

Files
main.py: The main application file that initializes Flask and sets up model loading and predictions.
Dockerfile: Defines the setup for the Docker container, including installing dependencies and configuring environment variables.
credentials.json: GCP credentials to authenticate the application for accessing Google Cloud Storage.
requirements.txt: Specifies dependencies for running the application.

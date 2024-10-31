# Propert-Price-Prediction
XGBoost Price Prediction Flask Application
This project is a Flask-based web application that predicts house prices using an XGBoost model. The app leverages Docker for containerization and Google Cloud Platform (GCP) for storing and downloading model files.

Project Structure

![Directory Structure Pic](https://github.com/MUSTAKIMSHAIKH2942/Propert-Price-Prediction/blob/main/filr%20structure.JPG)
                                                    
XGBoost Price Prediction Web Application
This project is a web-based application that predicts housing prices based on various input features using the XGBoost machine learning algorithm. Developed with Flask, the application is containerized using Docker and leverages Google Cloud Storage (GCS) to store the model. The project includes a user-friendly interface for inputting property features and displays predicted prices dynamically.

## Table of Contents
Project Overview
Data Description
XGBoost Algorithm
Architecture
Setup and Installation
Usage
Docker and GCP Setup
API Endpoints
Example Images
Future Enhancements
License
Project Overview
Objective

The objective of this project is to predict the median house value in California based on various features of the property, such as location, median income, room counts, and proximity to the ocean. Using XGBoost, a powerful gradient-boosted decision tree model, the application provides highly accurate predictions by training on historical data of property prices.

## Key Features
Flask Web Interface: A simple and intuitive user interface for data entry and displaying predictions.
Model Loading: The model is either loaded from a local directory or downloaded from Google Cloud Storage if not available locally.
Dockerized Setup: Enables easy deployment with dependencies and configurations bundled in a Docker container.
GCP Integration: Supports Google Cloud Storage to store and retrieve the trained XGBoost model securely.
Data Description
The dataset used in this project includes features important for predicting property prices, such as:

longitude: Property’s longitudinal coordinate
latitude: Property’s latitudinal coordinate
median_income: Median income of residents in the property’s vicinity
housing_median_age: Median age of housing units in the area
total_rooms: Total rooms per property
total_bedrooms: Total bedrooms per property
ocean_proximity: Proximity to ocean (categorical with values like “INLAND”, “NEAR OCEAN”)
## The target feature is:

median_house_value: The median house value for each property.
 XGBoost Algorithm
Why XGBoost?
XGBoost is a high-performance, scalable machine learning library that specializes in boosting weak learners to improve accuracy. It is particularly suitable for structured/tabular data like this project’s dataset and offers several advantages:

Speed and Performance: XGBoost is optimized for speed and memory, making it a good fit for large datasets.
Handling Missing Values: The algorithm handles missing values by default, a common challenge in real estate data.
Interpretability: XGBoost allows easy interpretation of feature importance, which is useful for understanding which factors influence house prices the most.
Algorithm Explanation
The algorithm works by training a series of decision trees where each tree attempts to correct the errors of the previous one. By combining these weak learners, XGBoost achieves strong predictive accuracy.

## Architecture
Data Loading: Reads housing data from a CSV file located in the DATA/ directory.
Feature Engineering: Converts categorical data (ocean proximity) to one-hot encoded features.
Model Training: Trains an XGBoost regression model on the dataset.
Model Storage: Saves the trained model to a local path or uploads it to GCS for secure storage.
Prediction: Accepts user input via the web interface, makes a prediction, and displays the result.
Setup and Installation
Prerequisites
Python 3.9
Google Cloud SDK (if using GCP for model storage)
Docker (for containerization)
## Installation
## Clone the repository:

git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
Set up environment variables: Configure paths and environment variables in a .env file.

## Install dependencies:

pip install -r requirements.txt
Usage
Running the App Locally
Train the model:

Run the train_model() function in main.py to train and save the model locally.
Run the Flask app:


python app/main.py
Access the app: Go to http://localhost:8080 in your browser.

Docker and GCP Setup
Docker Setup
Build the Docker image:

docker build -t xgboost-price-prediction .
## Run the container:

docker run -p 8080:8080 xgboost-price-prediction
Access the app: Visit http://localhost:8080.

Google Cloud Platform (GCP) Services
Google Cloud Storage: Stores the trained model file (MODEL_FILE_NAME).
IAM Permissions: Ensure the service account has Storage Object Viewer access for downloading the model file.
Environment Variables for GCP
GOOGLE_APPLICATION_CREDENTIALS: Path to GCP credentials in the Docker container.
BUCKET_NAME: GCS bucket name.
MODEL_FILE_NAME: Name of the model file in GCS.
## API Endpoints
GET /: Renders the main HTML page with the input form.
POST /predict: Accepts input from the form, processes it, and returns the price prediction.

## Example Input
Feature	Value
Longitude	-122.23
Latitude	37.88
Median Income	8.3252
Housing Median Age	41
Total Rooms	880
Total Bedrooms	129
Ocean Proximity	INLAND

## Example Output
Predicted House Price
$450,000


Example Images
Input Form
![Input form Pic](https://github.com/MUSTAKIMSHAIKH2942/Propert-Price-Prediction/blob/main/inputfrontend.JPG)
Output Prediction
![output  Pic](https://github.com/MUSTAKIMSHAIKH2942/Propert-Price-Prediction/blob/main/frontend.JPG)
Future Enhancements
Add More Features: Improve predictions by adding additional features such as population density, employment rates, or crime statistics.
Hyperparameter Tuning: Experiment with different hyperparameter settings to further improve accuracy.
Scalability: Deploy the app to Kubernetes for handling larger traffic loads.

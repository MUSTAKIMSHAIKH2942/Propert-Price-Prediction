# import pandas as pd
# from sklearn.model_selection import train_test_split
# import xgboost as xgb
# import joblib
# import sys
# import os

# # Add the parent directory to the system path
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# # Define the base path for the dataset
# base_path = r"C:\Users\MUSTAKIM\Desktop\MODELTEST"

# # Construct the full file path for the dataset
# data_file_path = os.path.join(base_path, 'DATA', 'PRICEDATA.CSV')

# # Load dataset
# data = pd.read_csv(data_file_path)

# # Convert categorical variable 'ocean_proximity' to one-hot encoded variables
# data = pd.get_dummies(data, columns=['ocean_proximity'], drop_first=True)

# # Define input features (X) and target variable (y)
# X = data[['longitude', 'latitude', 'median_income', 
#            'ocean_proximity_INLAND', 'ocean_proximity_NEAR OCEAN', 
#            'housing_median_age', 'total_rooms', 'total_bedrooms']]  # Include all selected features
# y = data['median_house_value']  # The target column

# # Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Initialize and train the XGBoost model
# xg_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
# xg_model.fit(X_train, y_train)

# # Define the model directory and create it if it doesn't exist
# model_dir = os.path.join(base_path, 'model')
# os.makedirs(model_dir, exist_ok=True)  # Create the directory if it doesn't exist

# # Save the model
# model_filename = os.path.join(model_dir, 'xgboost_model.joblib')
# joblib.dump(xg_model, model_filename)

# print(f"Model saved to {model_filename}")

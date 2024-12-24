import os
import pandas as pd
import joblib
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import time

# Load data
data_path = "College_Seats_Data.csv"
model_path = "static_xgboost_college_model.pkl"

# Read the CSV
seat_data = pd.read_csv(data_path)

# Debug column names
print("Columns in the dataset:", seat_data.columns)

# Load the pre-trained XGBoost model
static_model = joblib.load(model_path)

print("Data and model loaded successfully.")

def preprocess_data(data):
    # Example feature engineering
    data['availability_ratio'] = data['Seats Available'] / (data['total_seats'] + 1)
    if 'Department Name' in data.columns:
        data['category_encoded'] = LabelEncoder().fit_transform(data['Department Name'])
    features = ['rank', 'availability_ratio', 'category_encoded']  # Example features
    return data[features]

def update_model_with_new_data(new_data, model, data_file):
    # Ensure the file is writable
    if not os.access(data_file, os.W_OK):
        print(f"Error: The file {data_file} is not writable. Please close it if open elsewhere.")
        return model

    # Append new data to the existing CSV
    new_data.to_csv(data_file, mode='a', header=False, index=False)

    # Reload updated data
    updated_data = pd.read_csv(data_file)

    # Preprocess features and split data
    X = preprocess_data(updated_data)
    y = updated_data['admission_probability']  # Assuming this column exists

    # Retrain the model with updated data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=10, verbose=True)
    
    print("Model updated with new data.")
    return model

def predict_real_time(model, input_data):
    X_input = preprocess_data(input_data)
    predictions = model.predict(xgb.DMatrix(X_input))
    return predictions

def update_seat_matrix(data, college_name, department_name, change_in_seats):
    # Check if required columns exist
    if 'College Name' not in data.columns:
        print("Error: 'College Name' column not found in the dataset.")
        return data

    if 'Department Name' not in data.columns:
        print("Error: 'Department Name' column not found in the dataset.")
        return data

    # Find the matching rows
    condition = (data['College Name'] == college_name) & (data['Department Name'] == department_name)
    data.loc[condition, 'Seats Available'] += change_in_seats
    print(f"Seat matrix updated for {college_name}, {department_name}.")
    return data

def real_time_system(data_file, model):
    while True:
        # Simulate real-time updates (e.g., new data or seat changes)
        time.sleep(5)  # Check every 5 seconds for updates

        # Example: Simulated new data
        new_data = pd.DataFrame({
            'rank': [750],
            'Seats Available': [3],
            'total_seats': [10],
            'Department Name': ['OBC'],
            'admission_probability': [0.65]  # Placeholder
        })

        # Update seat matrix and retrain
        seat_data_updated = update_seat_matrix(seat_data, 'College A', 'OBC', -1)
        model = update_model_with_new_data(new_data, model, data_file)

        # Predict using the updated model
        predictions = predict_real_time(model, seat_data_updated)
        print(f"Real-time Predictions: {predictions}")

if __name__ == "__main__":
    real_time_system(data_path, static_model)

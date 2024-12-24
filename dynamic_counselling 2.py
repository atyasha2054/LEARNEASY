import joblib

# Load static model
def load_static_model(model_path="static_xgboost_college_model.pkl"):
    try:
        model = joblib.load(model_path)
        print("Static model loaded successfully.")
    except FileNotFoundError:
        raise ValueError("Static model not found. Ensure 'model.pkl' exists.")
    return model


import pandas as pd

# Load and preprocess data
def preprocess_data(file_path):
    data = pd.read_csv(file_path)

    # Feature engineering
    data['availability_score'] = data['seats_available'] / (data['rank'] + 1)
    data['category_encoded'] = data['category'].astype('category').cat.codes

    # Features and target
    features = ['rank', 'availability_score', 'category_encoded']
    X = data[features]
    y = data.get('admission_probability', None)  # Target column may be optional for predictions
    return X, y

# Fine-tune static model with updated data
def fine_tune_model(static_model, X, y, model_path="/content/static_xgboost_college_model.pkl"):
    if y is not None:
        static_model.fit(X, y)  # Retrain the model with updated data
        joblib.dump(static_model, model_path)  # Save the updated model
        print("Static model fine-tuned and saved.")
    else:
        print("No target data available. Skipping retraining.")
    return static_model


import os
import time

# Monitor the CSV file for updates
def monitor_and_update(csv_path, static_model, model_path="/content/static_xgboost_college_model.pkl"):
    last_modified_time = os.path.getmtime(csv_path)

    while True:
        current_modified_time = os.path.getmtime(csv_path)
        if current_modified_time != last_modified_time:
            print("CSV file updated. Retraining the model...")
            X, y = preprocess_data(csv_path)
            static_model = fine_tune_model(static_model, X, y, model_path)
            last_modified_time = current_modified_time
        time.sleep(1)  # Check for updates every second


def predict_admission(static_model, new_data):
    # Preprocess new data
    new_data['availability_score'] = new_data['seats_available'] / (new_data['rank'] + 1)
    new_data['category_encoded'] = new_data['category'].astype('category').cat.codes
    X_new = new_data[['rank', 'availability_score', 'category_encoded']]
    
    # Predict probabilities
    probabilities = static_model.predict(X_new)
    return probabilities


# Paths
csv_path = "College_Seats_Data.csv"
model_path = "static_xgboost_college_model.pkl"

# Load static model
static_model = load_static_model(model_path)

# Start monitoring and updating
monitor_and_update(csv_path, static_model, model_path)

# Example: Predict for a new candidate after retraining
new_candidate = pd.DataFrame({
    'rank': [500],
    'seats_available': [10],
    'category': ['General']
})

probabilities = predict_admission(static_model, new_candidate)
print(f"Predicted Admission Probabilities: {probabilities}")

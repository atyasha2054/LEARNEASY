import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import joblib
import os

# Define file paths
CSV_FILE = "admission_data.csv"
MODEL_FILE = "admission_model.pkl"

# Preprocessing function
def preprocess_data(data):
    # Feature engineering
    data['availability_score'] = data['seats_available'] / (data['rank'] + 1)
    data['category_encoded'] = data['category'].astype('category').cat.codes
    data['institution_encoded'] = data['institution'].astype('category').cat.codes
    
    # Features and target
    features = ['rank', 'availability_score', 'category_encoded', 'institution_encoded']
    X = data[features]
    y = data['admission_probability'] if 'admission_probability' in data.columns else None
    return X, y

# Load and train model
def train_model(csv_file):
    # Load data
    data = pd.read_csv(csv_file)
    X, y = preprocess_data(data)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Define model pipeline
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', GradientBoostingRegressor(n_estimators=100))
    ])
    
    # Train model
    model.fit(X_train, y_train)
    
    # Save the model
    joblib.dump(model, MODEL_FILE)
    print("Model trained and saved.")
    
    return model

# Predict function
def predict_admission(model, new_data):
    X, _ = preprocess_data(new_data)
    predictions = model.predict(X)
    return predictions



import time

def watch_csv_and_retrain(csv_file, model_file):
    last_modified_time = os.path.getmtime(csv_file)
    
    while True:
        current_time = os.path.getmtime(csv_file)
        if current_time != last_modified_time:
            print("CSV file updated. Retraining the model...")
            train_model(csv_file)
            last_modified_time = current_time
        time.sleep(5)  # Check every 5 seconds

# Start watching the CSV
watch_csv_and_retrain(CSV_FILE, MODEL_FILE)


# Load the latest model and make predictions
def make_predictions(new_data):
    # Load the latest model
    if not os.path.exists(MODEL_FILE):
        raise FileNotFoundError("No trained model found. Train the model first.")
    model = joblib.load(MODEL_FILE)
    
    # Predict probabilities
    predictions = predict_admission(model, new_data)
    return predictions

# Example usage
if __name__ == "__main__":
    # Sample new data for prediction
    new_data = pd.DataFrame({
        'rank': [550, 1300],
        'seats_available': [4, 6],
        'category': ['General', 'OBC'],
        'institution': ['Institution A', 'Institution B']
    })
    
    # Make predictions
    predictions = make_predictions(new_data)
    print(f"Predicted Admission Probabilities: {predictions}")




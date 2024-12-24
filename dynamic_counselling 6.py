import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
import numpy as np
import os
import time
import joblib
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# Global variables
TOTAL_APPLICANTS = 1000
MODEL_FILE = 'static_xgboost_college_model.pkl'
CSV_DIRECTORY = 'LEARNEASY'  # Directory to monitor for new CSV files
PROCESSED_FILES = set()  # Track processed files

def preprocess_data(data):
    """Preprocess the data for model training."""
    data['Seat_Percentage'] = data['Seats Available'] / data['Seats Available'].sum()
    data['Rank'] = np.random.randint(1, TOTAL_APPLICANTS + 1, size=len(data))
    data['Rank_Scaled'] = 1 - (data['Rank'] - 1) / (TOTAL_APPLICANTS - 1)
    data['Selected'] = (data['Rank'] <= TOTAL_APPLICANTS // 2).astype(int)
    return data[['Rank_Scaled', 'Seat_Percentage']], data['Selected']

def train_model(X, y):
    """Train and save the XGBoost model."""
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        use_label_encoder=False
    )
    model.fit(X, y)
    joblib.dump(model, MODEL_FILE)
    return model

def load_and_train_model(csv_file):
    """Load data from CSV, preprocess, and retrain the model."""
    data = pd.read_csv(csv_file)
    X, y = preprocess_data(data)
    model = train_model(X, y)
    print(f"Model retrained using {csv_file}.")
    return model

def calculate_probability(rank, college, department, seats_df, model):
    """Calculate selection probability based on user input."""
    filtered_data = seats_df[(seats_df['College Name'] == college) & 
                             (seats_df['Department Name'] == department)]
    if filtered_data.empty:
        raise ValueError("Invalid college or department selection.")
    department_seats = filtered_data.iloc[0]['Seats Available']
    total_seats = seats_df['Seats Available'].sum()
    seat_percentage = department_seats / total_seats
    rank_scaled = 1 - (rank - 1) / (TOTAL_APPLICANTS - 1)
    candidate = pd.DataFrame({'Rank_Scaled': [rank_scaled], 'Seat_Percentage': [seat_percentage]})
    probability = model.predict_proba(candidate)[:, 1][0]
    return probability

class CSVHandler(FileSystemEventHandler):
    """Handle new CSV files."""
    def on_created(self, event):
        if event.src_path.endswith('.csv') and event.src_path not in PROCESSED_FILES:
            PROCESSED_FILES.add(event.src_path)
            print(f"New file detected: {event.src_path}")
            model = load_and_train_model(event.src_path)
            try:
                rank = int(input("\nEnter your rank: "))
                college = input("Enter the college name: ")
                department = input("Enter the department name: ")
                seats_df = pd.read_csv(event.src_path)
                probability = calculate_probability(rank, college, department, seats_df, model)
                print(f"Probability of selection in {department} at {college}: {probability * 100:.2f}%")
            except ValueError as e:
                print(e)

def monitor_csv_directory():
    """Monitor the directory for new CSV files."""
    observer = Observer()
    event_handler = CSVHandler()
    observer.schedule(event_handler, CSV_DIRECTORY, recursive=False)
    observer.start()
    print("Monitoring for new CSV files...")

    try:
        while True:
            time.sleep(1)  # Keep the script running
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

# Ensure the directory exists
if not os.path.exists(CSV_DIRECTORY):
    os.makedirs(CSV_DIRECTORY)

monitor_csv_directory()

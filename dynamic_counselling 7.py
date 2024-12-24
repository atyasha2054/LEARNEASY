import pandas as pd
import xgboost as xgb
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import time
import os

# Constants
TOTAL_APPLICANTS = 1000
MODEL_PATH = 'static_xgboost_college_model.pkl'
CSV_PATH = 'College_Seats_Data.csv'

class ModelHandler:
    def __init__(self, csv_path, model_path):
        self.csv_path = csv_path
        self.model_path = model_path
        self.model = None
        self.load_or_initialize_model()

    def load_or_initialize_model(self):
        try:
            self.model = joblib.load(self.model_path)
            print("Model loaded successfully.")
        except:
            print("No pre-trained model found. A new model will be trained when the CSV is updated.")
            self.model = None

    def retrain_model(self, affected_college=None):
        print("Retraining the model...")
        data = pd.read_csv(self.csv_path)
        
        # Filter only affected college if provided
        if affected_college:
            data = data[data['College Name'] == affected_college]

        data['Seat_Percentage'] = data['Seats Available'] / data['Seats Available'].sum()
        data['Rank'] = np.random.randint(1, TOTAL_APPLICANTS + 1, size=len(data))
        data['Rank_Scaled'] = 1 - (data['Rank'] - 1) / (TOTAL_APPLICANTS - 1)
        data['Selected'] = (data['Rank'] <= TOTAL_APPLICANTS // 2).astype(int)

        X = data[['Rank_Scaled', 'Seat_Percentage']]
        y = data['Selected']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss', use_label_encoder=False)
        self.model.fit(X_train, y_train)

        # Save the model
        joblib.dump(self.model, self.model_path)
        print("Model retrained and saved.")

    def predict(self, rank, college, department):
        data = pd.read_csv(self.csv_path)
        filtered_data = data[(data['College Name'] == college) & (data['Department Name'] == department)]

        if filtered_data.empty:
            raise ValueError("Invalid college or department.")

        department_seats = filtered_data.iloc[0]['Seats Available']
        total_seats = data['Seats Available'].sum()

        seat_percentage = department_seats / total_seats
        rank_scaled = 1 - (rank - 1) / (TOTAL_APPLICANTS - 1)

        candidate = pd.DataFrame({'Rank_Scaled': [rank_scaled], 'Seat_Percentage': [seat_percentage]})
        probability = self.model.predict_proba(candidate)[:, 1][0]
        return probability

class CSVHandler(FileSystemEventHandler):
    def __init__(self, model_handler):
        self.model_handler = model_handler

    def on_modified(self, event):
        if event.src_path.endswith('.csv'):
            print("CSV file modified. Retraining the model...")
            self.model_handler.retrain_model()  # Retrain the model
            self.model_handler.load_or_initialize_model()  # Ensure model reload
        """if event.src_path.endswith('.csv'):
            time.sleep(1)  # Ensure file changes are complete
            print("CSV file detected. Retraining the model...")
            self.model_handler.retrain_model()"""

    """def on_modified(self, event):
        print(f"Event detected: {event.src_path}")
        if event.src_path.endswith('.csv'):
            print("CSV file detected. Retraining the model...")
            self.model_handler.retrain_model()"""


# Main Script
if __name__ == "__main__":
    model_handler = ModelHandler(CSV_PATH, MODEL_PATH)

    # Watchdog observer for CSV changes
    event_handler = CSVHandler(model_handler)
    observer = Observer()
    observer.schedule(event_handler, path=os.path.dirname(os.path.abspath(CSV_PATH)), recursive=False)
    observer.start()

    print("Monitoring CSV file for changes. Press Ctrl+C to stop.")
    try:
        while True:
            # Simulate continuous prediction
            time.sleep(10)
            print("Awaiting predictions...")

            # Example user input
            rank = int(input("Enter rank (or press Ctrl+C to exit): "))
            college = input("Enter college name: ")
            department = input("Enter department name: ")

            try:
                probability = model_handler.predict(rank, college, department)
                print(f"Selection probability for {department} at {college}: {probability * 100:.2f}%")
            except ValueError as e:
                print(e)
    except KeyboardInterrupt:
        print("Stopping the system.")
        observer.stop()
    observer.join()
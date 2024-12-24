import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split

# Load the dataset
data = pd.read_csv("College_Seats_Data.csv")

# Feature Engineering: Seat Percentage
data['Seat_Percentage'] = data['Seats Available'] / data['Seats Available'].sum()

# Ensure the target column (`Selected`) is binary
median_seat_percentage = data['Seat_Percentage'].median()
data['Selected'] = (data['Seat_Percentage'] > median_seat_percentage).astype(int)

# Split into features and target
X = data[['Seat_Percentage']]
y = data['Selected']

# Ensure that `y` has both classes (0 and 1)
if len(y.unique()) < 2:
    raise ValueError("The target variable `Selected` must have both 0 and 1 classes for training.")

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train XGBoost Model
model = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss')
model.fit(X_train, y_train)

# Function to calculate probability based on user input
def calculate_probability(college, department, csv_file):
    # Load the data
    seats_df = pd.read_csv(csv_file)

    # Filter for the selected college and department
    filtered_data = seats_df[(seats_df['College Name'] == college) & 
                             (seats_df['Department Name'] == department)]

    if filtered_data.empty:
        raise ValueError("Invalid college or department selection.")

    # Extract seat information
    department_seats = filtered_data.iloc[0]['Seats Available']
    total_seats = seats_df['Seats Available'].sum()

    # Calculate seat percentage
    seat_percentage = department_seats / total_seats

    # Create a dataframe for the input
    candidate = pd.DataFrame({'Seat_Percentage': [seat_percentage]})

    # Predict probability
    probability = model.predict_proba(candidate)[:, 1][0]
    return probability

# User interaction
print("\n--- College Seat Probability Predictor ---")
user_college = input("Enter the college name: ")
user_department = input("Enter the department name: ")

# Predict and display the result
csv_file = "College_Seats_Data.csv"
try:
    selection_probability = calculate_probability(user_college, user_department, csv_file)
    print(f"Probability of selection in {user_department} at {user_college}: {selection_probability * 100:.2f}%")
except ValueError as e:
    print(e)

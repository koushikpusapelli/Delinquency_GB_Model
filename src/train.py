# filepath: c:\Users\pusap\OneDrive\Desktop\projects\Delinquency_GB_Model\src\train.py
print("train.py started")
# ...existing code...
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

# Load dataset
data_path = os.path.join("data", "Delinquency_prediction_dataset.csv")
df = pd.read_csv(data_path)

# Check dataset info
print("Dataset Shape:", df.shape)
print("Columns:", df.columns)

# Drop identifier column
df = df.drop("Customer_ID", axis=1)

# One-hot encode categorical columns
categorical_cols = [
    "Employment_Status", "Credit_Card_Type", "Location", "Missed_Payments",
    "Month_1", "Month_2", "Month_3", "Month_4", "Month_5", "Month_6"
]
df = pd.get_dummies(df, columns=categorical_cols)

# Fill any remaining NaNs with 0
df = df.fillna(0)

# Check for non-numeric columns
non_numeric = df.select_dtypes(include=['object']).columns
print("Non-numeric columns after encoding:", non_numeric)
# ...existing code...

# Separate features and target
X = df.drop("Delinquent_Account", axis=1)  # Replace 'Delinquent' with your target column name
y = df["Delinquent_Account"]

# Split into train/test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize Gradient Boosting Model
gb_model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)

# Train the model
gb_model.fit(X_train, y_train)

# Evaluate on test set
y_pred = gb_model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Save the trained model
os.makedirs("models", exist_ok=True)
joblib.dump(gb_model, "models/gradient_boosting_model.pkl")
print("Model saved in 'models/gradient_boosting_model.pkl'")

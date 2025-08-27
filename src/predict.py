import pandas as pd
import joblib
import os

# Load model
model_path = os.path.join("models", "gradient_boosting_model.pkl")
gb_model = joblib.load(model_path)

# Load external test data
external_data_path = os.path.join("data", "external_data_sample.csv")
new_data = pd.read_csv(external_data_path)

# Drop columns not used for prediction
drop_cols = ["Customer_ID", "Delinquent_Account"]
new_data = new_data.drop(columns=[col for col in drop_cols if col in new_data.columns])

# One-hot encode categorical columns (same as training)
categorical_cols = [
    "Employment_Status", "Credit_Card_Type", "Location", "Missed_Payments",
    "Month_1", "Month_2", "Month_3", "Month_4", "Month_5", "Month_6"
]
new_data = pd.get_dummies(new_data, columns=categorical_cols)

# Align columns with training data
model_features = gb_model.feature_names_in_
new_data = new_data.reindex(columns=model_features, fill_value=0)

# Fill any remaining NaNs
new_data = new_data.fillna(0)

# Predict
predictions = gb_model.predict(new_data)
print("Predictions for external data:", predictions)
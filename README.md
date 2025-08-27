# Delinquency Gradient Boosting Model

This project provides a machine learning solution for **delinquency prediction** using a Gradient Boosting Classifier. The model is designed to predict the likelihood of credit delinquency for customers based on their financial and demographic data.

## Overview

- **Purpose:** Predict credit risk and delinquency using customer data.
- **Model:** Gradient Boosting Classifier (scikit-learn)
- **Features:** Includes customer demographics, credit history, payment behavior, and more.
- **Input:** CSV files with customer data.
- **Output:** Predicted delinquency status for new/external data.

## How It Works

1. **Training:**  
   The model is trained on historical customer data, with categorical features one-hot encoded and missing values handled.
2. **Prediction:**  
   The trained model can be used to predict delinquency on new customer data, ensuring the same preprocessing steps are applied.

---

## Usage Instructions

### 1. Clone the Repository

```sh
git clone https://github.com/your-username/Delinquency_GB_Model.git
cd Delinquency_GB_Model
```

### 2. Set Up the Python Environment

It is recommended to use a virtual environment:

```sh
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate
```

### 3. Install Dependencies

```sh
pip install -r requirements.txt
```

### 4. Prepare Your Data

- Place your training data CSV in the `data/` folder (e.g., `Delinquency_prediction_dataset.csv`).
- Place any external data for prediction in the same folder (e.g., `external_data_sample.csv`).

### 5. Train the Model

```sh
python src/train.py
```
- This will train the Gradient Boosting model and save it to the `models/` directory.

### 6. Run Predictions

```sh
python src/predict.py
```
- This will load the trained model and output predictions for the external data sample.

---

## File Structure

```
Delinquency_GB_Model/
│
├── data/
│   ├── Delinquency_prediction_dataset.csv
│   └── external_data_sample.csv
├── models/
│   └── gradient_boosting_model.pkl
├── src/
│   ├── train.py
│   └── predict.py
├── requirements.txt
└── README.md
```

---

## Notes

- Ensure your input CSV files have the same columns as used in training.
- The code automatically handles categorical encoding and missing values.
- You can modify the model or features in `src/train.py` as needed.

---

## License

This project is provided for educational purposes. Please review and adapt for production use

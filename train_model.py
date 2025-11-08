import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import pickle
import logging
import sys
from sklearn.linear_model import LogisticRegression  # Using a faster model

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def train_model():
    try:
        # Load the data
        logging.info("Loading data...")
        df = pd.read_csv('KaggleV2-May-2016.csv', nrows=10000)  # Using subset for faster training
        
        # Process the data
        logging.info("Processing data...")
        df['Gender'] = df['Gender'].map({'M': 1, 'F': 0})
        df['WeekDay'] = pd.to_datetime(df['ScheduledDay']).dt.weekday
        df['DayScheduled'] = pd.to_datetime(df['ScheduledDay']).dt.day

        # Select features
        features = ['Gender', 'Age', 'Scholarship', 'Hipertension', 'Diabetes', 'Alcoholism', 
                   'Handcap', 'SMS_received', 'WeekDay', 'DayScheduled']
        X = df[features]
        y = df['No-show'].map({'Yes': 1, 'No': 0})

        # Scale the features
        logging.info("Scaling features...")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Train the model
        logging.info("Training model...")
        model = LogisticRegression(max_iter=1000)  # Using logistic regression instead of SVM
        model.fit(X_scaled, y)

        # Save both model and scaler
        logging.info("Saving model and scaler...")
        model_data = {
            'model': model,
            'scaler': scaler
        }
        with open('model.pkl', 'wb') as f:
            pickle.dump(model_data, f)

        logging.info("Model trained and saved successfully!")
        return True
    except Exception as e:
        logging.error(f"Error during model training: {str(e)}")
        return False

if __name__ == "__main__":
    train_model()
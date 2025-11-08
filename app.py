from flask import Flask, request, render_template, jsonify
import numpy as np
import pickle
import logging
import os

logging.basicConfig(level=logging.INFO)
app = Flask(__name__)

# Global variables for model and scaler
model = None
scaler = None

def load_model():
    global model, scaler
    try:
        if os.path.exists('model.pkl'):
            with open('model.pkl', 'rb') as f:
                model_data = pickle.load(f)
                model = model_data['model']
                scaler = model_data['scaler']
            logging.info("Model and scaler loaded successfully")
            return True
        else:
            logging.error("Model file not found - please run train_model.py first")
            return False
    except Exception as e:
        logging.error(f"Error loading model: {str(e)}")
        return False

# Load the model when starting the application
load_model()

@app.route('/')
def home():
    if model is None:
        return jsonify({"error": "Model not initialized. Please train the model first."}), 500
    return render_template('index.html')

@app.route('/health')
def health():
    return jsonify({"status": "healthy", "model_loaded": model is not None})

@app.route('/predict', methods=['POST'])
def predict():
    if model is None or scaler is None:
        return jsonify({"error": "Model not initialized"}), 500
        
    try:
        # Extract input from form
        gender = 1 if request.form['Gender'].lower() == 'male' else 0
        age = int(request.form['Age'])
        scholarship = int(request.form['Scholarship'])
        hipertension = int(request.form['Hypertension'])
        diabetes = int(request.form['Diabetes'])
        alcoholism = int(request.form['Alcoholism'])
        handcap = int(request.form['Handicap'])
        sms_received = int(request.form['SMS_received'])
        weekday = int(request.form['WeekDay'])
        day_scheduled = int(request.form['DayScheduled'])

        # Create feature array
        features = np.array([[gender, age, scholarship, hipertension, diabetes, 
                            alcoholism, handcap, sms_received, weekday, day_scheduled]])
        
        # Scale features
        features_scaled = scaler.transform(features)
        
        # Make prediction
        prediction = model.predict(features_scaled)[0]
        probability = model.predict_proba(features_scaled)[0]
        # Get prediction result
        result = "Doctor Will be available" if prediction == 0 else "Doctor Will not be available"
        confidence = f"{max(probability) * 100:.2f}%"

        return render_template('index.html', prediction=result, confidence=confidence)

    except Exception as e:
        logging.error(f"Prediction error: {str(e)}")
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    # Check if model is loaded
    if not os.path.exists('svm_model.pkl'):
        logging.info("Model not found. Training new model...")
        from train_model import train_model
        if not train_model():
            logging.error("Failed to train model. Exiting...")
            exit(1)
        load_model()
    
    # Run the application
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)

from flask import Flask, request, render_template
import numpy as np
import pickle

app = Flask(__name__)

with open("svm_model.pkl", "rb") as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract input from form
        gender = 1 if request.form['Gender'].lower() == 'male' else 0
        age = int(request.form['Age'])
        scholarship = int(request.form['Scholarship'])
        hypertension = int(request.form['Hypertension'])
        diabetes = int(request.form['Diabetes'])
        alcoholism = int(request.form['Alcoholism'])
        handicap = int(request.form['Handicap'])
        sms_received = int(request.form['SMS_received'])
        weekday = int(request.form['WeekDay'])
        day_scheduled = int(request.form['DayScheduled'])

        features = np.array([[gender, age, scholarship, hypertension, diabetes, alcoholism, handicap, sms_received, weekday, day_scheduled]])
        
        prediction = model.predict(features)[0]
        result = "Doctor Will be available" if prediction == 0 else "Doctor Will not be available"

        return render_template('index.html', prediction=result)

    except Exception as e:
        return render_template('index.html', prediction=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)

from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the model once
model = pickle.load(open('model/heart_disease_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from form
        age = float(request.form['age'])
        chest_pain = float(request.form['chest_pain'])
        heart_rate = float(request.form['heart_rate'])
        
        # Create feature array
        features = np.array([[age, chest_pain, heart_rate]])
        
        # Predict
        prediction = model.predict(features)[0]
        
        result_text = "High Risk of Heart Disease" if prediction == 1 else "Low Risk of Heart Disease"
        return render_template('result.html', prediction=result_text)
    except Exception as e:
        return f"Error Occurred: {e}"

if __name__ == '__main__':
    app.run(debug=True)

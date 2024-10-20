from flask import Flask, render_template, request, jsonify
import numpy as np
import pickle

app = Flask(__name__)

# Load the trained model
with open('home_loan_prediction_model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route("/")
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    credit_score = request.form.get('credit_score')
    income = request.form.get('income')
    debt_to_income_ratio = request.form.get('debt_to_income_ratio')
    down_payment = request.form.get('down_payment')

    # Prepare input for the model (in the correct order and format)
    input_data = np.array([[
        int(credit_score), 
        int(income), 
        float(debt_to_income_ratio), 
        int(down_payment)
    ]])

    # Make a prediction
    loan_approval_prediction = model.predict(input_data)[0]

    # Convert prediction to a string (Approved/Not Approved)
    prediction_result = "Approved" if loan_approval_prediction == 1 else "Not Approved"

    # Pass the form data and prediction back to the template
    return render_template("index.html", 
                           prediction=prediction_result,
                           credit_score=credit_score, 
                           income=income, 
                           debt_to_income_ratio=debt_to_income_ratio, 
                           down_payment=down_payment)

if __name__ == "__main__":
    app.run(debug=True)
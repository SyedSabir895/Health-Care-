from flask import Flask, render_template, request
import random
import pandas as pd
from transformers import pipeline

app = Flask(__name__)

# Initialize the Hugging Face model for diagnosis
diagnosis_model = pipeline("text-generation", model="gpt2")

# Synthetic Data Generator Function
def generate_synthetic_data(num_records):
    diseases = ["Diabetes", "Hypertension", "Asthma", "Heart Disease"]
    data = []
    for _ in range(num_records):
        record = {
            "PatientID": random.randint(1000, 9999),
            "Age": random.randint(18, 80),
            "Gender": random.choice(["Male", "Female"]),
            "Disease": random.choice(diseases),
            "Severity": random.choice(["Mild", "Moderate", "Severe"]),
        }
        data.append(record)
    return pd.DataFrame(data)

# Diagnosis Predictor Function
def predict_diagnosis(symptoms):
    prompt = f"Patient presents with symptoms: {symptoms}. Possible diagnosis:"
    prediction = diagnosis_model(prompt, max_length=50, num_return_sequences=1)
    return prediction[0]['generated_text']

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/generate", methods=["POST"])
def generate_data():
    num_records = int(request.form.get("num_records"))
    synthetic_data = generate_synthetic_data(num_records)
    synthetic_data_html = synthetic_data.to_html(classes='table table-striped', index=False)
    return render_template("result.html", result=synthetic_data_html, title="Synthetic Data")

@app.route("/diagnosis", methods=["POST"])
def diagnosis():
    symptoms = request.form.get("symptoms")
    diagnosis_result = predict_diagnosis(symptoms)
    return render_template("result.html", result=diagnosis_result, title="Diagnosis Prediction")

if __name__ == "__main__":
    app.run(debug=True)

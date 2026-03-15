from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import json
import numpy as np
import pandas as pd

app = Flask(__name__)
CORS(app)

# Load model assets
model = joblib.load('disease_model.pkl')
le = joblib.load('label_encoder.pkl')
with open('symptoms_list.json') as f:
    all_symptoms = json.load(f)

# Load disease descriptions
desc_df = pd.read_csv('symptom_Description.csv')
desc_df['Disease'] = desc_df['Disease'].str.strip()
disease_descriptions = dict(zip(desc_df['Disease'], desc_df['Description']))

# Fix name mismatches between model and description file
name_fixes = {
    "Dimorphic hemmorhoids(piles)": "Dimorphic hemorrhoids(piles)",
    "Diabetes ": "Diabetes",
    "Hypertension ": "Hypertension",
}

def get_description(disease_name):
    fixed = name_fixes.get(disease_name, disease_name)
    return disease_descriptions.get(fixed, disease_descriptions.get(disease_name, "No description available."))

def get_confidence_label(confidence):
    if confidence >= 50:
        return "Very Common"
    elif confidence >= 25:
        return "Common"
    elif confidence >= 10:
        return "Possible"
    else:
        return "Unlikely"

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok"})

@app.route('/symptoms', methods=['GET'])
def get_symptoms():
    return jsonify({"symptoms": all_symptoms})

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    selected_symptoms = [s.strip().lower() for s in data.get('symptoms', [])]
    age = int(data.get('age', 25))
    gender = data.get('gender', 'male').lower()

    if not selected_symptoms:
        return jsonify({"error": "No symptoms provided"}), 400

    # Build feature vector
    row = {sym: 1 if sym in selected_symptoms else 0 for sym in all_symptoms}
    X = pd.DataFrame([row])

    # Predict top 3
    proba = model.predict_proba(X)[0]
    top3 = np.argsort(proba)[::-1][:3]
    results = []
    for i in top3:
        disease_name = le.inverse_transform([i])[0].strip()
        confidence = round(float(proba[i]) * 100, 1)
        results.append({
            "disease": disease_name,
            "confidence": confidence,
            "label": get_confidence_label(confidence),
            "description": get_description(disease_name)
        })

    return jsonify({
        "prediction": results[0]["disease"],
        "confidence": results[0]["confidence"],
        "label": results[0]["label"],
        "description": results[0]["description"],
        "top3": results
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
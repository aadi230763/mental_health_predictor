import pickle
import numpy as np
import pandas as pd
import shap
from flask import Flask, request, render_template

# Load model and encoders
with open('app/model.pkl', 'rb') as f:
    model = pickle.load(f)
print("Model type loaded:", type(model))


with open('app/label_encoders.pkl', 'rb') as f:
    label_encoders = pickle.load(f)

# Feature names expected by the model
features = ['Age', 'family_history', 'work_interfere', 'benefits', 'care_options', 'leave', 'supervisor']

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    input_data = {}

    # Prepare the input from form
    for feature in features:
        val = request.form[feature]
        if feature in label_encoders:
            val = label_encoders[feature].transform([val])[0]
        else:
            val = int(val)
        input_data[feature] = val

    df = pd.DataFrame([input_data])

    # Make prediction
    prediction = model.predict(df)[0]
    try:
        probability = model.predict_proba(df)[0][1]
    except IndexError:
        probability = model.predict_proba(df)[0][0]

    # SHAP Explanation
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(df)

    # Normalize SHAP format: shap_values can be list or array depending on model type
    if isinstance(shap_values, list):
        if len(shap_values) > 1:
            shap_vals = shap_values[1][0]  # Class 1 values
        else:
            shap_vals = shap_values[0][0]  # Binary case fallback
    else:
        shap_vals = shap_values[0]

    # Convert to numpy array if not already
    shap_vals = np.array(shap_vals).flatten()

    # Pair features with SHAP values
    explanation = sorted(
        zip(df.columns.tolist(), shap_vals),
        key=lambda x: abs(x[1]),
        reverse=True
    )

    return render_template(
        'result.html',
        prediction='Yes' if prediction == 1 else 'No',
        probability=round(probability * 100, 2),
        explanation=explanation
    )

if __name__ == '__main__':
    app.run(debug=True)

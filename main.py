import pickle
import numpy as np
import pandas as pd
import shap
from flask import Flask, request, render_template

# Load model and label encoders
with open('app/model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('app/label_encoders.pkl', 'rb') as f:
    label_encoders = pickle.load(f)

# Define feature order
features = ['Age', 'family_history', 'work_interfere', 'benefits', 'care_options', 'leave', 'supervisor']

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    input_data = {}

# Collect and encode input data
    for feature in features:
        val = request.form[feature]
        if feature in label_encoders:
            val = label_encoders[feature].transform([val])[0]
        else:
            val = int(val)
        input_data[feature] = val

    df = pd.DataFrame([input_data])

# Make prediction
    prediction = model.predict(df)[0]                     # 0 or 1
    proba_arr = model.predict_proba(df)[0]                # [p(class_0), p(class_1)]
    class_index = list(model.classes_).index(prediction)  # Ensure correct index
    probability = proba_arr[class_index] * 100            # Correct confidence



 # SHAP explanation
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(df)  
        if isinstance(shap_values, list):
            if len(shap_values) > 1:
                shap_vals = shap_values[1][0]  # For binary class: use class 1
            else:
                shap_vals = shap_values[0][0]
        else:
            shap_vals = shap_values[0]
    except:
        explainer = shap.Explainer(model, df)
        shap_values = explainer(df)
        shap_vals = shap_values.values[0]

    # Flatten SHAP values
    shap_vals = np.array(shap_vals).flatten()

    # Sort explanations by magnitude
    explanation = sorted(
        zip(df.columns.tolist(), shap_vals),
        key=lambda x: abs(x[1]),
        reverse=True
    )

    return render_template(
        'result.html',
        prediction='Yes' if prediction == 1 else 'No',
        probability=round(probability, 2),
        explanation=explanation
    )

if __name__ == '__main__':
    app.run(debug=True)
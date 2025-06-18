import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# 1. Load & preprocess data
df = pd.read_csv('HealthSurvey.csv')
df.drop(['comments', 'state', 'Timestamp'], axis=1, inplace=True)
df = df[(df['Age'] >= 18) & (df['Age'] <= 65)]
df.fillna({'self_employed': 'No', 'work_interfere': "Don't know"}, inplace=True)

# 2. Select features & target
features = [
    'Age',
    'family_history',
    'work_interfere',
    'benefits',
    'care_options',
    'leave',
    'supervisor'
]
df = df[features + ['treatment']]

# 3. Encode categoricals
label_encoders = {}
for col in df.columns:
    if df[col].dtype == 'object':
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

X = df[features].values
y = df['treatment'].values

# 4. Define models
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, class_weight='balanced'),
    'Random Forest': RandomForestClassifier(n_estimators=200, class_weight='balanced'),
    'XGBoost': XGBClassifier(eval_metric='logloss')
}

# 5. Cross‑validation setup
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scoring = ['accuracy', 'precision', 'recall', 'f1']

# 6. Evaluate each model
cv_results = {}
for name, model in models.items():
    scores = cross_validate(model, X, y, cv=cv, scoring=scoring, return_train_score=False)
    cv_results[name] = {metric: np.mean(scores[f'test_{metric}']) for metric in scoring}

# 7. Display results
print("5‑Fold CV Results (mean scores):\n")
for name, metrics in cv_results.items():
    print(f"--- {name} ---")
    for m, v in metrics.items():
        print(f"{m.capitalize():>9}: {v:.4f}")
    print()

# 8. Select best model by F1
best_name = max(cv_results, key=lambda n: cv_results[n]['f1'])
best_model = models[best_name]
print(f"Selected Best Model: {best_name}\n")

# 9. Retrain on full data & save
best_model.fit(X, y)
import os
os.makedirs('app', exist_ok=True)
with open('app/model.pkl', 'wb') as f:
    pickle.dump(best_model, f)
with open('app/label_encoders.pkl', 'wb') as f:
    pickle.dump(label_encoders, f)

print("Model and encoders saved to app/model.pkl and app/label_encoders.pkl")

print(df['treatment'].value_counts(normalize=True))

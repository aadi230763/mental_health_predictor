# trained_models_cv.py  (updated)

import os
import pickle
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from xgboost import XGBClassifier

##############################################################################
# 1. Load & preprocess data
##############################################################################
df = pd.read_csv("HealthSurvey.csv")
df.drop(["comments", "state", "Timestamp"], axis=1, inplace=True)
df = df[(df["Age"] >= 18) & (df["Age"] <= 65)]
df.fillna({"self_employed": "No", "work_interfere": "Don't know"}, inplace=True)

features = [
    "Age",
    "family_history",
    "work_interfere",
    "benefits",
    "care_options",
    "leave",
    "supervisor",
]
 

# Encode categoricals
label_encoders = {}
for col in df.columns:
    if df[col].dtype == "object":
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

X, y = df[features].values, df["treatment"].values

##############################################################################
# 2. Define models
##############################################################################
models = {
    # ORIGINAL THREE
    "Logistic Regression": LogisticRegression(max_iter=1_000, class_weight="balanced"),
    "Random Forest": RandomForestClassifier(n_estimators=200, class_weight="balanced"),
    "XGBoost": XGBClassifier(eval_metric="logloss"),
    # NEW THREE
    "Linear Discriminant": LinearDiscriminantAnalysis(),
    "SVM (Linear)": SVC(kernel="linear", class_weight="balanced", probability=True),
    "Gradient Boosting": GradientBoostingClassifier(),
}
 
initial_models = {
    "Logistic Regression",
    "Random Forest",
    "XGBoost",
}
extra_models = set(models) - initial_models

##############################################################################
# 3. Cross‑validation
##############################################################################
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scoring = ["accuracy", "precision", "recall", "f1"]

cv_results = {}
for name, mdl in models.items():
    scores = cross_validate(mdl, X, y, cv=cv, scoring=scoring, return_train_score=False)
    cv_results[name] = {s: np.mean(scores[f"test_{s}"]) for s in scoring}

##############################################################################
# 4. Pick best model (accuracy criterion)
##############################################################################
best_initial = max(initial_models, key=lambda n: cv_results[n]["accuracy"])
best_extra = max(extra_models, key=lambda n: cv_results[n]["accuracy"])

if cv_results[best_extra]["accuracy"] > cv_results[best_initial]["accuracy"]:
    best_name = best_extra
else:
    best_name = best_initial

best_model = models[best_name]

##############################################################################
# 5. Display results
##############################################################################
print("\n★ 5‑Fold CV mean scores ★\n")
for name, metrics in cv_results.items():
    print(f"{name:>20} | "
          + " | ".join(f"{m}:{v:.3f}" for m, v in metrics.items()))
print("\nSelected Best Model (by accuracy):", best_name, "\n")

##############################################################################
# 6. Retrain on full data & save
##############################################################################
best_model.fit(X, y)

os.makedirs("app", exist_ok=True)
with open("app/model.pkl", "wb") as f:
    pickle.dump(best_model, f)
with open("app/label_encoders.pkl", "wb") as f:
    pickle.dump(label_encoders, f)

print("✔ Model and encoders saved to app/model.pkl and app/label_encoders.pkl")

best_accuracy = cv_results[best_name]["accuracy"]
print(f"Best model mean CV accuracy: {best_accuracy:.3f}")

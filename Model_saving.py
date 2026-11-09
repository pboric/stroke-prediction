"""Train the final stroke model and save it as stroke_model.pkl.

Uses the same pipeline as the notebook: imputation, scaling, one-hot encoding,
undersampling and SMOTE are steps inside one imblearn pipeline, so resampling
only ever touches the training data. Logistic regression was the best model on
the held-out test set (ROC AUC 0.84), so it is the one saved here, trained on
all available data.

Usage: python Model_saving.py [path/to/healthcare-dataset-stroke-data.csv]
"""

import sys

import joblib
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC = ["age", "avg_glucose_level", "bmi"]
BINARY = ["hypertension", "heart_disease"]
CATEGORICAL = ["gender", "ever_married", "work_type", "Residence_type", "smoking_status"]


def build_model() -> ImbPipeline:
    preprocess = ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ("bin", "passthrough", BINARY),
    ])
    return ImbPipeline([
        ("prep", preprocess),
        ("under", RandomUnderSampler(sampling_strategy=0.1, random_state=42)),
        ("over", SMOTE(sampling_strategy=1.0, random_state=42)),
        ("classifier", LogisticRegression(max_iter=1000)),
    ])


def main() -> None:
    path = sys.argv[1] if len(sys.argv) > 1 else "healthcare-dataset-stroke-data.csv"
    data = pd.read_csv(path)
    X = data[NUMERIC + BINARY + CATEGORICAL]
    y = data["stroke"]

    model = build_model()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")
    print(f"5-fold cross-validated ROC AUC: {scores.mean():.3f} (± {scores.std():.3f})")

    model.fit(X, y)
    joblib.dump(model, "stroke_model.pkl")
    print("Model saved as 'stroke_model.pkl'")


if __name__ == "__main__":
    main()

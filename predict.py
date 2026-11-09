"""Predict stroke risk for new patients with the model saved by Model_saving.py.

The input CSV needs the same columns as the training data (gender, age,
hypertension, heart_disease, ever_married, work_type, Residence_type,
avg_glucose_level, bmi, smoking_status); id and stroke are ignored if present.
All encoding and scaling happen inside the saved pipeline.

Usage: python predict.py patients.csv [threshold]
"""

import sys

import joblib
import pandas as pd

from Model_saving import BINARY, CATEGORICAL, NUMERIC


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit("Usage: python predict.py patients.csv [threshold]")
    threshold = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
    patients = pd.read_csv(sys.argv[1])
    model = joblib.load("stroke_model.pkl")

    risk = model.predict_proba(patients[NUMERIC + BINARY + CATEGORICAL])[:, 1]
    out = pd.DataFrame({"stroke_risk": risk.round(3), "flagged": (risk >= threshold).astype(int)})
    if "id" in patients:
        out.insert(0, "id", patients["id"])
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()

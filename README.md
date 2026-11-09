# Stroke Prediction

Predicting whether a patient is likely to have a stroke from age, health conditions and lifestyle data,
with exploratory analysis, statistical tests and machine learning on a heavily imbalanced dataset
(249 strokes among 5,110 patients, about 1:19).

## Results

Measured on a held-out test set of 1,022 patients (50 strokes) that the models never saw and that was
never resampled:

| Model | ROC AUC | PR AUC | Recall (stroke) | Precision (stroke) |
|---|---:|---:|---:|---:|
| **Logistic regression** | **0.84** | **0.29** | 0.80 | 0.14 |
| XGBoost (tuned) | 0.82 | 0.19 | 0.84 | 0.12 |
| Random Forest (tuned) | 0.80 | 0.19 | 0.46 | 0.15 |
| Ensemble (XGBoost + Random Forest) | 0.82 | 0.21 | 0.74 | 0.15 |

Recall and precision are at a threshold of 0.5. A random guess gives a ROC AUC of 0.5 and a PR AUC of
0.049 (the share of strokes). With only 50 strokes in the test set the uncertainty is large: the
ensemble's 95 % bootstrap interval is 0.76–0.87.

**A note on an earlier version.** The first version of this project applied undersampling and SMOTE to
the whole dataset before splitting it into training and test sets. Synthetic patients built from test
cases then ended up in the training data, and the test set was artificially balanced, which gave a
misleading ROC AUC of 98 %. The notebook now keeps all resampling inside the training pipeline and shows
the inflated score next to the correct one.

## Key findings

- **Age** is by far the strongest predictor: the stroke rate grows from under 1 % below 50 to 18 % above 70.
- **Hypertension**, **heart disease** and high **average glucose** each roughly triple the stroke rate.
- Rates, not counts, are what matter: most strokes occur among people without hypertension only because
  that group is much larger.
- Chi-square tests confirm higher stroke rates for people older than 50 and for people with BMI > 25
  (p < 0.001), although BMI is partly confounded with age.

## What the notebook covers

1. **Exploratory data analysis**: distributions, stroke rates by group, Spearman correlations, outliers.
2. **Statistical inference**: chi-square tests with 95 % confidence intervals for two hypotheses.
3. **Feature scores**: mutual information, chi-squared and ANOVA, computed on the training split only.
4. **Modelling**: one `imblearn` pipeline (imputation, scaling, one-hot encoding, undersampling, SMOTE,
   classifier), tuned with stratified 5-fold cross-validation; XGBoost, Random Forest, their soft-voting
   ensemble and a logistic regression baseline.

## Run it

1. Download `healthcare-dataset-stroke-data.csv` from the
   [Stroke Prediction Dataset on Kaggle](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset)
   and put it in this folder.
2. Install the requirements: `pip install -r requirements.txt`
3. Open `Stroke_Prediction_EDA_Modeling.ipynb` and run all cells (about a minute).

To train and use the final model outside the notebook:

```bash
python Model_saving.py                      # cross-validates and saves stroke_model.pkl
python predict.py new_patients.csv 0.5      # stroke risk per patient, flagged above the threshold
```

The dataset is small and imbalanced, so this is a learning project, not a clinical tool.

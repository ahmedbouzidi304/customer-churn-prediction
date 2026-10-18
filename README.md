# Customer Churn Prediction · Telecom

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ahmedbouzidi304/customer-churn-prediction/blob/main/churn_prediction.ipynb)
![Python](https://img.shields.io/badge/Python-3.11-blue) ![scikit-learn](https://img.shields.io/badge/scikit--learn-pipeline-orange) ![LightGBM](https://img.shields.io/badge/LightGBM-Optuna-green) ![SHAP](https://img.shields.io/badge/SHAP-explainability-purple)

End-to-end machine-learning project that predicts which telecom subscribers will cancel, explains **why**, and turns the scores into a **profit-optimal retention policy**.

## Business question
The retention team can contact only a limited number of customers. They need:
1. a ranked list of customers most likely to churn,
2. the drivers of each customer's risk (to choose the right offer),
3. a decision rule (threshold) that maximises campaign profit.

## Pipeline
| Step | What is done |
|---|---|
| 1. Audit | types, blanks, duplicates; domain-correct fix of `TotalCharges` for new customers |
| 2. EDA | churn by contract, service, payment method and tenure |
| 3. Feature engineering | 8 domain features (services count, price increase, auto-pay, protection gap, lifecycle stage...) |
| 4. Modelling | leak-free `Pipeline` + `ColumnTransformer`, 5-fold stratified CV of 5 models |
| 5. Tuning | Optuna (TPE) search over LightGBM; soft-voting ensemble candidate |
| 6. Evaluation | held-out test set: ROC-AUC, PR-AUC, calibration, classification report |
| 7. Business decision | threshold chosen on out-of-fold profit; lift by risk decile |
| 8. Explainability | SHAP beeswarm, dependence plots, per-customer waterfall |
| 9. Packaging | `joblib` artifact + `predict_churn()` inference function |

## Results
<!-- RESULTS -->
_Run the notebook to reproduce: results are printed in the final section._

## Run it
Click **Open in Colab** above, then *Runtime → Run all* (CPU is enough, about 10 minutes).
Locally: `pip install -r requirements.txt` and open the notebook in Jupyter.

## Data
[IBM Telco Customer Churn](https://github.com/IBM/telco-customer-churn-on-icp4d) (7,043 customers, 21 columns), downloaded automatically.

## Tech stack
pandas · scikit-learn · LightGBM · XGBoost · Optuna · SHAP · matplotlib/seaborn

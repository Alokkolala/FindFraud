# FindFraud

Transaction anomaly detection pipeline that blends machine learning and configurable rules with explainability and reporting.

## Project structure

```
src/findfraud/
├── data_loader.py      # Schema-aware CSV ingestion
├── features.py         # Feature engineering for tabular fraud dataset
├── model.py            # IsolationForest training and SHAP explainability
├── rules.py            # Configurable heuristic rules
├── scorer.py           # Combined scoring pipeline
├── report.py           # HTML/PDF reporting
└── cli.py              # Command-line entrypoint
```

## Supported CSV schema

The pipeline is tailored to the common "PaySim"-style transaction schema. Required columns:

- `step` (int): time step of the transaction
- `type` (string): transaction type (e.g., `PAYMENT`, `TRANSFER`, `CASH_OUT`)
- `amount` (float)
- `nameOrig` (string): source account/customer
- `oldbalanceOrg` (float)
- `newbalanceOrig` (float)
- `nameDest` (string): destination account
- `oldbalanceDest` (float)
- `newbalanceDest` (float)

Optional columns (used when present):

- `isFraud`, `isFlaggedFraud` — retained for completeness but not required for scoring
- `transaction_id` — if absent, one is synthesized as `"{nameOrig}_{step}"`

Example row:

```
step,type,amount,nameOrig,oldbalanceOrg,newbalanceOrig,nameDest,oldbalanceDest,newbalanceDest,isFraud,isFlaggedFraud
1,PAYMENT,9839.64,C1231006815,170136.0,160296.36,M1979787155,0.0,0.0,0,0
```

## Setup

1. Install Python 3.10+.
2. Install dependencies:

```bash
pip install -e .
```

The editable install exposes `findfraud` as a package and `python -m findfraud.cli` as a CLI.

## Usage

### Train a model

```bash
python -m findfraud.cli train data/transactions.csv models/anomaly.joblib
```

This fits feature encoders, trains an `IsolationForest`, stores SHAP background data, and saves everything to `models/anomaly.joblib`.

### Score new transactions

```bash
python -m findfraud.cli score data/new_transactions.csv models/anomaly.joblib outputs/scores.csv \
  --html-report outputs/report.html --pdf-report outputs/report.pdf
```

The scorer outputs `transaction_id`, `fraud_score`, `is_suspicious`, and `explanation` columns. HTML reporting is always produced when requested; PDF generation will occur if `weasyprint` is installed.

### Explainability

The anomaly detector computes feature-level attributions using SHAP over the model decision function and combines them with triggered rules to produce human-readable per-transaction explanations.

## Development

- Code lives under `src/findfraud/` with dependencies: pandas, numpy, scikit-learn, shap, and (optionally) weasyprint for PDF output.
- No automated tests are provided; run the CLI locally to exercise the pipeline.

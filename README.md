# FindFraud

Transaction anomaly detection pipeline that blends machine learning and configurable rules with explainability and reporting.

## Project structure

```
src/findfraud/
├── data_loader.py      # Schema-aware CSV ingestion
├── features.py         # Feature engineering for tabular fraud dataset
├── model.py            # IsolationForest training and SHAP explainability
├── graph_builder.py    # Graph construction and aggregation utilities
├── graph_model.py      # Graph neural network (GraphSAGE) training/inference
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

For graph-based scoring, install the optional dependencies:

```bash
pip install -e .[graph]
```

On Windows, use the bundled launcher:

```powershell
py -m pip install -e .[graph]
```

## Usage

### Train a model

```bash
python -m findfraud.cli train data/transactions.csv models/anomaly.joblib
```

This fits feature encoders, trains an `IsolationForest`, stores SHAP background data, and saves everything to `models/anomaly.joblib`.

Train a graph neural network instead of the tabular model:

```bash
python -m findfraud.cli train data/transactions.csv models/gnn.pt \
  --model-type gnn --graph-artifacts outputs/graph.pt --window-size 24 --min-edge-count 2 \
  --gnn-hidden 64 --gnn-layers 2 --gnn-epochs 50
```

Windows (PowerShell) equivalents without line continuations:

```powershell
py -m findfraud.cli train data\transactions.csv models\anomaly.joblib
py -m findfraud.cli train data\transactions.csv models\gnn.pt --model-type gnn --graph-artifacts outputs\graph.pt --window-size 24 --min-edge-count 2 --gnn-hidden 64 --gnn-layers 2 --gnn-epochs 50
```

This builds an account-to-account interaction graph with rolling window aggregations, trains a GraphSAGE model, and optionally
persists the serialized graph snapshot alongside the model weights.

If you see a saved graph with many nodes but zero edges, lower `--min-edge-count` (or leave it at the default `1`). When every
edge would be filtered out, the builder now relaxes the threshold to `1` automatically so sparse networks still render in
visualizations.

### Score new transactions

```bash
python -m findfraud.cli score data/new_transactions.csv models/anomaly.joblib outputs/scores.csv \
  --html-report outputs/report.html --pdf-report outputs/report.pdf --profiles-csv outputs/profiles.csv
```

The scorer outputs `transaction_id`, `fraud_score`, `is_suspicious`, and `explanation` columns. When `--html-report` is supplied, the generated file consolidates everything in one place: model/training metadata, the combined-score transaction table, top suspicious rows, the optional graph snapshot summary, and any account profile classifications. PDF generation will occur if `weasyprint` is installed.

Each profile row summarizes an account’s combined role as sender/receiver with the maximum and mean transaction score, total transaction/amount volume, and a `risk_level` bucket (`high_risk`, `elevated`, `low_risk`). Accounts that ever touched a flagged transaction or reached a score ≥ 0.8 are marked `high_risk`.

To score with a trained GNN while keeping a copy of the reconstructed graph:

```bash
python -m findfraud.cli score data/new_transactions.csv models/gnn.pt outputs/graph_scores.csv \
  --model-type gnn --graph-artifacts outputs/scored_graph.pt
```

Windows (PowerShell) equivalents:

```powershell
py -m findfraud.cli score data\new_transactions.csv models\anomaly.joblib outputs\scores.csv --html-report outputs\report.html --pdf-report outputs\report.pdf --profiles-csv outputs\profiles.csv
py -m findfraud.cli score data\new_transactions.csv models\gnn.pt outputs\graph_scores.csv --model-type gnn --graph-artifacts outputs\scored_graph.pt --html-report outputs\graph_report.html --profiles-csv outputs\graph_profiles.csv
```

### Serve the model as an API

The project ships a FastAPI app (`findfraud.api:app`) that exposes two endpoints:

- `POST /detect_fraud` — JSON payload for a single transaction; returns `transaction_id`, `fraud_score`, `is_suspicious`, and `explanation`.
- `POST /upload_csv` — multipart CSV upload; returns a CSV with the same columns as above for each row.

Start the API locally (Linux/macOS):

```bash
FINDFRAUD_MODEL_PATH=models/anomaly.joblib uvicorn findfraud.api:app --host 0.0.0.0 --port 8000
```

Windows (PowerShell):

```powershell
$env:FINDFRAUD_MODEL_PATH="models\anomaly.joblib"
py -m uvicorn findfraud.api:app --host 0.0.0.0 --port 8000
```

Expose the API to other machines or browsers by enabling CORS and binding to all interfaces:

```bash
FINDFRAUD_MODEL_PATH=models/anomaly.joblib \
FINDFRAUD_CORS_ORIGINS="https://my-frontend.example.com,http://localhost:3000" \
uvicorn findfraud.api:app --host 0.0.0.0 --port 8000
```

```powershell
$env:FINDFRAUD_MODEL_PATH="models\anomaly.joblib"
$env:FINDFRAUD_CORS_ORIGINS="https://my-frontend.example.com,http://localhost:3000"
py -m uvicorn findfraud.api:app --host 0.0.0.0 --port 8000
```

Notes for cross-site access:

- `FINDFRAUD_CORS_ORIGINS` accepts a comma-separated list of allowed origins (e.g., the URL of your web app). Use `*` to allow any origin during testing.
- Ensure the host machine’s firewall allows inbound traffic to the chosen port (8000 by default), and configure router/NAT port forwarding if you need access from the public internet.
- When running behind a reverse proxy or tunneling tool (ngrok, cloudflared, SSH remote forwarding), point it at `http://localhost:8000` and share the public URL with your frontend.

Example requests against a running server:

```bash
curl -X POST http://localhost:8000/detect_fraud \
  -H "Content-Type: application/json" \
  -d '{"step": 1, "type": "PAYMENT", "amount": 1200, "nameOrig": "C1", "oldbalanceOrg": 5000, "newbalanceOrig": 3800, "nameDest": "C2", "oldbalanceDest": 0, "newbalanceDest": 1200}'

curl -X POST http://localhost:8000/upload_csv \
  -F "file=@data/new_transactions.csv" -o scored_transactions.csv
```

PowerShell equivalents with Windows-style paths:

```powershell
Invoke-RestMethod -Method Post -Uri http://localhost:8000/detect_fraud -ContentType "application/json" -Body '{"step":1,"type":"PAYMENT","amount":1200,"nameOrig":"C1","oldbalanceOrg":5000,"newbalanceOrig":3800,"nameDest":"C2","oldbalanceDest":0,"newbalanceDest":1200}'

Invoke-RestMethod -Method Post -Uri http://localhost:8000/upload_csv -Form @{file=Get-Item "data\new_transactions.csv"} -OutFile scored_transactions.csv
```

Deploying to Heroku or any WSGI/ASGI-friendly service can use Gunicorn:

```bash
gunicorn -k uvicorn.workers.UvicornWorker findfraud.api:app --bind 0.0.0.0:$PORT
```

For AWS Lambda with API Gateway, install `mangum` and point the handler at `findfraud.api:handler`.

### Explainability

The anomaly detector computes feature-level attributions using SHAP over the model decision function and combines them with triggered rules to produce human-readable per-transaction explanations.

### Visualize saved graphs

You can explore the serialized graph artifacts (`graph.pt`) with NetworkX and Matplotlib (install them with `pip install networkx matplotlib` if needed):

```python
import networkx as nx
import matplotlib.pyplot as plt
from findfraud.graph_model import GraphModelTrainer

trainer = GraphModelTrainer()
artifacts = trainer.load_artifacts("outputs/graph.pt")
G = artifacts.to_networkx(
    min_txn_count=3,       # hide edges with fewer than 3 transfers
    min_total_amount=1e5,  # hide edges that moved less than 100k total
    top_n_nodes=50,        # keep only the busiest 50 accounts to reduce clutter
)

pos = nx.spring_layout(G, seed=0, k=0.3)
edge_amounts = [G.edges[e]["edge_attr"][0] for e in G.edges]
nx.draw_networkx_nodes(G, pos, node_size=80)
nx.draw_networkx_edges(G, pos, alpha=0.3, edge_color=edge_amounts, edge_cmap=plt.cm.Blues)
nx.draw_networkx_labels(G, pos, labels={n: G.nodes[n]["account_name"] for n in G.nodes}, font_size=6)
plt.tight_layout()
plt.show()
```

If you prefer to color by transaction volume, the edge attribute order is `[total_amount, txn_count, mean_step_delta, last_seen_step]`, so you can access `G.edges[u, v]["edge_attr"][0]` for the summed amount when styling edges.

## Development

- Code lives under `src/findfraud/` with dependencies: pandas, numpy, scikit-learn, shap, and (optionally) weasyprint for PDF output.
- No automated tests are provided; run the CLI locally to exercise the pipeline.

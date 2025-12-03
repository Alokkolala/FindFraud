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

### Score new transactions

```bash
python -m findfraud.cli score data/new_transactions.csv models/anomaly.joblib outputs/scores.csv \
  --html-report outputs/report.html --pdf-report outputs/report.pdf
```

The scorer outputs `transaction_id`, `fraud_score`, `is_suspicious`, and `explanation` columns. HTML reporting is always produced when requested; PDF generation will occur if `weasyprint` is installed.

To score with a trained GNN while keeping a copy of the reconstructed graph:

```bash
python -m findfraud.cli score data/new_transactions.csv models/gnn.pt outputs/graph_scores.csv \
  --model-type gnn --graph-artifacts outputs/scored_graph.pt
```

Windows (PowerShell) equivalents:

```powershell
py -m findfraud.cli score data\new_transactions.csv models\anomaly.joblib outputs\scores.csv --html-report outputs\report.html --pdf-report outputs\report.pdf
py -m findfraud.cli score data\new_transactions.csv models\gnn.pt outputs\graph_scores.csv --model-type gnn --graph-artifacts outputs\scored_graph.pt
```

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
G = artifacts.to_networkx()

pos = nx.spring_layout(G, seed=0)
nx.draw_networkx_nodes(G, pos, node_size=80)
nx.draw_networkx_edges(G, pos, alpha=0.2)
nx.draw_networkx_labels(G, pos, labels={n: G.nodes[n]["account_name"] for n in G.nodes}, font_size=6)
plt.tight_layout()
plt.show()
```

If you prefer to color by transaction volume, the edge attribute order is `[total_amount, txn_count, mean_step_delta, last_seen_step]`, so you can access `G.edges[u, v]["edge_attr"][0]` for the summed amount when styling edges.

## Development

- Code lives under `src/findfraud/` with dependencies: pandas, numpy, scikit-learn, shap, and (optionally) weasyprint for PDF output.
- No automated tests are provided; run the CLI locally to exercise the pipeline.

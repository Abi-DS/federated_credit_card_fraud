# Federated Credit Card Fraud Detection (Flower + Streamlit)

This project demonstrates privacy-preserving federated learning for credit card fraud detection using Flower (FedAvg) and PyTorch, with a Streamlit UI.

## Quickstart

1. Install dependencies (recommended in a virtual environment):

```bash
pip install -r requirements.txt
```

2. Launch the app:

```bash
streamlit run streamlit_app.py
```

3. In the UI, set the number of clients and rounds, then click "Run Federated Simulation". Optionally train a centralized baseline for comparison.

## What it shows

- Federated setup: multiple simulated clients train locally; only model weights are aggregated (no raw data sharing).
- Severe class imbalance handling via class-weighted loss.
- Global ROC-AUC/AP evolving across federated rounds.
- Centralized baseline ROC/PR curves and confusion matrix for comparison.

## Notes

- Dataset downloads automatically on first run (creditcard.csv).
- All clients are simulated on a single machine for demonstration.



import os
import urllib.request
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve, confusion_matrix

import flwr as fl
from collections import OrderedDict


# -----------------------------
# Data loading and preparation
# -----------------------------

DATA_URL = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"
DATA_PATH = os.path.join(os.getcwd(), "creditcard.csv")


@st.cache_data(show_spinner=False)
def load_dataset() -> pd.DataFrame:
    if not os.path.exists(DATA_PATH):
        urllib.request.urlretrieve(DATA_URL, DATA_PATH)
    return pd.read_csv(DATA_PATH)


def preprocess_and_partition(raw: pd.DataFrame, num_clients: int, seed: int = 42):
    X = raw.drop(columns=["Class"]).values.astype(np.float32)
    y = raw["Class"].values.astype(np.int64)

    scaler = StandardScaler()
    X = scaler.fit_transform(X).astype(np.float32)

    X_train_full, X_test_global, y_train_full, y_test_global = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    classes = np.unique(y_train_full)
    class_weights = compute_class_weight(
        class_weight="balanced", classes=classes, y=y_train_full
    )
    class_weight_tensor = {int(c): float(w) for c, w in zip(classes, class_weights)}

    # Stratified split into clients
    np.random.seed(seed)
    client_indices = [[] for _ in range(num_clients)]
    for cls in classes:
        cls_idx = np.where(y_train_full == cls)[0]
        np.random.shuffle(cls_idx)
        splits = np.array_split(cls_idx, num_clients)
        for i in range(num_clients):
            client_indices[i].extend(splits[i].tolist())
    for i in range(num_clients):
        rng = np.random.default_rng(seed=100 + i)
        rng.shuffle(client_indices[i])

    clients_data = []
    for i in range(num_clients):
        ci = np.array(client_indices[i])
        Xc, yc = X_train_full[ci], y_train_full[ci]
        X_train_c, X_val_c, y_train_c, y_val_c = train_test_split(
            Xc, yc, test_size=0.2, random_state=42, stratify=yc
        )
        clients_data.append({
            "train": (X_train_c, y_train_c),
            "val": (X_val_c, y_val_c),
        })

    return {
        "X_test_global": X_test_global,
        "y_test_global": y_test_global,
        "clients_data": clients_data,
        "class_weight_tensor": class_weight_tensor,
        "input_dim": X.shape[1],
        "class_ratio": raw["Class"].value_counts(normalize=True).to_dict(),
    }


# -----------------------------
# Model and training utilities
# -----------------------------

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class MLP(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)


def make_loader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    X_t = torch.from_numpy(X)
    y_t = torch.from_numpy(y.astype(np.float32))
    ds = TensorDataset(X_t, y_t)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, pos_weight: float) -> float:
    model.train()
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight], dtype=torch.float32, device=DEVICE)
    )
    epoch_loss = 0.0
    for xb, yb in loader:
        xb = xb.to(DEVICE)
        yb = yb.to(DEVICE)
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * xb.size(0)
    return epoch_loss / max(1, len(loader.dataset))


def evaluate(model: nn.Module, loader: DataLoader) -> Dict[str, float]:
    model.eval()
    all_logits, all_y = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(DEVICE)
            logits = model(xb)
            all_logits.append(logits.cpu())
            all_y.append(yb)
    logits = torch.cat(all_logits).numpy()
    y_true = torch.cat(all_y).numpy()
    y_prob = 1.0 / (1.0 + np.exp(-logits))
    try:
        roc = roc_auc_score(y_true, y_prob)
    except Exception:
        roc = float("nan")
    try:
        apr = average_precision_score(y_true, y_prob)
    except Exception:
        apr = float("nan")
    y_pred = (y_prob >= 0.5).astype(np.int64)
    acc = (y_pred == y_true).mean() if len(y_true) else float("nan")
    return {"roc_auc": float(roc), "avg_precision": float(apr), "accuracy": float(acc)}


# -----------------------------
# Flower client and simulation
# -----------------------------

def get_parameters(model: nn.Module):
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def set_parameters(model: nn.Module, parameters) -> None:
    state_dict = model.state_dict()
    new_state_dict = OrderedDict()
    for (k, _), v in zip(state_dict.items(), parameters):
        new_state_dict[k] = torch.tensor(v)
    model.load_state_dict(new_state_dict, strict=True)


def _fedavg(params_and_sizes):
    # params_and_sizes: List[ (List[np.ndarray], int) ]
    num_total = sum(n for _, n in params_and_sizes)
    num_layers = len(params_and_sizes[0][0])
    agg = []
    for li in range(num_layers):
        weighted = None
        for params, n in params_and_sizes:
            w = params[li]
            if weighted is None:
                weighted = w * (n / num_total)
            else:
                weighted = weighted + w * (n / num_total)
        agg.append(weighted)
    return agg


def run_fl_simulation(artifact: dict, rounds: int, local_epochs: int, train_bs: int, val_bs: int):
    input_dim = artifact["input_dim"]
    pos_weight = artifact["class_weight_tensor"].get(1, 1.0)
    X_test, y_test = artifact["X_test_global"], artifact["y_test_global"]
    test_loader = make_loader(X_test, y_test, batch_size=2048, shuffle=False)

    metrics_over_rounds: List[Tuple[int, Dict[str, float]]] = []

    class FraudClient(fl.client.NumPyClient):
        def __init__(self, train_set, val_set):
            self.model = MLP(input_dim).to(DEVICE)
            self.train_loader = make_loader(train_set[0], train_set[1], batch_size=train_bs, shuffle=True)
            self.val_loader = make_loader(val_set[0], val_set[1], batch_size=val_bs, shuffle=False)
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)

        def get_parameters(self, config):
            return get_parameters(self.model)

        def fit(self, parameters, config):
            set_parameters(self.model, parameters)
            ep = int(config.get("local_epochs", local_epochs))
            for _ in range(ep):
                train_one_epoch(self.model, self.train_loader, self.optimizer, pos_weight)
            m = evaluate(self.model, self.val_loader)
            return get_parameters(self.model), len(self.train_loader.dataset), m

        def evaluate(self, parameters, config):
            set_parameters(self.model, parameters)
            m = evaluate(self.model, self.val_loader)
            loss = 1.0 - (m.get("roc_auc") or 0.0)
            return float(loss), len(self.val_loader.dataset), m

    def client_fn(cid: str):
        idx = int(cid)
        data = artifact["clients_data"][idx]
        return FraudClient(data["train"], data["val"]).to_client()

    def get_evaluate_fn():
        def evaluate_fn(server_round: int, parameters, config):
            model = MLP(input_dim).to(DEVICE)
            set_parameters(model, parameters)
            m = evaluate(model, test_loader)
            metrics_over_rounds.append((server_round, m))
            return float(1.0 - (m.get("roc_auc") or 0.0)), m
        return evaluate_fn

    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=len(artifact["clients_data"]),
        min_evaluate_clients=len(artifact["clients_data"]),
        min_available_clients=len(artifact["clients_data"]),
        evaluate_fn=get_evaluate_fn(),
        on_fit_config_fn=lambda rnd: {"local_epochs": local_epochs},
    )

    # Try Ray-backed simulation first; if Ray is unavailable, fall back to a
    # lightweight in-process FedAvg loop (no Ray required).
    try:
        history = fl.simulation.start_simulation(
            client_fn=client_fn,
            num_clients=len(artifact["clients_data"]),
            config=fl.server.ServerConfig(num_rounds=rounds),
            strategy=strategy,
        )
        return metrics_over_rounds
    except ImportError:
        # Fallback: manual FedAvg loop
        input_dim_local = input_dim
        pos_w = pos_weight

        # Initialize global parameters with a fresh model
        global_model = MLP(input_dim_local).to(DEVICE)
        global_params = get_parameters(global_model)

        for r in range(1, rounds + 1):
            client_results = []
            for data in artifact["clients_data"]:
                # Local training
                model_c = MLP(input_dim_local).to(DEVICE)
                set_parameters(model_c, global_params)
                opt = torch.optim.Adam(model_c.parameters(), lr=1e-3)
                train_loader_c = make_loader(data["train"][0], data["train"][1], batch_size=train_bs, shuffle=True)
                val_loader_c = make_loader(data["val"][0], data["val"][1], batch_size=val_bs, shuffle=False)
                for _ in range(local_epochs):
                    train_one_epoch(model_c, train_loader_c, opt, pos_w)
                params_c = get_parameters(model_c)
                num_examples = len(train_loader_c.dataset)
                client_results.append((params_c, num_examples))

            # Aggregate
            global_params = _fedavg(client_results)

            # Evaluate global model
            set_parameters(global_model, global_params)
            m = evaluate(global_model, test_loader)
            metrics_over_rounds.append((r, m))

        return metrics_over_rounds


# -----------------------------
# Centralized baseline
# -----------------------------

def train_central_baseline(artifact: dict, epochs: int, batch_size: int):
    input_dim = artifact["input_dim"]
    pos_weight = artifact["class_weight_tensor"].get(1, 1.0)
    X_test, y_test = artifact["X_test_global"], artifact["y_test_global"]
    test_loader = make_loader(X_test, y_test, batch_size=2048, shuffle=False)

    X_train_merged = np.concatenate([c["train"][0] for c in artifact["clients_data"]], axis=0)
    y_train_merged = np.concatenate([c["train"][1] for c in artifact["clients_data"]], axis=0)
    train_loader = make_loader(X_train_merged, y_train_merged, batch_size=batch_size, shuffle=True)

    model = MLP(input_dim).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    hist = []
    for ep in range(epochs):
        loss = train_one_epoch(model, train_loader, opt, pos_weight)
        m = evaluate(model, test_loader)
        hist.append((ep + 1, loss, m))
    return model, hist


# -----------------------------
# Streamlit UI
# -----------------------------

st.set_page_config(page_title="Federated Fraud Detection (Flower)", layout="wide")
st.title("Federated Credit Card Fraud Detection")
st.caption("Privacy-preserving training demo using Flower (FedAvg) and PyTorch")

with st.sidebar:
    st.header("Simulation Settings")
    num_clients = st.slider("Number of clients", 2, 10, 5, 1)
    rounds = st.slider("Federated rounds", 1, 20, 5, 1)
    local_epochs = st.slider("Local epochs per client", 1, 5, 1, 1)
    train_bs = st.selectbox("Train batch size", [256, 512, 1024, 2048], index=2)
    val_bs = 2048
    do_baseline = st.checkbox("Run centralized baseline after FL", value=True)
    base_epochs = st.slider("Baseline epochs", 1, 10, 5, 1)
    base_bs = st.selectbox("Baseline batch size", [512, 1024, 2048, 4096], index=1)

with st.expander("About the data and imbalance", expanded=False):
    st.write("Dataset: Credit card transactions (highly imbalanced). Class 1 = fraud.")

raw = load_dataset()
artifact = preprocess_and_partition(raw, num_clients=num_clients)

cols = st.columns(3)
with cols[0]:
    st.metric("Samples", f"{len(raw):,}")
with cols[1]:
    ratio = artifact["class_ratio"]
    fraud_ratio = ratio.get(1, 0.0)
    st.metric("Fraud ratio", f"{fraud_ratio:.4%}")
with cols[2]:
    st.metric("Features", str(artifact["input_dim"]))

st.divider()

run_clicked = st.button("Run Federated Simulation", type="primary")

if run_clicked:
    with st.status("Running federated simulation...", expanded=True) as status:
        st.write("Starting Flower in-process simulation")
        metrics_rounds = run_fl_simulation(
            artifact=artifact,
            rounds=rounds,
            local_epochs=local_epochs,
            train_bs=train_bs,
            val_bs=val_bs,
        )
        st.write("Completed federated training.")
        status.update(label="Federated simulation completed", state="complete")

    # Plot ROC-AUC over rounds
    if metrics_rounds:
        r_list = [r for r, _ in metrics_rounds]
        roc_list = [m.get("roc_auc") for _, m in metrics_rounds]
        ap_list = [m.get("avg_precision") for _, m in metrics_rounds]

        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        ax[0].plot(r_list, roc_list, marker='o')
        ax[0].set_xlabel("Round"); ax[0].set_ylabel("ROC-AUC"); ax[0].set_title("Global ROC-AUC across rounds")
        ax[0].grid(True)
        ax[1].plot(r_list, ap_list, marker='o', color='orange')
        ax[1].set_xlabel("Round"); ax[1].set_ylabel("Average Precision"); ax[1].set_title("Global AP across rounds")
        ax[1].grid(True)
        st.pyplot(fig)

        # Plain-language explanation below the plots
        last_r = r_list[-1]
        last_roc = roc_list[-1]
        last_ap = ap_list[-1]
        st.markdown(
            f"**What this means (Round {last_r}):**\n\n"
            f"- **ROC-AUC ≈ {last_roc:.3f}** — probability the model ranks a random fraud higher than a random legitimate transaction. Closer to 1.0 is better.\n"
            f"- **Average Precision ≈ {last_ap:.3f}** — precision-recall quality on imbalanced data; higher means fewer false alarms for a given recall.\n"
            "- The upward trend across rounds indicates federated averaging is helping the global model learn from all clients without sharing raw data."
        )

    st.success("Federated training done.")

    if do_baseline:
        with st.status("Training centralized baseline...", expanded=True) as status2:
            model_c, hist = train_central_baseline(artifact, epochs=base_epochs, batch_size=base_bs)
            st.write("Baseline training completed.")
            status2.update(label="Baseline completed", state="complete")

        # Evaluate curves
        test_loader = make_loader(artifact["X_test_global"], artifact["y_test_global"], batch_size=2048, shuffle=False)
        model_c.eval()
        probs, truth = [], []
        with torch.no_grad():
            for xb, yb in test_loader:
                xb = xb.to(DEVICE)
                logits = model_c(xb)
                p = 1.0 / (1.0 + torch.exp(-logits))
                probs.append(p.cpu().numpy())
                truth.append(yb.numpy())
        probs = np.concatenate(probs)
        truth = np.concatenate(truth)
        fpr, tpr, _ = roc_curve(truth, probs)
        prec, rec, _ = precision_recall_curve(truth, probs)
        cm = confusion_matrix(truth, (probs >= 0.5).astype(int))

        final_metrics = evaluate(model_c, test_loader)
        st.subheader("Centralized baseline results")
        cols2 = st.columns(3)
        cols2[0].metric("ROC-AUC", f"{final_metrics['roc_auc']:.4f}")
        cols2[1].metric("Avg Precision", f"{final_metrics['avg_precision']:.4f}")
        cols2[2].metric("Accuracy", f"{final_metrics['accuracy']:.4f}")

        fig2, ax2 = plt.subplots(1, 2, figsize=(12, 4))
        ax2[0].plot(fpr, tpr, label=f"ROC-AUC={final_metrics['roc_auc']:.4f}")
        ax2[0].plot([0,1],[0,1], 'k--')
        ax2[0].set_xlabel("FPR"); ax2[0].set_ylabel("TPR"); ax2[0].set_title("Centralized ROC")
        ax2[0].legend(); ax2[0].grid(True)
        ax2[1].plot(rec, prec, label=f"AP={final_metrics['avg_precision']:.4f}", color='orange')
        ax2[1].set_xlabel("Recall"); ax2[1].set_ylabel("Precision"); ax2[1].set_title("Centralized PR")
        ax2[1].legend(); ax2[1].grid(True)
        st.pyplot(fig2)

        st.write("Confusion Matrix (thr=0.5)")
        fig3, ax3 = plt.subplots(figsize=(4, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax3)
        st.pyplot(fig3)

        # Plain-language explanation for baseline
        st.markdown(
            "**How to read this baseline:**\n\n"
            "- Centralized training represents an upper bound in this demo (all data pooled).\n"
            "- Compare its ROC-AUC/AP to the federated curves; if close, your FL approach is competitive while preserving privacy.\n"
            "- The confusion matrix at threshold 0.5 shows trade-offs: lowering the threshold will catch more fraud (higher recall) but may increase false positives (lower precision)."
        )

with st.expander("How to explain this demo", expanded=False):
    st.markdown(
        "- Clients keep data local. Only model weights are aggregated (FedAvg).\n"
        "- Highly imbalanced fraud data: we use class-weighted loss.\n"
        "- Track global ROC-AUC/AP across rounds to show improvements.\n"
        "- Compare to centralized baseline as an upper bound in this synthetic setting."
    )



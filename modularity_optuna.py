import optuna
from optuna.exceptions import TrialPruned
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import os
from sklearn import metrics
from models_pytorch import TGCN
from utils import load_corpus_torch, get_edge_tensor
from train_modularity import evaluate

dataset = "mr"  # default dataset
layers = 2
epochs = 1000

# -------------------
# Global log file
# -------------------
LOG_FILE = "optuna_training.log"

def log(msg: str):
    with open(LOG_FILE, "a") as f:
        f.write(msg + "\n")
    print(msg)

# -------------------
# Optuna objective
# -------------------
def objective(trial):

    # --- Hyperparameters to tune ---
    hidden_dim   = 200
    lr           = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    dropout      = trial.suggest_float("dropout", 0.2, 0.8)
    lam          = trial.suggest_float("lambda", 0.3, 0.9)  # CE vs modularity weight
    gamma        = trial.suggest_float("gamma", 0.5, 2.0)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # --- Load corpus & modularity ---
    adj, adj1, adj2, y_train, y_val, y_test, train_mask, val_mask, test_mask, \
        train_size, val_size, test_size, num_labels, doc_indices = load_corpus_torch(dataset, device)

    modularity_path = os.path.join(".", "data", f"{dataset}.full.full.modularity_adj")
    with open(modularity_path, "rb") as f:
        modularity_matrix = np.load(f)

    # Build adjacency list
    support_mix = [adj.tocoo(), adj1.tocoo(), adj2.tocoo()]
    indice_list, weight_list = [], []
    for adjacency in support_mix:
        ind, dat = get_edge_tensor(adjacency)
        indice_list.append(ind.to(device))
        weight_list.append(dat.to(device))

    in_dim = adj.shape[0]
    features = torch.tensor(list(range(in_dim)), dtype=torch.long).to(device)

    # --- Model ---
    model = TGCN(
        in_dim=in_dim,
        hidden_dim=hidden_dim,
        out_dim=num_labels,
        num_graphs=3,
        dropout=dropout,
        n_layers=layers,
        bias=False,
    ).to(device)

    # --- Optimizer ---
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fct = nn.CrossEntropyLoss(reduction="none")

    best_val_acc = 0.0
    best_epoch = 0
    best_test_acc = 0.0

        # -------------------
    # Training Loop
    # -------------------
    try:
        for epoch in range(epochs):
            model.train()
            outs = model(features, indice_list, weight_list, 1 - dropout)

            # Mask docs
            train_label_docs = y_train[doc_indices]
            train_mask_docs  = train_mask[doc_indices].bool()

            output_prob = F.softmax(outs[doc_indices], dim=1).clone()
            output_prob[train_mask_docs] = F.one_hot(
                train_label_docs[train_mask_docs], num_classes=output_prob.size(1)
            ).float()

            # Modularity loss
            modularity_matrix_t = torch.tensor(modularity_matrix, dtype=torch.float, device=device)
            D_W = torch.sum(modularity_matrix_t, dim=0)
            e_W = torch.sum(modularity_matrix_t) / 2.0
            B_W = modularity_matrix_t - gamma * torch.outer(D_W, D_W) / (2.0 * e_W)
            B_W = B_W / (torch.max(torch.abs(B_W)) + 1e-8)
            loss_mod = torch.trace(output_prob.T @ B_W @ output_prob) / (2.0 * e_W)

            # CE loss
            pre_loss = loss_fct(outs, y_train)
            ce_loss = (pre_loss * train_mask/train_mask.mean()).mean()

            loss = lam * ce_loss + (1 - lam) * loss_mod

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # --- Validation ---
            val_loss, val_acc, _, _, _ = evaluate(features, y_val, val_mask, model, indice_list, weight_list)
            trial.report(val_acc, epoch)

            if trial.should_prune():
                # log once for pruning, then raise
                log(f"=== Trial {trial.number} PRUNED at epoch {epoch} | Best Val Acc: {best_val_acc:.4f} "
                    f"| Test Acc: {best_test_acc:.4f} | Params: {trial.params} ===")
                raise TrialPruned()

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch
                _, test_acc, _, _, _ = evaluate(features, y_test, test_mask, model, indice_list, weight_list)
                best_test_acc = test_acc

            # --- Epoch logging ---
            if epoch % 10 == 0 or epoch == epochs - 1:
                print(f"Trial {trial.number} | Epoch {epoch} | CE: {ce_loss.item():.4f} "
                    f"| MOD: {loss_mod.item():.4f} | Val Acc: {val_acc:.4f}")

        # --- Trial finished normally ---
        log(f"=== Trial {trial.number} FINISHED | Best Val Acc: {best_val_acc:.4f} "
            f"at epoch {best_epoch} | Test Acc: {best_test_acc:.4f} | Params: {trial.params} ===")

        return best_val_acc

    except TrialPruned:
        log(f"=== Trial {trial.number} PRUNED | Best Val Acc: {best_val_acc:.4f} "
            f"at epoch {best_epoch} | Test Acc: {best_test_acc:.4f} "
            f"| Params: {trial.params} ===")
        raise



# -------------------
# Run study
# -------------------
if __name__ == "__main__":

    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=5,
        n_warmup_steps=20,
        interval_steps=5
    )

    study = optuna.create_study(
        direction="maximize",
        study_name=f"tgcn_{dataset}_optuna",
        pruner=pruner
    )

    # Clear old log
    open(LOG_FILE, "w").close()
    log("==== Starting Optuna Study ====")

    study.optimize(lambda trial: objective(trial), n_trials=100, timeout=None)

    log("Best trial:")
    trial = study.best_trial
    log(f"  Value: {trial.value:.4f}")
    log("  Params: ")
    for k, v in trial.params.items():
        log(f"    {k}: {v}")



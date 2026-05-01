"""
GNN-SDM model definition and training function.

Shared across notebooks 20 (architecture search), 21 (selected species),
and 22 (production training).
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from sklearn.svm import OneClassSVM
from sklearn.metrics import roc_auc_score


class GNNSDM(torch.nn.Module):
    """
    GraphSAGE-based Species Distribution Model.

    Configurable number of layers and hidden dimensions.
    Uses mean aggregation, LeakyReLU activation, dropout,
    and sigmoid output for habitat suitability [0, 1].

    Parameters
    ----------
    in_channels : int
        Number of input features per node.
    hidden_dims : list[int]
        Hidden layer dimensions. Length determines number of GraphSAGE layers.
        Default [24, 18, 8] matches Wu et al. (2025).
    dropout : float
        Dropout probability during training.
    """

    def __init__(self, in_channels, hidden_dims=[24, 18, 8], dropout=0.2):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.dropout = dropout

        prev = in_channels
        for dim in hidden_dims:
            self.convs.append(SAGEConv(prev, dim, aggr='mean'))
            prev = dim

        self.out = torch.nn.Linear(prev, 1)

    def forward(self, x, edge_index):
        for conv in self.convs:
            x = F.leaky_relu(conv(x, edge_index))
            x = F.dropout(x, p=self.dropout, training=self.training)
        return torch.sigmoid(self.out(x)).squeeze(-1)


def train_gnn_species(
    species_name,
    presence_patches,
    X,
    edge_index,
    pr_values,
    n_patches,
    device,
    hidden_dims=[24, 18, 8],
    epochs=500,
    lr=0.001,
    patience=50,
    return_history=False,
):
    """
    Train GNN-SDM for one species with early stopping.

    Pipeline:
    1. Select background patches via One-Class SVM (most dissimilar to presence)
    2. Weight presence patches by PageRank (structural importance)
    3. Train with weighted MSE loss, 80/20 train/val split
    4. Early stopping if val AUC doesn't improve for *patience* epochs
    5. Restore best model and predict suitability for all patches

    Parameters
    ----------
    species_name : str
        Species name (for logging only).
    presence_patches : set[int]
        Set of patch IDs where the species is present.
    X : torch.Tensor (n_patches, n_features)
        Node feature matrix, already on *device*.
    edge_index : torch.Tensor (2, n_edges)
        Edge index, already on *device*.
    pr_values : torch.Tensor (n_patches,)
        PageRank values for all patches (CPU).
    n_patches : int
        Total number of patches.
    device : torch.device
        GPU or CPU device.
    hidden_dims : list[int]
        Architecture hidden layer sizes.
    epochs : int
        Maximum training epochs.
    lr : float
        Learning rate for Adam optimizer.
    patience : int
        Early stopping patience (in epochs, checked every 10).
    return_history : bool
        If True, also return training history list.

    Returns
    -------
    suitability : np.ndarray (n_patches,)
        Predicted habitat suitability [0, 1] for all patches.
    best_val_auc : float
        Best validation AUC achieved during training.
    history : list[dict] (only if return_history=True)
        List of {'epoch', 'train_loss', 'val_loss', 'val_auc'} dicts.
    """
    presence = np.array(list(presence_patches))
    n_pres = len(presence)

    # Background selection via One-Class SVM
    X_np = X.cpu().numpy()
    oc_svm = OneClassSVM(kernel='rbf', gamma='scale', nu=0.1)
    oc_svm.fit(X_np[presence])

    non_presence = np.setdiff1d(np.arange(n_patches), presence)
    scores = oc_svm.decision_function(X_np[non_presence])
    n_bg = min(len(non_presence), n_pres * 2)
    bg_idx = non_presence[np.argsort(scores)[:n_bg]]

    # Labels and PageRank weights
    labels = torch.zeros(n_patches, dtype=torch.float32)
    labels[presence] = 1.0
    weights = torch.zeros(n_patches, dtype=torch.float32)
    weights[presence] = pr_values[presence]
    weights[presence] = weights[presence] / weights[presence].sum() * n_pres
    weights[bg_idx] = 1.0

    # Train/val split
    labelled = np.concatenate([presence, bg_idx])
    rng = np.random.default_rng(42)
    rng.shuffle(labelled)
    split = int(0.8 * len(labelled))
    train_mask = torch.zeros(n_patches, dtype=torch.bool, device=device)
    val_mask = torch.zeros(n_patches, dtype=torch.bool, device=device)
    train_mask[labelled[:split]] = True
    val_mask[labelled[split:]] = True

    labels = labels.to(device)
    weights = weights.to(device)

    # Model and optimizer
    model = GNNSDM(X.shape[1], hidden_dims).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_auc = 0.0
    best_state = None
    epochs_no_improve = 0
    history = [] if return_history else None

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        pred = model(X, edge_index)
        loss = (weights[train_mask] * (pred[train_mask] - labels[train_mask]) ** 2).mean()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                pred_all = model(X, edge_index)
                val_pred = pred_all[val_mask].cpu().numpy()
                val_labels = labels[val_mask].cpu().numpy()
                train_loss = loss.item()

            val_auc = 0.0
            if len(np.unique(val_labels)) == 2:
                val_auc = roc_auc_score(val_labels, val_pred)
                if val_auc > best_val_auc:
                    best_val_auc = val_auc
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 10

            if return_history:
                val_loss = (weights[val_mask] * (pred_all[val_mask] - labels[val_mask]) ** 2).mean().item()
                history.append({
                    'epoch': epoch + 1,
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'val_auc': val_auc,
                })

            # Early stopping
            if epochs_no_improve >= patience:
                break

    # Restore best model
    if best_state:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    # Predict suitability for all patches
    model.eval()
    with torch.no_grad():
        suitability = model(X, edge_index).cpu().numpy()

    if return_history:
        return suitability, best_val_auc, history
    return suitability, best_val_auc

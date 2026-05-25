"""
SageMaker Training Job: GNN-SDM for 12 selected species.

Reads input data from /opt/ml/input/data/training/
Writes outputs to /opt/ml/model/

Usage (local test):
    python train_selected_species.py --local

Usage (SageMaker):
    Submitted via Estimator in notebook 21b_submit_training_job.ipynb
"""

import argparse
import json
import os
import pickle
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
from sklearn.metrics import roc_auc_score


# ---- Early GPU diagnostics ----

def log_device_info():
    """Log GPU/CPU info early so we know what hardware we got."""
    print("=" * 60)
    print("DEVICE DIAGNOSTICS")
    print("=" * 60)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available:  {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version:    {torch.version.cuda}")
        print(f"GPU count:       {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            mem = getattr(props, 'total_memory', None) or getattr(props, 'total_mem', 0)
            print(f"  GPU {i}: {props.name} ({mem / 1024**3:.1f} GB)")
        device = torch.device('cuda')
    else:
        print("WARNING: No GPU detected — training will run on CPU.")
        print("  This may indicate GPUs are disabled or the instance has none.")
        device = torch.device('cpu')
    print(f"Selected device: {device}")
    print("=" * 60)
    sys.stdout.flush()
    return device

# ---- Model definition (same as gnn_model.py) ----

class GNNSDM(torch.nn.Module):
    def __init__(self, in_channels, hidden_dims=[64, 48, 32], dropout=0.2):
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


def train_one_species(presence_patches, X, edge_index, pr_values,
                      n_patches, device, hidden_dims, epochs=500, patience=80):
    """Train GNN-SDM for one species with random background and early stopping."""
    presence = np.array(list(presence_patches))
    n_pres = len(presence)
    non_presence = np.setdiff1d(np.arange(n_patches), presence)

    # Random background (3:1 ratio)
    rng = np.random.default_rng(42)
    n_bg = min(len(non_presence), n_pres * 3)
    bg_idx = rng.choice(non_presence, n_bg, replace=False)

    # Labels and weights
    labels = torch.zeros(n_patches, dtype=torch.float32)
    labels[presence] = 1.0
    weights = torch.zeros(n_patches, dtype=torch.float32)
    weights[presence] = pr_values[presence]
    weights[presence] = weights[presence] / weights[presence].sum() * n_pres
    weights[bg_idx] = 1.0

    # Train/val split
    labelled = np.concatenate([presence, bg_idx])
    rng.shuffle(labelled)
    split = int(0.8 * len(labelled))
    train_mask = torch.zeros(n_patches, dtype=torch.bool, device=device)
    val_mask = torch.zeros(n_patches, dtype=torch.bool, device=device)
    train_mask[labelled[:split]] = True
    val_mask[labelled[split:]] = True

    labels = labels.to(device)
    weights = weights.to(device)

    model = GNNSDM(X.shape[1], hidden_dims).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    best_val_auc = 0.0
    best_state = None
    epochs_no_improve = 0
    history = []

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

            val_auc = 0.0
            if len(np.unique(val_labels)) == 2:
                val_auc = roc_auc_score(val_labels, val_pred)
                if val_auc > best_val_auc:
                    best_val_auc = val_auc
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 10

            train_loss = loss.item()
            val_loss = (weights[val_mask] * (pred_all[val_mask] - labels[val_mask]) ** 2).mean().item()
            history.append({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_auc': val_auc,
            })

            if epochs_no_improve >= patience:
                break

    # Restore best model
    if best_state:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    # Predict suitability
    model.eval()
    with torch.no_grad():
        suitability = model(X, edge_index).cpu().numpy()

    return model.state_dict(), suitability, best_val_auc, history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--local', action='store_true', help='Run locally (not in SageMaker)')
    args = parser.parse_args()

    # Paths
    if args.local:
        input_dir = '.'
        output_dir = './training_output'
    else:
        input_dir = '/opt/ml/input/data/training'
        output_dir = '/opt/ml/model'

    os.makedirs(output_dir, exist_ok=True)

    # Device (with full diagnostics)
    device = log_device_info()

    # Load data
    print('Loading data...')
    with open(os.path.join(input_dir, 'landscape_graph.pkl'), 'rb') as f:
        G = pickle.load(f)

    patch_features = np.load(os.path.join(input_dir, 'patch_features.npy'))
    with open(os.path.join(input_dir, 'species_patches.pkl'), 'rb') as f:
        all_species_patches = pickle.load(f)

    n_patches = patch_features.shape[0]
    n_features = patch_features.shape[1]
    print(f'Graph: {n_patches:,} nodes, {G.number_of_edges():,} edges, {n_features} features')

    # Convert to PyG
    X = torch.tensor(patch_features, dtype=torch.float32)
    X = (X - X.mean(dim=0)) / X.std(dim=0).clamp(min=1e-6)

    edges = list(G.edges())
    src = [e[0] for e in edges] + [e[1] for e in edges]
    dst = [e[1] for e in edges] + [e[0] for e in edges]
    edge_index = torch.tensor([src, dst], dtype=torch.long)

    # PageRank
    print('Computing PageRank...')
    pagerank = nx.pagerank(G)
    pr_values = torch.tensor([pagerank[i] for i in range(n_patches)], dtype=torch.float32)

    # Move to device
    X = X.to(device)
    edge_index = edge_index.to(device)

    # Species to train
    SPECIES = [
        'Picea abies', 'Fagus sylvatica', 'Trifolium pratense',
        'Potentilla erecta', 'Phragmites australis', 'Senecio inaequidens',
        'Cypripedium calceolus', 'Drosera rotundifolia', 'Pulsatilla vulgaris',
        'Gladiolus palustris', 'Aquilegia alpina', 'Stipa pennata',
    ]

    hidden_dims = [64, 48, 32]
    results = []
    all_suitability = {}

    print(f'\nTraining {len(SPECIES)} species with architecture {hidden_dims}...\n')

    for i, sp in enumerate(SPECIES):
        presence = all_species_patches.get(sp, set())
        if len(presence) < 5:
            print(f'  {sp}: too few patches ({len(presence)}), skipping')
            continue

        t0 = time.time()
        state_dict, suitability, val_auc, history = train_one_species(
            presence, X, edge_index, pr_values,
            n_patches, device, hidden_dims,
            epochs=500, patience=80,
        )
        dt = time.time() - t0

        # Save model weights
        model_path = os.path.join(output_dir, f'model_{sp.replace(" ", "_")}.pt')
        torch.save(state_dict, model_path)

        # Save suitability
        all_suitability[sp] = suitability

        results.append({
            'species': sp,
            'n_presence_patches': len(presence),
            'val_auc': val_auc,
            'epochs_trained': history[-1]['epoch'] if history else 0,
            'train_time_s': dt,
        })

        print(f'  [{i+1}/{len(SPECIES)}] {sp:30s}  AUC={val_auc:.3f}  '
              f'epochs={history[-1]["epoch"] if history else 0}  ({dt:.1f}s)')

    # Save results
    import pandas as pd
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(output_dir, 'training_results.csv'), index=False)

    # Save all suitability scores
    np.savez_compressed(
        os.path.join(output_dir, 'suitability_scores.npz'),
        **{sp.replace(' ', '_'): scores for sp, scores in all_suitability.items()}
    )

    print(f'\nDone. Results saved to {output_dir}')
    print(results_df.to_string(index=False))


if __name__ == '__main__':
    main()

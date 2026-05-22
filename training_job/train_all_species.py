"""
SageMaker Training Job: GNN-SDM for ALL qualifying species.

Reads input data from /opt/ml/input/data/training/
Writes outputs to /opt/ml/model/
Supports checkpointing for spot instance interruptions via /opt/ml/checkpoints/

Usage (local test):
    python train_all_species.py --local

Usage (SageMaker):
    Submitted via Estimator in notebook 21b_submit_training_job.ipynb
"""

import argparse
import os
import pickle
import signal
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import networkx as nx
from torch_geometric.nn import SAGEConv
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score,
    recall_score, f1_score, matthews_corrcoef, cohen_kappa_score,
)


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
        device = torch.device('cpu')
    print(f"Selected device: {device}")
    print("=" * 60)
    sys.stdout.flush()
    return device


# ---- Model definition ----

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
                      n_patches, device, hidden_dims, epochs=500, patience=50):
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

            if len(np.unique(val_labels)) == 2:
                val_auc = roc_auc_score(val_labels, val_pred)
                if val_auc > best_val_auc:
                    best_val_auc = val_auc
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 10

            if epochs_no_improve >= patience:
                break

    # Restore best model
    if best_state:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    # Predict suitability
    model.eval()
    with torch.no_grad():
        suitability = model(X, edge_index).cpu().numpy()

    final_epoch = epoch + 1
    return suitability, best_val_auc, final_epoch


def evaluate_species(suitability, presence_patches, n_patches):
    """Compute evaluation metrics for one species."""
    pres_arr = np.array(list(presence_patches))
    absence = np.setdiff1d(np.arange(n_patches), pres_arr)
    rng = np.random.default_rng(42)
    abs_sample = rng.choice(absence, min(len(absence), len(pres_arr) * 3), replace=False)

    eval_idx = np.concatenate([pres_arr, abs_sample])
    y_true = np.concatenate([np.ones(len(pres_arr)), np.zeros(len(abs_sample))])
    y_score = suitability[eval_idx]
    y_pred = (y_score >= 0.5).astype(int)

    return {
        'auc': roc_auc_score(y_true, y_score),
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred),
        'mcc': matthews_corrcoef(y_true, y_pred),
        'kappa': cohen_kappa_score(y_true, y_pred),
        'tss': recall_score(y_true, y_pred) + recall_score(y_true, y_pred, pos_label=0) - 1,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--local', action='store_true', help='Run locally (not in SageMaker)')
    parser.add_argument('--min-patches', type=int, default=5,
                        help='Minimum presence patches to train a species')
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--patience', type=int, default=50)
    parser.add_argument('--hidden-dims', type=str, default='64,48,32')
    parser.add_argument('--checkpoint-every', type=int, default=50)
    args = parser.parse_args()

    hidden_dims = [int(x) for x in args.hidden_dims.split(',')]

    # Paths
    if args.local:
        input_dir = '.'
        output_dir = './training_output'
        checkpoint_dir = './checkpoints'
    else:
        input_dir = '/opt/ml/input/data/training'
        output_dir = '/opt/ml/model'
        checkpoint_dir = '/opt/ml/checkpoints'

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Device
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

    # Convert to PyG format
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

    # Filter species
    species_list = sorted([
        sp for sp, patches in all_species_patches.items()
        if len(patches) >= args.min_patches
    ])
    print(f'\nQualifying species: {len(species_list)} (min {args.min_patches} patches)')
    print(f'Architecture: {hidden_dims}, epochs: {args.epochs}, patience: {args.patience}')

    # Resume from checkpoint if available (for spot instance restarts)
    checkpoint_file = os.path.join(checkpoint_dir, 'progress.csv')
    suitability_checkpoint = os.path.join(checkpoint_dir, 'suitability_partial.npz')

    if os.path.exists(checkpoint_file):
        done_df = pd.read_csv(checkpoint_file)
        results = done_df.to_dict('records')
        done_species = set(done_df['species'])
        print(f'Resuming from checkpoint: {len(done_species)} species already done')
        # Load partial suitability
        if os.path.exists(suitability_checkpoint):
            loaded = np.load(suitability_checkpoint)
            all_suitability = {k: loaded[k] for k in loaded.files}
        else:
            all_suitability = {}
    else:
        results = []
        done_species = set()
        all_suitability = {}

    remaining = [sp for sp in species_list if sp not in done_species]
    print(f'Remaining to train: {len(remaining)}')
    sys.stdout.flush()

    # Handle spot instance termination (SIGTERM sent 2 min before shutdown)
    interrupted = False

    def handle_sigterm(signum, frame):
        nonlocal interrupted
        interrupted = True
        print('\n⚠️  SIGTERM received — checkpointing and exiting gracefully...')
        sys.stdout.flush()

    signal.signal(signal.SIGTERM, handle_sigterm)

    t0 = time.time()

    for i, sp in enumerate(remaining):
        if interrupted:
            print('Stopping due to spot interruption.')
            break

        presence = all_species_patches[sp]

        sp_t0 = time.time()
        suitability, val_auc, final_epoch = train_one_species(
            presence, X, edge_index, pr_values,
            n_patches, device, hidden_dims,
            epochs=args.epochs, patience=args.patience,
        )
        sp_dt = time.time() - sp_t0

        # Evaluate
        metrics = evaluate_species(suitability, presence, n_patches)

        all_suitability[sp.replace(' ', '_')] = suitability

        results.append({
            'species': sp,
            'n_presence_patches': len(presence),
            'val_auc': val_auc,
            'auc_mean': metrics['auc'],
            'accuracy_mean': metrics['accuracy'],
            'precision_mean': metrics['precision'],
            'recall_mean': metrics['recall'],
            'f1_mean': metrics['f1'],
            'mcc_mean': metrics['mcc'],
            'kappa_mean': metrics['kappa'],
            'tss_mean': metrics['tss'],
            'epochs_trained': final_epoch,
            'train_time_s': sp_dt,
        })

        # Progress logging
        if (i + 1) % 10 == 0 or (i + 1) == len(remaining):
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(remaining) - i - 1) / rate if rate > 0 else 0
            print(f'  [{len(done_species) + i + 1}/{len(species_list)}] '
                  f'{sp:30s}  AUC={val_auc:.3f}  '
                  f'({sp_dt:.1f}s)  ETA={eta/60:.0f}min')
            sys.stdout.flush()

        # Checkpoint periodically (critical for spot instances)
        if (i + 1) % args.checkpoint_every == 0 or interrupted:
            pd.DataFrame(results).to_csv(checkpoint_file, index=False)
            # Save suitability less often (it's large) — every 200 species or on interrupt
            if (i + 1) % 200 == 0 or interrupted:
                np.savez_compressed(suitability_checkpoint, **all_suitability)
                print(f'  [checkpoint saved: {len(results)} species, incl. suitability]')
            else:
                print(f'  [checkpoint saved: {len(results)} species]')
            sys.stdout.flush()

    # Final save to output dir
    elapsed = time.time() - t0

    # If interrupted, save checkpoint so next run can resume
    if interrupted:
        pd.DataFrame(results).to_csv(checkpoint_file, index=False)
        np.savez_compressed(suitability_checkpoint, **all_suitability)
        print(f'Checkpoint saved before exit: {len(results)} species')
        sys.exit(0)  # Exit cleanly so SageMaker saves checkpoints

    print(f'\nTraining complete: {len(results)} species in {elapsed:.0f}s ({elapsed/60:.1f} min)')

    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(output_dir, 'gnn_sdm_results.csv'), index=False)

    np.savez_compressed(
        os.path.join(output_dir, 'suitability_scores.npz'),
        **all_suitability
    )

    # Summary
    print(f'\nMean AUC: {results_df["auc_mean"].mean():.3f}')
    print(f'Mean F1:  {results_df["f1_mean"].mean():.3f}')
    print(f'Mean TSS: {results_df["tss_mean"].mean():.3f}')
    print(f'\nResults saved to {output_dir}')


if __name__ == '__main__':
    main()

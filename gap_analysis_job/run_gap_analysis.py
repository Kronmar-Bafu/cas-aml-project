"""
SageMaker Training Job: Circuit-theory connectivity for ALL species.

Computes current-flow connectivity for each species using the GNN-SDM
suitability scores. Saves the FULL per-species connectivity arrays
(both raw and normalised) so that aggregation strategy can be decided
downstream — the distributions are heavily right-skewed and the best
aggregation method (mean, median, rank-based, weighted) is TBD.

Reads input data from /opt/ml/input/data/training/
Writes outputs to /opt/ml/model/
Supports checkpointing for spot instance interruptions via /opt/ml/checkpoints/

Outputs:
    - current_flow_raw.npz    — raw current flow per species (float32)
    - current_flow_normed.npz — per-species normalised [0,1] current flow (float32)
    - gap_analysis_results.csv — per-species summary stats + distributional info

Usage (local test):
    python run_gap_analysis.py --local --max-species 10

Usage (SageMaker):
    Submitted via Estimator in notebook 32_submit_gap_analysis_job.ipynb
"""

import argparse
import os
import pickle
import signal
import sys
import time

import numpy as np
import pandas as pd
import networkx as nx
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve


def compute_current_flow(G, suitability, core_patches, n_pairs=5, min_component_size=3):
    """Circuit-theory current flow between known population clusters.

    Parameters
    ----------
    G : networkx.Graph
        Landscape graph (nodes = patch IDs).
    suitability : np.ndarray
        Per-patch suitability scores from GNN-SDM.
    core_patches : set or list
        Patch IDs where the species has been observed.
    n_pairs : int
        Number of component pairs to solve current flow between.
    min_component_size : int
        Minimum connected component size to consider as a population core.

    Returns
    -------
    np.ndarray — per-patch current flow values.
    """
    n = G.number_of_nodes()
    nodes = sorted(G.nodes())
    node_to_idx = {nd: i for i, nd in enumerate(nodes)}

    # Build conductance Laplacian
    L = lil_matrix((n, n), dtype=np.float64)
    for u, v in G.edges():
        i, j = node_to_idx[u], node_to_idx[v]
        conductance = max(1e-4, (suitability[u] + suitability[v]) / 2)
        L[i, j] -= conductance
        L[j, i] -= conductance
        L[i, i] += conductance
        L[j, j] += conductance
    L = csr_matrix(L)

    # Core components from actual observations
    cores = set(core_patches)
    core_subgraph = G.subgraph(cores)
    components = [c for c in nx.connected_components(core_subgraph)
                  if len(c) >= min_component_size]
    components.sort(key=len, reverse=True)

    if len(components) < 2:
        return np.zeros(len(suitability))

    # Solve current flow between top component pairs
    current = np.zeros(n, dtype=np.float64)
    n_solve = min(len(components), n_pairs)

    for pair_i in range(n_solve - 1):
        src_comp = components[pair_i]
        gnd_comp = components[pair_i + 1]

        rhs = np.zeros(n)
        src_idx = [node_to_idx[nd] for nd in src_comp]
        gnd_idx = [node_to_idx[nd] for nd in gnd_comp]
        for idx in src_idx:
            rhs[idx] = 1.0 / len(src_idx)
        for idx in gnd_idx:
            rhs[idx] = -1.0 / len(gnd_idx)

        pin = gnd_idx[0]
        L_solve = L.copy().tolil()
        L_solve[pin, :] = 0
        L_solve[pin, pin] = 1
        rhs[pin] = 0

        voltages = spsolve(csr_matrix(L_solve), rhs)

        node_current = np.zeros(n)
        for u, v in G.edges():
            i, j = node_to_idx[u], node_to_idx[v]
            cond = max(1e-4, (suitability[u] + suitability[v]) / 2)
            ec = abs(cond * (voltages[i] - voltages[j]))
            node_current[i] += ec
            node_current[j] += ec
        current += node_current

    # Map back to patch IDs
    result = np.zeros(len(suitability))
    for nd, idx in node_to_idx.items():
        result[nd] = current[idx]
    return result


def main():
    parser = argparse.ArgumentParser(description='Gap analysis batch job')
    parser.add_argument('--local', action='store_true',
                        help='Run locally (not in SageMaker)')
    parser.add_argument('--max-species', type=int, default=0,
                        help='Limit number of species (0 = all)')
    parser.add_argument('--min-patches', type=int, default=10,
                        help='Minimum presence patches to include a species')
    parser.add_argument('--n-pairs', type=int, default=6,
                        help='Number of component pairs for current flow')
    parser.add_argument('--checkpoint-every', type=int, default=25,
                        help='Checkpoint every N species')
    parser.add_argument('--species-list', type=str, default='',
                        help='Path to CSV with species to process (column: scientific_name). '
                             'If empty, process all species meeting min-patches threshold.')
    args = parser.parse_args()

    # Paths
    if args.local:
        input_dir = '.'
        output_dir = './gap_analysis_output'
        checkpoint_dir = './gap_checkpoints'
    else:
        input_dir = '/opt/ml/input/data/training'
        output_dir = '/opt/ml/model'
        checkpoint_dir = '/opt/ml/checkpoints'

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # ---- Load data ----
    print('=' * 60)
    print('GAP ANALYSIS BATCH JOB')
    print('=' * 60)
    print(f'Loading data from: {input_dir}')
    sys.stdout.flush()

    with open(os.path.join(input_dir, 'landscape_graph.pkl'), 'rb') as f:
        G = pickle.load(f)

    n_patches = G.number_of_nodes()
    print(f'Graph: {n_patches:,} nodes, {G.number_of_edges():,} edges')

    with open(os.path.join(input_dir, 'species_patches.pkl'), 'rb') as f:
        all_species_patches = pickle.load(f)

    scores_path = os.path.join(input_dir, 'suitability_scores.npz')
    scores_npz = np.load(scores_path)
    print(f'Suitability scores available for {len(scores_npz.files):,} species')

    # ---- Build species list ----
    # Only include species that have both suitability scores AND enough patches
    available_species = set(k.replace('_', ' ') for k in scores_npz.files)

    # If a species filter list is provided, restrict to those species only
    if args.species_list:
        filter_path = os.path.join(input_dir, args.species_list)
        if os.path.exists(filter_path):
            filter_df = pd.read_csv(filter_path)
            filter_species = set(filter_df['scientific_name'])
            print(f'Species filter loaded: {len(filter_species)} species from {args.species_list}')
            available_species = available_species & filter_species
        else:
            print(f'WARNING: species list file not found at {filter_path}, processing all species')

    species_list = sorted([
        sp for sp in available_species
        if sp in all_species_patches
        and len(all_species_patches[sp]) >= args.min_patches
    ])

    if args.max_species > 0:
        species_list = species_list[:args.max_species]

    print(f'Species to process: {len(species_list)} '
          f'(min {args.min_patches} patches)')
    print(f'Parameters: n_pairs={args.n_pairs}, '
          f'checkpoint_every={args.checkpoint_every}')
    print('=' * 60)
    sys.stdout.flush()

    # ---- Resume from checkpoint ----
    checkpoint_file = os.path.join(checkpoint_dir, 'progress.csv')
    raw_checkpoint = os.path.join(checkpoint_dir, 'current_flow_raw_partial.npz')
    normed_checkpoint = os.path.join(checkpoint_dir, 'current_flow_normed_partial.npz')

    if os.path.exists(checkpoint_file):
        done_df = pd.read_csv(checkpoint_file)
        results = done_df.to_dict('records')
        done_species = set(done_df['species'])
        print(f'Resuming from checkpoint: {len(done_species)} species already done')
        if os.path.exists(raw_checkpoint):
            loaded_raw = np.load(raw_checkpoint)
            all_current_raw = {k: loaded_raw[k] for k in loaded_raw.files}
        else:
            all_current_raw = {}
        if os.path.exists(normed_checkpoint):
            loaded_norm = np.load(normed_checkpoint)
            all_current_normed = {k: loaded_norm[k] for k in loaded_norm.files}
        else:
            all_current_normed = {}
    else:
        results = []
        done_species = set()
        all_current_raw = {}
        all_current_normed = {}

    remaining = [sp for sp in species_list if sp not in done_species]
    print(f'Remaining to process: {len(remaining)}')
    sys.stdout.flush()

    # ---- Handle spot instance termination ----
    interrupted = False

    def handle_sigterm(signum, frame):
        nonlocal interrupted
        interrupted = True
        print('\n⚠️  SIGTERM received — checkpointing and exiting gracefully...')
        sys.stdout.flush()

    signal.signal(signal.SIGTERM, handle_sigterm)

    # ---- Process species ----
    t0 = time.time()

    for i, sp in enumerate(remaining):
        if interrupted:
            print('Stopping due to spot interruption.')
            break

        sp_key = sp.replace(' ', '_')
        suitability = scores_npz[sp_key]
        presence = all_species_patches[sp]

        sp_t0 = time.time()
        current = compute_current_flow(
            G, suitability, presence, n_pairs=args.n_pairs
        )
        sp_dt = time.time() - sp_t0

        # Save raw current flow
        all_current_raw[sp_key] = current.astype(np.float32)

        # Normalise to [0,1] per species
        current_max = current.max()
        if current_max > 0:
            current_norm = current / current_max
        else:
            current_norm = current
        all_current_normed[sp_key] = current_norm.astype(np.float32)

        # Per-species distributional stats (for deciding aggregation later)
        nonzero_current = current[current > 0]
        results.append({
            'species': sp,
            'n_presence_patches': len(presence),
            'current_max': float(current_max),
            'current_mean': float(current.mean()),
            'current_median': float(np.median(current)),
            'current_std': float(current.std()),
            'current_p75': float(np.percentile(current, 75)),
            'current_p90': float(np.percentile(current, 90)),
            'current_p95': float(np.percentile(current, 95)),
            'current_p99': float(np.percentile(current, 99)),
            'current_nonzero_frac': float(len(nonzero_current) / n_patches),
            'current_nonzero_mean': float(nonzero_current.mean()) if len(nonzero_current) > 0 else 0.0,
            'current_skewness': float(
                ((current - current.mean()) ** 3).mean() / max(current.std() ** 3, 1e-12)
            ),
            'time_s': sp_dt,
        })

        # Progress logging
        total_done = len(done_species) + i + 1
        if (i + 1) % 5 == 0 or (i + 1) == len(remaining):
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(remaining) - i - 1) / rate if rate > 0 else 0
            print(f'  [{total_done}/{len(species_list)}] '
                  f'{sp:35s} ({len(presence):>5} patches)  '
                  f'{sp_dt:.1f}s  ETA={eta/60:.0f}min')
            sys.stdout.flush()

        # Checkpoint periodically
        if (i + 1) % args.checkpoint_every == 0 or interrupted:
            pd.DataFrame(results).to_csv(checkpoint_file, index=False)
            # Save arrays less often (they're large) — every 100 or on interrupt
            if (i + 1) % 100 == 0 or interrupted:
                np.savez_compressed(raw_checkpoint, **all_current_raw)
                np.savez_compressed(normed_checkpoint, **all_current_normed)
                print(f'  [checkpoint saved: {len(results)} species, incl. arrays]')
            else:
                print(f'  [checkpoint saved: {len(results)} species]')
            sys.stdout.flush()

    # ---- Handle interruption ----
    if interrupted:
        pd.DataFrame(results).to_csv(checkpoint_file, index=False)
        np.savez_compressed(raw_checkpoint, **all_current_raw)
        np.savez_compressed(normed_checkpoint, **all_current_normed)
        print(f'Checkpoint saved before exit: {len(results)} species')
        sys.exit(0)

    # ---- Final output ----
    elapsed = time.time() - t0
    print(f'\n{"=" * 60}')
    print(f'COMPLETE: {len(results)} species in {elapsed:.0f}s ({elapsed/3600:.1f}h)')
    print(f'{"=" * 60}')

    # Save results CSV (with distributional stats for aggregation decisions)
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(output_dir, 'gap_analysis_results.csv'), index=False)

    # Save full per-species current flow — raw values
    print(f'\nSaving raw current flow ({len(all_current_raw)} species)...')
    np.savez_compressed(
        os.path.join(output_dir, 'current_flow_raw.npz'),
        **all_current_raw
    )

    # Save normalised [0,1] per-species current flow
    print(f'Saving normalised current flow ({len(all_current_normed)} species)...')
    np.savez_compressed(
        os.path.join(output_dir, 'current_flow_normed.npz'),
        **all_current_normed
    )

    # Summary stats
    print(f'\nProcessed {len(results)} species')
    print(f'  Mean skewness: {results_df["current_skewness"].mean():.2f}')
    print(f'  Median skewness: {results_df["current_skewness"].median():.2f}')
    print(f'  Mean nonzero fraction: {results_df["current_nonzero_frac"].mean():.3f}')
    print(f'\nNote: aggregation into priority surface deferred to analysis notebook.')
    print(f'      Use distributional stats in gap_analysis_results.csv to decide method.')
    print(f'\nResults saved to {output_dir}/')
    sys.stdout.flush()


if __name__ == '__main__':
    main()

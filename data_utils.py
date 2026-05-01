"""
Data loading and preprocessing utilities for the GNN-SDM pipeline.

Shared functions that are used across multiple notebooks to avoid
code duplication.
"""

import numpy as np
import pandas as pd


def load_species_patches(config, min_records=100, s3=None):
    """
    Load GBIF occurrences, map them to landscape patches, and return
    a dict of {species_name: set(patch_ids)} for all species with
    at least *min_records* GBIF records.

    Parameters
    ----------
    config : module
        Project config with GBIF_PARQUET and S3_PROCESSED paths.
    min_records : int
        Minimum GBIF records for a species to be included.
    s3 : s3fs.S3FileSystem | None
        Optional pre-configured filesystem.

    Returns
    -------
    species_patches : dict[str, set[int]]
        Mapping from species name to set of patch IDs where it occurs.
    species_counts : pd.Series
        Record counts per species (for all species, not just qualifying).
    """
    import rioxarray
    from s3_utils import load_zarr
    from rasterio.transform import rowcol

    # Load patch labels raster
    patch_labels_da = load_zarr(
        config.S3_PROCESSED + '/patches/patch_labels_30m.zarr',
        name='patch_label',
    )
    patch_labels = patch_labels_da.values
    transform = patch_labels_da.rio.transform()
    h, w = patch_labels.shape

    # Load GBIF
    gbif = pd.read_parquet(config.GBIF_PARQUET, storage_options={'anon': False})

    # Vectorized mapping: all records → patch IDs
    rows, cols = rowcol(transform, gbif['decimallongitude'].values,
                        gbif['decimallatitude'].values)
    rows, cols = np.array(rows), np.array(cols)
    valid = (rows >= 0) & (rows < h) & (cols >= 0) & (cols < w)
    gbif_patch_ids = np.full(len(gbif), -1, dtype=np.int32)
    gbif_patch_ids[valid] = patch_labels[rows[valid], cols[valid]]
    gbif['patch_id'] = gbif_patch_ids

    # Group by species
    species_counts = gbif['species'].value_counts()
    species_patch_groups = (
        gbif[gbif['patch_id'] >= 0]
        .groupby('species')['patch_id']
        .unique()
    )
    species_patches = {
        sp: set(patches)
        for sp, patches in species_patch_groups.items()
        if species_counts.get(sp, 0) >= min_records
    }

    print(f'GBIF records: {len(gbif):,}')
    print(f'Species with >= {min_records} records: {len(species_patches):,}')

    return species_patches, species_counts


def load_patch_data():
    """
    Load patch features, counts, and feature names from local files.

    Returns
    -------
    patch_features : np.ndarray (n_patches, n_features)
    patch_counts : np.ndarray (n_patches,)
    feature_names : list[str]
    """
    import json

    patch_features = np.load('patch_features.npy')
    patch_counts = np.load('patch_counts.npy')
    with open('patch_feature_names.json') as f:
        feature_names = json.load(f)

    print(f'Patches: {patch_features.shape[0]:,}, features: {len(feature_names)}')
    return patch_features, patch_counts, feature_names


def post_scale(X, feature_names):
    """
    Domain-specific adjustments applied after RobustScaler.

    1. Clip forest_edge_dist to [-5, 5] (beyond ~500m is ecologically equivalent)
    2. Boost LC fractions ×3 (small IQR since most pixels are 0 or 1)
    3. Global clip to [-5, 5] to tame extreme outliers

    Parameters
    ----------
    X : np.ndarray (n_samples, n_features)
        Scaled feature matrix.
    feature_names : list[str]
        Feature names matching columns of X.

    Returns
    -------
    np.ndarray — post-processed copy of X.
    """
    X = X.copy()
    if 'forest_edge_dist' in feature_names:
        X[:, feature_names.index('forest_edge_dist')] = np.clip(
            X[:, feature_names.index('forest_edge_dist')], -5, 5)
    for name in feature_names:
        if name.startswith('lc_'):
            X[:, feature_names.index(name)] *= 3.0
    return np.clip(X, -5, 5)

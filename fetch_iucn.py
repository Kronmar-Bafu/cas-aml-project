"""Fetch IUCN Red List categories from GBIF Species API for all species in the dataset."""

import os
import sys
import time

import pandas as pd
import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

# Load species keys
gbif = pd.read_parquet(config.GBIF_PARQUET, storage_options={'anon': False})
species_keys = gbif.groupby('species')['specieskey'].first().reset_index()
species_keys = species_keys.dropna(subset=['specieskey'])
species_keys['specieskey'] = species_keys['specieskey'].astype(int)

GBIF_API = 'https://api.gbif.org/v1/species'
CACHE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'iucn_cache.csv')

# Resume from cache
if os.path.exists(CACHE_PATH):
    cache_df = pd.read_csv(CACHE_PATH)
    cache = dict(zip(cache_df['specieskey'], cache_df['iucn_category']))
    print(f'Loaded cache: {len(cache)} species already looked up')
else:
    cache = {}

to_lookup = species_keys[~species_keys['specieskey'].isin(cache)]
print(f'Total species: {len(species_keys)}, remaining: {len(to_lookup)}')
sys.stdout.flush()

session = requests.Session()
session.headers.update({'User-Agent': 'GNN-SDM-Switzerland/1.0 (CAS project)'})

t0 = time.time()
for i, row in enumerate(to_lookup.itertuples()):
    key = int(row.specieskey)
    try:
        resp = session.get(f'{GBIF_API}/{key}/iucnRedListCategory', timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            category = data.get('category', 'NOT_EVALUATED')
        else:
            category = 'NOT_EVALUATED'
    except Exception:
        category = 'NOT_EVALUATED'

    cache[key] = category

    if (i + 1) % 500 == 0:
        elapsed = time.time() - t0
        rate = (i + 1) / elapsed
        eta = (len(to_lookup) - i - 1) / rate
        print(f'  [{i+1}/{len(to_lookup)}] {row.species} -> {category}  '
              f'({rate:.1f} req/s, ETA {eta/60:.1f}min)')
        sys.stdout.flush()
        pd.DataFrame([
            {'specieskey': k, 'iucn_category': v} for k, v in cache.items()
        ]).to_csv(CACHE_PATH, index=False)

    time.sleep(0.05)

# Final save
cache_df = pd.DataFrame([
    {'specieskey': k, 'iucn_category': v} for k, v in cache.items()
])
cache_df.to_csv(CACHE_PATH, index=False)
print(f'\nDone. {len(cache_df)} species saved to {CACHE_PATH}')
print(f'Total time: {(time.time()-t0)/60:.1f} min')
print()
print('Distribution:')
print(cache_df['iucn_category'].value_counts().to_string())

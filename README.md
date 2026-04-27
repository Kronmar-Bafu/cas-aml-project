# GNN-SDM Switzerland

Species Distribution Modelling using Graph Neural Networks, applied to Swiss plant species.

Based on the GNN-SDM framework by [Wu et al. (2025)](https://doi.org/10.1111/geb.70162), adapted for a regional study at 30 m resolution.

## Pipeline

### Data Acquisition & Feature Engineering

| Notebook | Description |
|---|---|
| `00_gbif.ipynb` | Download and filter plant occurrences from GBIF |
| `01_geographic_features.ipynb` | DEM, terrain derivatives (slope, aspect, TWI, curvature), ESA Worldcover land-cover fractions |
| `02_climatological_features.ipynb` | Raw CHELSA download + bioclimatic variables (Bio1, Bio4, Bio12, Bio15, Bio18) |
| `03_biological_features.ipynb` | Canopy height, NDVI, Human Footprint Index, forest-edge distance |
| `04_quality_check.ipynb` | Visual QC of all predictor layers + feature correlation matrix |
| `06_combine_features.ipynb` | Preprocess (aspect sin/cos, curvature smoothing) and export combined feature stack |
| `07_feature_selection.ipynb` | RF-based feature importance ranking on 6 representative species |

### SOM Clustering & Patch Network

| Notebook | Description |
|---|---|
| `10_som_training.ipynb` | SOM training (somoclu) with hyperparameter analysis |
| `11_som_mapping.ipynb` | BMU assignment, Ward linkage (layer 2), landscape type mapping |
| `12_patch_network.ipynb` | Connected components, small patch merging, adjacency graph construction |

### Species Distribution Modelling

| Notebook | Description |
|---|---|
| `13_baseline_rf.ipynb` | Random Forest baseline for all qualifying species (3,756) |
| `20_gnn_architecture_search.ipynb` | GraphSAGE architecture comparison on 6 key species |
| `21_gnn_selected_species.ipynb` | GNN-SDM deep dive: 6 common + 6 vulnerable species |
| `22_gnn_sdm_production.ipynb` | Full GNN-SDM training for all species + RF comparison |

## Features (21 total)

**Terrain** (30 m native): elevation, slope, aspect (sin/cos), TWI, profile curvature, plan curvature

**Land cover** (10 m → 30 m): tree cover, grassland, cropland, built-up, snow/ice, water (as fractions)

**Climate** (~1 km → 30 m): Bio1, Bio4, Bio12, Bio15, Bio18

**Biological / anthropogenic**: canopy height (10 m → 30 m), NDVI annual max (250 m → 30 m), Human Footprint Index (1 km → 30 m), forest-edge distance (30 m native)

## Setup

```bash
cp config.py.example config.py
# Edit config.py with your S3 bucket name
pip install -r requirements.txt
```

## Shared modules

- `geo_utils.py` — DEM loading, terrain features, land-cover fractions, canopy height, NDVI, HFP
- `plot_utils.py` — Map plotting utilities (plot_raster, plot_roi, plot_landcover_fractions)
- `s3_utils.py` — S3 Zarr save/load, file streaming
- `config.py` — S3 paths (gitignored)

## Data sources

- [GBIF](https://www.gbif.org/) — species occurrences
- [Copernicus GLO-30 DEM](https://registry.opendata.aws/copernicus-dem/) — elevation
- [ESA Worldcover v200](https://esa-worldcover.org/) — land cover (10 m)
- [CHELSAch](https://www.chelsa-climate.org/) — high-res Swiss climatologies
- [ETH Global Canopy Height](https://doi.org/10.3929/ethz-b-000609802) — canopy height (10 m, Lang et al. 2023)
- [GIMMS MODIS NDVI](https://gimms.gsfc.nasa.gov/MODIS/) — vegetation index (250 m)
- [Human Footprint Index](https://doi.org/10.6084/m9.figshare.16571064) — anthropogenic pressure (1 km, Mu et al. 2022)

## Reference

Wu, Z., Wang, J., Wu, H., Li, S., & Dai, W. (2025). GNN-SDM: A Graph Neural Network-Based Framework Integrating Complex Landscape Patterns Into Species Distribution Modelling. *Global Ecology and Biogeography*, 34, e70162. https://doi.org/10.1111/geb.70162

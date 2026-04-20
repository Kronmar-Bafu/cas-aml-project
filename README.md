# GNN-SDM Switzerland

Species Distribution Modelling using Graph Neural Networks, applied to Swiss plant species.

Based on the GNN-SDM framework by [Wu et al. (2025)](https://doi.org/10.1111/geb.70162), adapted for a regional study at 30 m resolution.

## Pipeline

| Notebook | Description |
|---|---|
| `00_gbif.ipynb` | Download and filter Swiss plant occurrences from GBIF |
| `01_geographic_features.ipynb` | DEM, terrain derivatives (slope, aspect, TWI, curvature), ESA Worldcover land-cover fractions |
| `02_climatological_features.ipynb` | Raw CHELSA download + bioclimatic variables (Bio1, Bio4, Bio12, Bio15, Bio18) |
| `03_biological_features.ipynb` | Canopy height, NDVI, Human Footprint Index, forest-edge distance |
| `04_quality_check.ipynb` | Visual QC of all predictor layers for selected regions |

### ML Pipeline

| Notebook | Description |
|---|---|
| `06_combine_features.ipynb` | Preprocess (aspect sin/cos, curvature smoothing) and export combined feature stack |
| `10_som_clustering.ipynb` | SOM landscape patch clustering (somoclu) |

## Features (21 total)

**Terrain** (30 m native): elevation, slope, aspect (sin/cos), TWI, profile curvature (smoothed), plan curvature (smoothed)

**Land cover** (10 m → 30 m): tree cover, grassland, cropland, built-up, snow/ice, water (as fractions)

**Climate** (~1 km → 30 m): Bio1, Bio4, Bio12, Bio15, Bio18

**Biological / anthropogenic**: canopy height (10 m → 30 m), NDVI annual max (250 m → 30 m), Human Footprint Index (1 km → 30 m), forest-edge distance (30 m native)

## Setup

```bash
cp config.py.example config.py
# Edit config.py with your S3 bucket name
```

## Shared modules

- `geo_utils.py` — DEM loading, terrain features, SOM clustering, land-cover fractions, canopy height, NDVI, HFP
- `plot_utils.py` — Map plotting utilities
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

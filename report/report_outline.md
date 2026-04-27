# GNN-SDM Switzerland — Report Outline

## 1. Abstract (~200 words)
- Problem: SDMs rely on point data, miss landscape structure
- Approach: Adapted GNN-SDM (Wu et al. 2025) for Swiss flora at 30 m resolution
- Key result: GNN vs RF comparison across 3,756 species, focus on vulnerable species
- Conclusion: [fill after results]

## 2. Introduction (1.5 pages)
- Biodiversity loss and habitat fragmentation in Switzerland
- Traditional SDMs: strengths and limitations (point-based, no spatial context)
- GNN-SDM framework: how graph structure captures landscape interactions
- Our contribution: regional adaptation at 30 m, flora-specific architecture tuning
- Research questions:
  1. Can GNN-SDM outperform traditional RF at patch-level SDM for Swiss plants?
  2. Does the graph structure help more for rare/vulnerable species?
  3. What architecture works best for flora (vs the paper's fauna-oriented design)?

## 3. Data & Study Area (2 pages)

### 3.1 Study area
- Switzerland + border buffer (CHELSAch extent)
- 30 m master grid from Copernicus GLO-30 DEM

### 3.2 Environmental features (21 total)
- Table: Feature | Source | Native resolution | Regridding method
- **Terrain** (6): elevation, slope, aspect (sin/cos), TWI, profile/plan curvature
- **Land cover** (6): ESA Worldcover fractions (tree, grass, crop, built-up, snow, water)
- **Climate** (5): CHELSAch Bio1, Bio4, Bio12, Bio15, Bio18
- **Biological/anthropogenic** (4): canopy height, NDVI, HFP, forest-edge distance
- Preprocessing: aspect sin/cos encoding, curvature smoothing (5×5)
- **Figure**: Feature correlation matrix

### 3.3 Species occurrence data
- GBIF: 14M records, 7,912 species within study extent
- Filtering: ≥100 records → 3,756 qualifying species
- **Figure**: Distribution of records per species (histogram)

### 3.4 Feature selection
- RF importance ranking on 6 representative species
- Selected 12 of 21 features for SOM/GNN
- **Figure**: Feature importance heatmap

## 4. Methods (3 pages)

### 4.1 Landscape patch construction
#### 4.1.1 SOM clustering
- Two-layer approach: SOM (somoclu, 64×64 hexagonal) + Ward linkage → 30 types
- Training: 500K subsample, RobustScaler + domain post-processing
- Hyperparameters: 150 epochs, radius decay, learning rate 0.05→0.016
- **Figure**: SOM codebook feature maps + dendrogram

#### 4.1.2 Connected components and patch merging
- skimage connected-component labelling on landscape types
- Small patch merging (< 100 pixels) via mapping-table approach
- Result: 166K patches
- **Figure**: Landscape patches (Switzerland + Bern ROI)

#### 4.1.3 Patch network
- Adjacency from shared boundaries
- Node attributes: mean environmental features per patch
- NetworkX graph: 166K nodes, 465K edges
- **Figure**: Degree distribution, patch size distribution

### 4.2 Baseline: Random Forest
- Per-species binary classification (presence vs 3× pseudo-absence)
- Hyperparameter tuning on 6 key species (GridSearchCV)
- 5-fold cross-validation, 8 metrics (AUC, accuracy, precision, recall, F1, MCC, Kappa, TSS)

### 4.3 GNN-SDM
#### 4.3.1 Model architecture
- GraphSAGE with mean aggregator
- Architecture search: 5 variants on 6 species
- Best for flora: [64, 48, 32] (wider than paper's [24, 18, 8])
- LeakyReLU, dropout 0.2, sigmoid output
- **Figure**: Architecture comparison heatmap

#### 4.3.2 Training procedure
- Per-species: One-Class SVM background selection, PageRank weighting
- Adam optimizer, lr=0.001, weighted MSE loss
- Early stopping (patience=50), max 500 epochs
- **Figure**: Training curves (loss + AUC) for selected species

## 5. Results (2.5 pages)

### 5.1 RF baseline performance
- Summary statistics across 3,756 species
- **Figure**: AUC distribution histogram
- **Table**: Mean metrics (AUC, TSS, MCC, etc.)

### 5.2 GNN-SDM performance
- Same metrics, same species set
- **Figure**: GNN vs RF AUC scatter plot (diagonal = equal)
- **Table**: Side-by-side metric comparison

### 5.3 Common vs vulnerable species
- 6 common + 6 vulnerable species deep dive
- **Figure**: Bar chart (common blue, vulnerable coral)
- **Table**: Per-species results with habitat type
- Key finding: does GNN help more for rare species?

### 5.4 Habitat suitability maps
- Example suitability maps for 2-3 species (Bern ROI)
- **Figure**: Side-by-side RF vs GNN suitability for same species

## 6. Discussion (1.5 pages)

### 6.1 Architecture adaptation for flora
- Why wider networks work better than the paper's design
- Fewer hops needed? (plants don't disperse like animals)
- Comparison with paper's results on virtual species

### 6.2 Value of graph structure
- Where GNN outperforms RF (species with patchy distributions?)
- Where RF is sufficient (ubiquitous species?)

### 6.3 Limitations
- 30 m resolution: SOM fragmentation, computational cost
- GBIF sampling bias (roads, cities)
- Single time snapshot (no temporal dynamics)
- No independent validation (cross-validation only)

### 6.4 Future work
- Multi-species GNN (shared representations)
- Temporal dynamics (climate change scenarios)
- Transfer learning to other Alpine regions

## 7. Conclusion (~200 words)
- Summary of key findings
- Practical implications for Swiss conservation

## References (~15-20 citations)
- Wu et al. 2025 (GNN-SDM)
- Lang et al. 2023 (canopy height)
- Mu et al. 2022 (HFP)
- GBIF, CHELSAch, ESA Worldcover, Copernicus DEM
- Hamilton et al. 2017 (GraphSAGE)
- Fey & Lenssen 2019 (PyG)
- Kohonen 1982 (SOM)
- Céréghino & Park 2009 (SOM BMU rule)
- + standard SDM references (MaxEnt, Elith & Leathwick, etc.)

---

## Suggested key figures (10-12 total)
1. Feature correlation matrix (notebook 04)
2. Feature importance heatmap (notebook 07)
3. SOM landscape types map — Switzerland (notebook 11)
4. Dendrogram (notebook 11)
5. Landscape patches — Bern ROI (notebook 12)
6. Architecture search heatmap (notebook 20)
7. Training curves — loss + AUC (notebook 21)
8. RF AUC distribution (notebook 13)
9. GNN vs RF scatter plot (notebook 22)
10. Common vs vulnerable bar chart (notebook 21)
11. Habitat suitability map example (notebook 22)
12. Patch size + degree distributions (notebook 12)

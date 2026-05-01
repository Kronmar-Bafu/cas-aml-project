# Conservation Prioritisation — Future Work Ideas

## Context

Once the GNN-SDM produces per-species habitat suitability scores for all 166K patches,
we can leverage the graph structure to identify patches that are critical for
conservation — not just because they're suitable, but because they're structurally
important for connectivity.

## Approach 1: Betweenness Centrality on Suitability Subgraph

**Idea**: Patches with high betweenness centrality in the "suitable habitat network"
are bottleneck corridors — their loss would fragment populations.

**Steps**:
1. For each species, threshold suitability (e.g., >0.5) to get "suitable" patches
2. Extract the subgraph of suitable patches (keep only edges between suitable nodes)
3. Compute betweenness centrality on this subgraph
4. Patches with high betweenness = critical corridors

**Implementation**: `nx.betweenness_centrality(G_suitable)` — already available in NetworkX.

**Output**: Per-species corridor maps; aggregate across species for multi-species corridors.

## Approach 2: Multi-Species Cumulative Suitability

**Idea**: Stack suitability maps across all (or vulnerable) species to find hotspots
where many species co-occur with high suitability.

**Steps**:
1. For each species, get the suitability vector (166K values)
2. Sum across species: `cumulative[patch] = Σ suitability[species, patch]`
3. High-cumulative patches = multi-species hotspots

**Reference**: Wu et al. (2025) did this for 199 threatened mammals in South America,
producing a "continental-scale habitat hotspot map."

**Output**: A single map showing where conservation actions benefit the most species.

## Approach 3: Connectivity Removal Experiments

**Idea**: Systematically remove patches and measure how much graph connectivity drops.
Patches whose removal causes the largest fragmentation are critical links.

**Steps**:
1. Build the suitability-weighted subgraph (edges weighted by min suitability of endpoints)
2. For each candidate patch, compute the number of connected components after removal
3. Patches that increase fragmentation the most = highest priority for protection

**Caveat**: O(n²) for brute-force; can be approximated with random sampling or
limited to high-betweenness candidates from Approach 1.

## Approach 4: PageRank on Suitability-Weighted Graph

**Idea**: Weight edges by ecological connectivity (min suitability of connected patches).
PageRank on this weighted graph identifies patches that are both structurally central
AND ecologically suitable.

**Steps**:
1. Create edge weights: `w(i,j) = min(suitability[i], suitability[j])`
2. Compute PageRank on the weighted graph
3. High PageRank = important hub in the suitable habitat network

**Advantage**: Combines structural importance with ecological quality in one metric.

## Practical Considerations

- **Species selection**: Focus on the 6 vulnerable species (Cypripedium, Drosera,
  Pulsatilla, Gladiolus, Aquilegia, Stipa) for the most conservation-relevant results.
- **Threshold sensitivity**: Test multiple suitability thresholds (0.3, 0.5, 0.7)
  to check robustness of identified corridors.
- **Validation**: Compare identified corridors with known Swiss ecological corridors
  (e.g., from the Swiss Ecological Network REN).
- **Visualisation**: Map critical patches on the Bern ROI and full Switzerland,
  colored by priority score.

## Suggested Notebook: `30_conservation_prioritisation.ipynb`

Would implement Approaches 1 + 2 as the minimum viable analysis, with 3 + 4 as
extensions if time permits.


---

# Additional Future Work Ideas (from literature)

## 5. Transfer Learning Across Regions

**Idea**: Train on Switzerland (data-rich), fine-tune on neighboring Alpine regions
with sparse occurrence data (Tyrol, Savoie, Lombardy).

**Reference**: Wu et al. (2025): "cross-regional generalisation could be explored
through transfer learning, where knowledge from well-sampled species or regions is
leveraged to improve predictions for data-limited species elsewhere."

**Steps**:
1. Train GNN-SDM on Swiss graph (full pipeline as-is)
2. Construct a patch graph for the target region (same SOM features)
3. Fine-tune only the last 1–2 layers on the target region's sparse data
4. Evaluate whether transfer improves over training from scratch

**Value**: Enables SDM for regions where GBIF data is sparse but environmental
data is available (most of the Alps).

## 6. Multi-Species GNN (Heterogeneous Graph)

**Idea**: Instead of one model per species, build a bipartite graph with species
nodes and patch nodes. Edges encode presence. The GNN learns shared environmental
responses across species.

**Reference**: "Heterogeneous graph neural networks for species distribution
modeling" (arxiv 2503.11900, 2025) — treats species and locations as two distinct
node sets, predicting detection records as edges.

**Advantage**: Rare species benefit from patterns learned from common species.
One model instead of 3,756.

**Challenge**: Requires rethinking the graph structure and training procedure.
Not a drop-in replacement.

## 7. Temporal Dynamics / Climate Change Scenarios

**Idea**: Train separate SOMs/GNNs for different time periods or climate scenarios
to predict how habitat suitability shifts.

**Data sources**:
- CHELSA future projections (CMIP6 scenarios)
- MODIS NDVI time series (2000–2024) for vegetation change
- ESA Worldcover 2020 vs 2021 for land-use change

**Output**: Maps showing where species will gain/lose suitable habitat under
warming scenarios. Critical for proactive conservation planning.

## 8. Uncertainty Quantification via Ensemble

**Idea**: Vary SOM parameters (BMU count, epochs, random seed) and GNN
hyperparameters to produce an ensemble of predictions. Map the variance
spatially to identify where predictions are uncertain.

**Reference**: Wu et al. (2025) varied BMU count from 1√n to 20√n and computed
95% confidence intervals. "Maps of CI width revealed that variations were mainly
concentrated in transitional regions of predicted suitability."

**Steps**:
1. Train 10–20 GNN models with different SOM seeds / BMU counts
2. For each patch, compute mean and std of suitability across ensemble
3. Map CI width — high uncertainty = transitional zones needing more data

## 9. GNN Explainability

**Idea**: Understand which neighbours and features the GNN relies on for each
species prediction.

**Methods**:
- **GNNExplainer** (Ying et al. 2019): identifies important subgraph and features
  for each prediction
- **GAT (Graph Attention Networks)**: replace GraphSAGE with attention-based
  aggregation — attention weights directly show which neighbours matter
- **Feature ablation**: remove one feature at a time, measure AUC drop

**Value**: Reveals which landscape transitions matter most for each species.
E.g., "Cypripedium calceolus suitability depends heavily on adjacent forest patches"
— directly actionable for conservation.

**Implementation**: PyG has `torch_geometric.explain.GNNExplainer` built in.

---

## Priority ranking (effort vs impact)

| Idea | Effort | Impact | Recommended? |
|---|---|---|---|
| Conservation prioritisation (1–4) | Low | High | ✅ Do first |
| Uncertainty ensemble (8) | Medium | High | ✅ Good for report |
| GNN explainability (9) | Medium | High | ✅ Insightful |
| Multi-species GNN (6) | High | High | ⚡ If time permits |
| Transfer learning (5) | High | Medium | 📋 Future paper |
| Temporal dynamics (7) | High | High | 📋 Future paper |

# Next Steps — Gap Analysis Species Scope

## Context

Only **23 threatened species** (NT/VU/EN/CR) have GNN-SDM suitability scores
(they needed ≥100 GBIF records to be trained). This feels low but is comparable
to published regional studies.

## Literature reference points

| Study | Species count | Scope |
|-------|--------------|-------|
| USGS Gap Analysis Project | 2,000+ | National (US), decades of funding |
| Rodrigues et al. 2004 (Nature) | ~11,000 | Global, all mammals/turtles/amphibians/threatened birds |
| Socio-ecological gap analysis (PNAS 2022) | 91 | African carnivores |
| Stacked SDMs for rare plants, Ontario | 22 | Regional, rare plants — very similar to ours |
| European wetland gap analysis (Natura 2000) | hundreds | Continental |

**Takeaway:** 23 species is on the low end but defensible for a regional CAS project,
especially given the data constraint (≥100 records needed for reliable SDM training).
The Ontario study (22 rare plants) is a direct comparable.

## Options to consider

### Option A: Run gap analysis for 23 threatened species only (~12 min)
- Pros: fast, conservation-relevant, defensible scope
- Cons: small sample, limited statistical power for aggregate patterns

### Option B: Add a species-richness context layer (no circuit-theory needed)
- Compute mean suitability richness = sum of thresholded suitability across all
  3,751 species per patch
- Gives a "predicted species richness" surface to compare against the
  threatened-species connectivity map
- Cheap to compute (vectorised NumPy, no Circuitscape)
- Strengthens the narrative without 10h batch job

### Option C: Full circuit-theory for all ~1,200 LC species (~10h batch job)
- Provides a baseline comparison: are threatened-species corridors different
  from general-biodiversity corridors?
- Expensive but the batch job infrastructure is ready

### Option D: Lower min-records threshold and retrain GNN-SDM
- Many threatened species have 20–99 records
- Could recover more CR/EN/VU species for the gap analysis
- Risk: SDM quality degrades with fewer training points
- Would need to validate model performance at lower thresholds

## Recommended approach (to decide after marinating)

A pragmatic two-tier strategy:
1. **Primary analysis** — full circuit-theory for the 23 threatened species
2. **Context layer** — simple suitability-richness surface from all 3,751 species
   (Option B) to show where overall biodiversity hotspots align or diverge from
   threatened-species corridors

This is comparable to published work, computationally feasible, and tells a
compelling conservation story without requiring a 10h batch job.

## GBIF API note (May 2025)

The GBIF Species API no longer returns `iucnRedListCategory` in the main
`/v1/species/{key}` response. The correct endpoint is now:
```
GET /v1/species/{key}/iucnRedListCategory
```
Returns `category` field with full names: `CRITICALLY_ENDANGERED`, `ENDANGERED`,
`VULNERABLE`, `NEAR_THREATENED`, `LEAST_CONCERN`, `DATA_DEFICIENT`,
`NOT_EVALUATED`, `EXTINCT`, `EXTINCT_IN_THE_WILD`.

HTTP 204 = no IUCN assessment available.

Notebook 32 has been updated accordingly.

## Notes from presentation
- look at rare species more: when we have only a couple of patches, is the graph really add value?
- for very few occurences, RF instead of GNN?
- play around with the cut-off of 0.5 - maybe that's too arbitrary?

## Cypripedium calceolus: RF vs GNN case study

Compare RF and GNN suitability predictions for *Cypripedium calceolus* specifically.
This species is a good test case because:

- It's vulnerable and persists in small, isolated forest-edge populations
- Limited seed dispersal + mycorrhizal dependency means connectivity between patches matters
- A point-based model (RF) might flag isolated clearings as suitable even if they're
  unreachable from existing populations
- The GNN should downweight isolated suitable patches and produce more realistic
  (connected) suitability surfaces

**What to check:**
- Do RF and GNN agree on core habitat areas?
- Does the GNN predict lower suitability for isolated patches that RF rates highly?
- Spatial pattern: is the GNN prediction more spatially coherent / clustered?
- Overlay with known populations (GBIF points) — which model better captures the
  fragmented distribution pattern?

This could make a compelling figure in the report (Discussion section 6.2: "Value of graph structure")
and directly supports the introduction's argument about why spatial context matters for SDMs.

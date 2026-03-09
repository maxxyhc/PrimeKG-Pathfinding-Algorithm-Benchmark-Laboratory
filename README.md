# PrimeKG Pathfinding Algorithm Benchmark Laboratory

A benchmarking framework for evaluating graph pathfinding algorithms on the task of **drug mechanism-of-action (MoA) discovery** using the [PrimeKG](https://github.com/mims-harvard/PrimeKG) biomedical knowledge graph.

Given a drug and a disease, can a graph algorithm recover the *mechanistic pathway* — the chain of proteins, biological processes, and anatomical structures that explains *how* the drug treats the disease?

## Overview

We evaluate **8 pathfinding algorithms across 2 phases** against **150 curated ground truth pathways** (4–11 nodes each) extracted from [DrugMechDB](https://drugmechdb.github.io/) and mapped onto PrimeKG.

### Algorithms

#### Phase 1 — Edge Weighting & Constrained Search

| # | Algorithm | Strategy | Avg Time/Pathway |
|---|-----------|----------|-----------------|
| 1 | **Dijkstra** | Unweighted baseline (NetworkX shortest path) | 0.03 ms |
| 2 | **Meta-Path BFS** | BFS constrained to valid biological edge-type sequences | — |
| 3 | **Hub-Penalized** | Penalizes high-degree hub nodes (`w = 1 + α·log(degree)`) | 2,049 ms |
| 4 | **PageRank-Inverse** | Prefers low-centrality nodes (`w = 1/(1 + PageRank)`) | 4,929 ms |
| 5 | **Semantic Bridging** | TF-IDF cosine similarity edge weighting | 3,394 ms |

#### Phase 2 — Search Strategy

| # | Algorithm | Strategy |
|---|-----------|----------|
| 6 | **Bidirectional ★** | Forward + backward Dijkstra; meets in the middle |
| 7 | **K-Shortest + Bio** | k=4 candidate paths (Yen's algorithm), re-ranked by biological plausibility |
| 8 | **Bidir + Relation Weighting** | Bidirectional search with ground-truth-derived relation weights |

### Key Results (150 pathways, ≥ 4 nodes)

#### Phase 1 — Edge Weighting

| Metric | Dijkstra | Hub-Penalized | PageRank-Inverse | Semantic Bridging |
|--------|:---:|:---:|:---:|:---:|
| **F1 Score** ↑ | 0.545 | 0.540 | 0.547 | **0.557** |
| **Edit Distance** ↓ | 0.615 | 0.551 | 0.544 | **0.537** |

> Phase 1 F1 spread = 0.017. Edge weighting alone does not significantly differentiate algorithms — graph topology dominates.

#### Phase 2 — Search Strategy

| Metric | Bidirectional ★ | K-Shortest + Bio | Bidir + Rel. Wt |
|--------|:---:|:---:|:---:|
| **F1 Score** ↑ | **0.571** | 0.555 | 0.549 |
| **Edit Distance** ↓ | **0.429** | 0.445 | 0.451 |

> Bidirectional search produces the first statistically significant improvement. Edit distance gain vs. Dijkstra: −0.186 (p < 0.000001, Cohen's d = −1.334).

---

## Repository Structure

```
.
├── benchmark_runner.py              ← Main entry point: loads data, builds graph,
│                                      runs all 8 algorithms, evaluates, saves results
├── src/                             ← Core modules
│   ├── Algorithms.py                ← All 8 pathfinding algorithm implementations
│   ├── evaluation_metrics.py        ← 7 evaluation metrics (F1, edit distance, MRR, etc.)
│   ├── evaluation_helpers.py        ← Helper functions (degree counts, hub threshold, timing)
│   └── prepare_primekg.py           ← PrimeKG data cleaning for Neo4j export
│
├── data/                            ← All data files
│   ├── benchmark_pathways_nodes.csv ← Ground truth pathways (included in repo)
│   ├── benchmark_pathways_edges.csv ← Ground truth pathway edges (included in repo)
│   ├── benchmark_pathways_metadata.csv
│   ├── nodes.csv                    ← PrimeKG nodes (NOT in repo — download below)
│   ├── edges.csv                    ← PrimeKG edges (NOT in repo — download below)
│   └── raw/                         ← Additional raw data (NOT in repo)
│       ├── indication_paths.yaml    ← DrugMechDB source pathways
│       ├── kg.csv                   ← Raw PrimeKG knowledge graph
│       ├── mesh_to_mondo_lookup.csv ← MeSH → Mondo disease ID mapping
│       └── uniprot_to_entrez_lookup.csv ← UniProt → Entrez gene ID mapping
│
├── Ground_Truth_automation/         ← Ground truth extraction pipeline
│   ├── automated_pipeline.py        ← End-to-end pathway extraction
│   ├── pathway_config.yaml          ← Configuration for pathway mapping
│   ├── pathway_validation_demo.ipynb← Demo notebook for pathway validation
│   └── pathway_validator.py         ← Validates extracted pathways against PrimeKG
│
├── tests/                           ← Unit tests
│   ├── test_evaluation_helpers.py
│   └── test_evaluation_metrics.py
│
├── notebook/                        ← Intermediate algorithm output files
│   ├── bidir_paths.csv              ← Bidirectional search predictions
│   └── brw_paths.csv                ← Bidir + Relation Weighted predictions
│
├── index.html                       ← Interactive results dashboard
├── results.html                     ← Results visualization page
├── groundtruths.html                ← Ground truth pathway viewer
├── style.css                        ← Dashboard styling
├── subgraph_with_pathways.json      ← Subgraph data for visualization
├── requirements.txt                 ← Python dependencies
└── README.md
```

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/maxxyhc/PrimeKG-Pathfinding-Algorithm-Benchmark-Laboratory.git
cd PrimeKG-Pathfinding-Algorithm-Benchmark-Laboratory
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

Requires Python 3.9+. Core dependencies: `pandas`, `numpy`, `networkx`, `scikit-learn`, `scipy`.

### 3. Download PrimeKG data

The raw PrimeKG files are too large to store in the repository. Download them directly from Harvard Dataverse:

**[→ Harvard Dataverse: PrimeKG](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/IXA7BM)**

Download these two files:
- `nodes.csv` (~5 MB)
- `edges.csv` (~60 MB)

Place them in `data/`:

```
data/
├── nodes.csv      ← from Harvard Dataverse
└── edges.csv      ← from Harvard Dataverse
```

**Expected formats:**

`nodes.csv`:
```csv
node_index,node_id,node_type,node_name,node_source
0,9796,gene/protein,PHYHIP,NCBI
1,7918,gene/protein,GPANK1,NCBI
```

`edges.csv`:
```csv
relation,display_relation,x_index,y_index
protein_protein,ppi,0,8889
protein_protein,ppi,1,2798
```

### 4. Ground truth pathways (already in the repo)

The curated ground truth pathways are included — no extra download needed:

| File | Description |
|------|-------------|
| `data/benchmark_pathways_nodes.csv` | 343 pathways (1,456 nodes total); 150 with ≥ 4 nodes used for benchmarking |
| `data/benchmark_pathways_edges.csv` | 1,113 annotated pathway edges |
| `data/benchmark_pathways_metadata.csv` | Pathway metadata (drug, disease, path length) |

These were extracted from [DrugMechDB](https://drugmechdb.github.io/) and mapped onto PrimeKG node indices using the pipeline in `Ground_Truth_automation/`.

### 5. Run the benchmark

```bash
python benchmark_runner.py
```

This will:
1. Load PrimeKG and ground truth data
2. Filter to pathways with ≥ 4 nodes (150 pathways)
3. Build the knowledge graph (~129K nodes, ~8M edges)
4. Run all 8 algorithms on all 150 pathways
5. Evaluate with 7 metrics per algorithm per pathway
6. Save results to `results/` (predictions, detailed results, summary by algorithm)

Logs are saved to `logs/` with timestamps.

### 6. Run tests

```bash
python -m pytest tests/ -v
```

---

## Evaluation Metrics

| Category | Metric | Description |
|----------|--------|-------------|
| **Node Overlap** | Precision | Fraction of predicted nodes that are in ground truth |
| | Recall | Fraction of ground truth nodes recovered |
| | F1 Score | Harmonic mean of precision and recall |
| **Sequence Quality** | Edit Distance | Normalized edit distance between node sequences (0 = perfect) |
| | MRR | Mean Reciprocal Rank of first correct intermediate node |
| **Path Characteristics** | Hub Node Ratio | Fraction of path nodes that are high-degree hubs (lower is better) |
| | Path Length Accuracy | How close predicted length is to ground truth length (1 = exact) |

---

## Key Findings

- **Edge weighting alone doesn't matter.** Phase 1 F1 spread = 0.017 across algorithms. Graph topology dominates over weighting strategy.
- **Bidirectional search is the first real improvement.** Searching from both drug and disease simultaneously cuts edit distance by 0.186 vs. Dijkstra (p < 0.000001). Adding biological signals on top made results *worse* — the search strategy already implicitly captures mechanistic grammar.
- **The core problem is path length.** Average predicted path = 3.4 nodes; average ground truth = 5.8 nodes. 68% of bidirectional paths are exactly 3 nodes. The task effectively reduces to picking one intermediate.
- **Hub shortcuts are the dominant failure mode.** 43% of failures involve routing through high-degree hub nodes (e.g., Seizure: 4,218 connections) that are structurally close to everything but mechanistically irrelevant.
- **The first edge determines everything.** Paths starting with `drug_protein` achieve F1 = 0.604; paths starting with `drug_effect` achieve F1 = 0.437 (gap: +0.167, p = 0.000004). All 30 drugs that route through side effects *have* protein edges available — the algorithm chooses the side-effect route because it's shorter.
- **Performance degrades sharply with pathway length.** F1 drops from ~0.66 (4-node pathways) to ~0.26 (10-node pathways). No algorithm reliably reconstructs long mechanistic chains.
- **Biological constraints help selectively.** Meta-Path BFS enforces valid edge-type sequences but sacrifices recall; K-Shortest + Bio scoring re-ranks candidate paths by node type diversity and relation quality.

---

## Ground Truth Construction

Ground truth pathways were sourced from [DrugMechDB](https://drugmechdb.github.io/), a curated database of drug mechanism-of-action pathways. Each pathway was:

1. Parsed from DrugMechDB's YAML format
2. Mapped onto PrimeKG using identifier lookups (DrugBank → PrimeKG drug index, UniProt → PrimeKG protein index, MeSH/MONDO → PrimeKG disease index)
3. Validated to ensure all nodes and edges exist in PrimeKG
4. Filtered to retain only fully-mapped pathways

The extraction and mapping pipeline is in `Ground_Truth_automation/`.

## Interactive Dashboard

Open `index.html` in a browser to explore results interactively, including per-pathway comparisons, metric breakdowns, and ground truth visualization.

---

## Known Limitations

- **Shortest path is degenerate** — uses `indication` shortcut edges on 100% of pathways, producing 2-node direct paths rather than mechanistic routes
- **Performance degrades on long pathways** — F1 drops from ~0.65 (4 nodes) to ~0.30 (10+ nodes); no algorithm reliably reconstructs long mechanistic chains
- **Side-effect routing** — weighted algorithms frequently route through phenotype/side-effect edges rather than the true molecular mechanism
- **Bidirectional edges** — the graph treats all edges as bidirectional with identical relation labels, erasing biological directionality

---

## Citation

If you use this benchmark, please cite:

- **PrimeKG:** Chandak, P., Huang, K., & Zitnik, M. (2023). Building a knowledge graph to enable precision medicine. *Scientific Data*, 10(1), 67.
- **DrugMechDB:** Mayers, M., et al. (2022). DrugMechDB: A Curated Database of Drug Mechanisms of Action. *Scientific Data*, 9(1), 648.

## License

This project uses PrimeKG data licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Ground truth pathways are derived from [DrugMechDB](https://drugmechdb.github.io/).
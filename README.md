# PrimeKG Pathfinding Algorithm Benchmark Laboratory

A benchmarking framework for evaluating graph pathfinding algorithms on the task of **drug mechanism-of-action (MoA) discovery** using the [PrimeKG](https://github.com/mims-harvard/PrimeKG) biomedical knowledge graph.

Given a drug and a disease, can a graph algorithm recover the *mechanistic pathway* — the chain of proteins, biological processes, and anatomical structures that explains *how* the drug treats the disease?

## Overview

We evaluate **8 pathfinding algorithms across 2 phases** against **150 curated ground truth pathways** (4–11 nodes each) extracted from [DrugMechDB](https://drugmechdb.github.io/) and mapped onto PrimeKG.

### Algorithms

#### Phase 1 — Edge Weighting

| # | Algorithm | Strategy | Avg Time/Pathway |
|---|-----------|----------|-----------------|
| 1 | **Dijkstra** | Unweighted baseline | 0.03 ms |
| 2 | **Hub-Penalized** | Penalizes high-degree hub nodes (`w = 1 + α·log(degree)`) | 2,049 ms |
| 3 | **PageRank-Inverse** | Prefers low-centrality nodes (`w = 1/(1 + PageRank)`) | 4,929 ms |
| 4 | **Learned A\*** | Spectral embeddings + MLP-learned edge weights + A* search | 38,072 ms |
| 5 | **Semantic Bridging** | TF-IDF cosine similarity edge weighting | 3,394 ms |

#### Phase 2 — Search Strategy

| # | Algorithm | Strategy |
|---|-----------|----------|
| 6 | **Bidirectional ★** | Forward + backward Dijkstra; meets in the middle |
| 7 | **K-Shortest + Bio** | k=4 candidate paths, re-ranked by biological plausibility |
| 8 | **Bidir + Relation Weighting** | Bidirectional search with enrichment-informed edge weights |

### Key Results (150 pathways, ≥ 4 nodes)

#### Phase 1 — Edge Weighting

| Metric | Dijkstra | Hub-Penalized | PageRank-Inverse | Learned A* | Semantic Bridging |
|--------|:---:|:---:|:---:|:---:|:---:|
| **F1 Score** ↑ | 0.545 | 0.540 | 0.547 | 0.460 | **0.557** |
| **Edit Distance** ↓ | 0.615 | 0.551 | 0.544 | 0.617 | **0.537** |

> Phase 1 F1 spread = 0.023. Edge weighting alone does not significantly differentiate algorithms — graph topology dominates.

#### Phase 2 — Search Strategy

| Metric | Bidirectional ★ | K-Shortest + Bio | Bidir + Rel. Wt |
|--------|:---:|:---:|:---:|
| **F1 Score** ↑ | **0.571** | 0.555 | 0.549 |
| **Edit Distance** ↓ | **0.429** | 0.445 | 0.451 |

> Bidirectional search produces the first statistically significant improvement. Edit distance gain vs. Dijkstra: −0.186 (p < 0.000001, Cohen's d = −1.334).

Full results: `algos_/evaluation_results_all_algorithms.csv`

---

## Repository Structure

```
.
├── notebook/                        ← Main benchmark (run from here)
│   ├── algo_final.ipynb             ← Primary self-contained benchmark notebook
│   ├── Algorithms.py
│   ├── evaluation_helpers.py
│   ├── evaluation_metrics.py
│   ├── evaluation_runner.py
│   └── evaluation_visualization.py
│
├── algos_/                          ← Legacy version (kept for reference)
│   ├── primekg_benchmark_.ipynb
│   ├── evaluation_results_all_algorithms.csv
│   ├── algorithm_summary.csv
│   ├── predictions_*.csv
│   ├── algorithm_comparison.png
│   ├── f1_by_length.png
│   └── timing_comparison.png
│
├── data/                            ← All data files
│   ├── benchmark_pathways_nodes.csv ← Ground truth pathways (included in repo)
│   ├── benchmark_pathways_edges.csv ← Ground truth pathway edges (included in repo)
│   ├── benchmark_pathways_metadata.csv
│   ├── nodes.csv                    ← PrimeKG nodes (NOT in repo — download below)
│   ├── edges.csv                    ← PrimeKG edges (NOT in repo — download below)
│   └── raw/                         ← Additional raw data (NOT in repo)
│       ├── kg.csv
│       ├── mesh_to_mondo_lookup.csv
│       └── uniprot_to_entrez_lookup.csv
│
├── Ground_Truth_automation/         ← Pipeline for extracting pathways from DrugMechDB
│   ├── automated_pipeline.py
│   ├── pathway_validator.py
│   └── pathway_config.yaml
│
├── tests/                           ← Unit tests
│   ├── test_evaluation_helpers.py
│   └── test_evaluation_metrics.py
│
├── src/
│   └── prepare_primekg.py
│
├── pytest.ini
├── requirements.txt
└── README.md
```

> **Note on `algos_/`:** An earlier version of the benchmark kept for reference. The primary notebook is `notebook/algo_final.ipynb`, which is fully self-contained — all algorithms, metrics, and helpers are defined inline.

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

Requires Python 3.9+. Core dependencies: `pandas`, `numpy`, `networkx`, `scikit-learn`, `scipy`, `matplotlib`.

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
jupyter notebook notebook/algo_final.ipynb
```

The notebook will:
1. Load PrimeKG and ground truth data
2. Filter to pathways with ≥ 4 nodes (150 pathways)
3. Build the knowledge graph (~129K nodes, ~8M edges)
4. Run all 8 algorithms on all 150 pathways
5. Evaluate with 9 metrics per algorithm per pathway
6. Generate comparison figures and export CSVs

> **Expected runtime:** ~90–120 minutes total. Learned A* dominates at ~38s/pathway (~95 min). All other algorithms finish in under 15 minutes combined.

### 6. Run tests

```bash
pip install pytest
python -m pytest -v
```

Runs 65 unit tests covering all evaluation metrics and helper functions.

---

## Evaluation Metrics

| Category | Metric | Description |
|----------|--------|-------------|
| **Node Accuracy** | Precision | Fraction of predicted nodes that are in ground truth |
| | Recall | Fraction of ground truth nodes recovered |
| | F1 Score | Harmonic mean of precision and recall |
| **Target Finding** | Hits@1, @3, @5 | Whether the disease appears in the last k predicted nodes |
| **Mechanistic Quality** | Relation Accuracy | Fraction of predicted edge types found in ground truth |
| | Edit Distance | Normalized Levenshtein distance between node sequences (0 = perfect) |
| | Hub Node Ratio | Fraction of path nodes that are high-degree hubs (lower is better) |
| **Efficiency** | Path Length MAE | Absolute difference between predicted and ground truth path length |
| | Speed | Wall-clock time per pathway in milliseconds |

---

## Key Findings

- **Edge weighting alone doesn't matter.** Phase 1 F1 spread = 0.023 across 5 algorithms; 0/10 pairwise comparisons are statistically significant. Graph topology dominates over weighting strategy.
- **Bidirectional search is the first real improvement.** Searching from both drug and disease simultaneously cuts edit distance by 0.186 vs. Dijkstra (p < 0.000001). Adding biological signals on top made results *worse* — the search strategy already implicitly captures mechanistic grammar.
- **The core problem is path length.** Average predicted path = 3.4 nodes; average ground truth = 5.8 nodes. 68% of bidirectional paths are exactly 3 nodes. The task effectively reduces to picking one intermediate.
- **Hub shortcuts are the dominant failure mode.** 43% of failures involve routing through high-degree hub nodes (e.g., Seizure: 4,218 connections) that are structurally close to everything but mechanistically irrelevant.
- **The first edge determines everything.** Paths starting with `drug_protein` achieve F1 = 0.604; paths starting with `drug_effect` achieve F1 = 0.437 (gap: +0.167, p = 0.000004). All 30 drugs that route through side effects *have* protein edges available — the algorithm chooses the side-effect route because it's shorter. Penalizing shortcuts just reveals the next shortcut layer.
- **Performance degrades sharply with pathway length.** F1 drops from ~0.66 (4-node pathways) to ~0.26 (10-node pathways). No algorithm reliably reconstructs long mechanistic chains.

---

## Ground Truth Construction

Ground truth pathways were sourced from [DrugMechDB](https://drugmechdb.github.io/), a curated database of drug mechanism-of-action pathways. Each pathway was:

1. Parsed from DrugMechDB's YAML format
2. Mapped onto PrimeKG using identifier lookups (DrugBank → PrimeKG drug index, UniProt → PrimeKG protein index, MESH/MONDO → PrimeKG disease index)
3. Validated to ensure all nodes and edges exist in PrimeKG
4. Filtered to retain only fully-mapped pathways

The extraction and mapping pipeline is in `Ground_Truth_automation/`.

---

## Known Limitations

- **Shortest path is degenerate** — uses `indication` shortcut edges on 100% of pathways, producing 2-node direct paths rather than mechanistic routes
- **Performance degrades on long pathways** — F1 drops from ~0.65 (4 nodes) to ~0.30 (10+ nodes); no algorithm reliably reconstructs long mechanistic chains
- **Side-effect routing** — weighted algorithms frequently route through phenotype/side-effect edges rather than the true molecular mechanism
- **Bidirectional edges** — the graph treats all edges as bidirectional with identical relation labels, erasing biological directionality
- **Learned A\* data leakage** — currently trains and evaluates on the same pathways without cross-validation

---

## Citation

If you use this benchmark, please cite:

- **PrimeKG:** Chandak, P., Huang, K., & Zitnik, M. (2023). Building a knowledge graph to enable precision medicine. *Scientific Data*, 10(1), 67.
- **DrugMechDB:** Mayers, M., et al. (2022). DrugMechDB: A Curated Database of Drug Mechanisms of Action. *Scientific Data*, 9(1), 648.

## License

This project uses PrimeKG data licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Ground truth pathways are derived from [DrugMechDB](https://drugmechdb.github.io/).
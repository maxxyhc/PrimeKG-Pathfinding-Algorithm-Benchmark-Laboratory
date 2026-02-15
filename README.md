# PrimeKG Pathfinding Algorithm Benchmark Laboratory

A benchmarking framework for evaluating graph pathfinding algorithms on the task of **drug mechanism-of-action (MoA) discovery** using the [PrimeKG](https://github.com/mims-harvard/PrimeKG) biomedical knowledge graph.

Given a drug and a disease, can a graph algorithm recover the *mechanistic pathway* — the chain of proteins, biological processes, and anatomical structures that explains *how* the drug treats the disease?

## Overview

We evaluate **5 pathfinding algorithms** against **150 curated ground truth pathways** (4–11 nodes each) extracted from [DrugMechDB](https://drugmechdb.github.io/) and mapped onto PrimeKG.

### Algorithms

| # | Algorithm | Strategy | Avg Time/Pathway |
|---|-----------|----------|-----------------|
| 1 | **Shortest Path** | Unweighted BFS baseline | 0.03 ms |
| 2 | **Hub-Penalized** | Penalizes high-degree hub nodes (`w = 1 + α·log(degree)`) | 2,049 ms |
| 3 | **PageRank-Inverse** | Prefers low-centrality nodes (`w = 1/(1 + PageRank)`) | 4,929 ms |
| 4 | **Learned A\*** | Spectral embeddings + MLP-learned edge weights + A* search | 38,072 ms |
| 5 | **Semantic Bridging** | TF-IDF cosine similarity edge weighting | 3,394 ms |

### Key Results (150 pathways, ≥ 4 nodes)

| Metric | Shortest Path | Hub-Penalized | PageRank-Inverse | Learned A* | Semantic Bridging |
|--------|:---:|:---:|:---:|:---:|:---:|
| **F1 Score** ↑ | 0.545 | 0.540 | 0.547 | 0.460 | **0.565** |
| **Recall** ↑ | 0.385 | 0.454 | 0.462 | **0.504** | 0.474 |
| **Relation Accuracy** ↑ | 0.307 | 0.546 | 0.477 | 0.583 | **0.629** |
| **Edit Distance** ↓ | 0.615 | 0.551 | 0.544 | 0.617 | **0.531** |
| **Hub Ratio** ↓ | 0.693 | **0.594** | 0.763 | 0.782 | 0.700 |
| **Path Length MAE** ↓ | 3.827 | 2.553 | 2.553 | **2.427** | 2.553 |

**Semantic Bridging** is the best overall performer. Full analysis is in `algos_/evaluation_results_all_algorithms.csv`.

## Repository Structure

```
.
├── algos_/                          ← Main benchmark (run from here)
│   ├── primekg_benchmark_.ipynb     ← Self-contained benchmark notebook
│   ├── evaluation_results_all_algorithms.csv
│   ├── algorithm_summary.csv
│   ├── predictions_*.csv            ← Per-algorithm prediction outputs
│   ├── algorithm_comparison.png
│   ├── f1_by_length.png
│   └── timing_comparison.png
│
├── data/
│   ├── raw/                         ← PrimeKG source files (NOT in repo — see Setup)
│   │   ├── nodes.csv
│   │   ├── edges.csv
│   │   └── ...
│   └── processed/                   ← Ground truth + cleaned data
│       ├── benchmark_pathways_nodes.csv   ← 150 curated pathways (in repo)
│       ├── benchmark_pathways_edges.csv   ← Pathway edge annotations (in repo)
│       └── ...
│
├── notebook/                        ← Legacy multi-file version (kept for reference)
│   ├── algorithm_benchmark.ipynb
│   ├── Algorithms.py
│   ├── evaluation_helpers.py
│   └── evaluation_metrics.py
│
├── Ground_Truth_automation/         ← Pipeline for extracting pathways from DrugMechDB
│   ├── automated_pipeline.py
│   ├── pathway_validator.py
│   └── ...
│
├── src/
│   └── prepare_primekg.py           ← PrimeKG preprocessing script
│
├── requirements.txt
└── README.md
```

### A note on `notebook/`

The `notebook/` directory contains an earlier multi-file version of the benchmark where algorithms, evaluation helpers, and metrics lived in separate `.py` files. It is kept for reference and testing purposes. **The primary notebook is `algos_/primekg_benchmark_.ipynb`**, which is fully self-contained — all algorithms, metrics, and helpers are defined inline with no external imports required.

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/PrimeKG-Pathfinding-Algorithm-Benchmark-Laboratory.git
cd PrimeKG-Pathfinding-Algorithm-Benchmark-Laboratory
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

Requires Python 3.9+. Core dependencies: `pandas`, `numpy`, `networkx`, `scikit-learn`, `scipy`, `matplotlib`.

### 3. Download PrimeKG data

The raw PrimeKG knowledge graph files (`nodes.csv` and `edges.csv`) are too large to store in the repository. Download them manually:

1. Go to the **Harvard Dataverse PrimeKG page**:  
   [https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/IXA7BM](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/IXA7BM)

2. Download these files:
   - `nodes.csv` (~5 MB)
   - `edges.csv` (~60 MB)

3. Place them in `data/raw/`:

```
data/
└── raw/
    ├── nodes.csv      ← Download from Harvard Dataverse
    └── edges.csv      ← Download from Harvard Dataverse
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

The curated ground truth pathways are included in the repository — no extra download needed:

- `data/processed/benchmark_pathways_nodes.csv` — 343 pathways (1,456 nodes total), 150 with ≥ 4 nodes
- `data/processed/benchmark_pathways_edges.csv` — 1,113 annotated pathway edges

These were extracted from [DrugMechDB](https://drugmechdb.github.io/) and mapped onto PrimeKG node indices using the pipeline in `Ground_Truth_automation/`.

### 5. Run the benchmark

Open and run the notebook top-to-bottom:

```bash
jupyter notebook algos_/primekg_benchmark_.ipynb
```

The notebook will:
1. Load PrimeKG and ground truth data
2. Filter to pathways with ≥ 4 nodes (150 pathways)
3. Build the knowledge graph (~129K nodes, ~8M edges)
4. Run all 5 algorithms on all 150 pathways
5. Evaluate with 9 metrics per algorithm per pathway
6. Generate comparison figures and export CSVs

> **Expected runtime:** ~90–120 minutes total. Learned A* dominates at ~38s/pathway (~95 min). The other 4 algorithms finish in under 10 minutes combined.

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

## Ground Truth Construction

Ground truth pathways were sourced from [DrugMechDB](https://drugmechdb.github.io/), a curated database of drug mechanism-of-action pathways. Each pathway was:

1. Parsed from DrugMechDB's YAML format
2. Mapped onto PrimeKG using identifier lookups (DrugBank → PrimeKG drug index, UniProt → PrimeKG protein index, MESH/MONDO → PrimeKG disease index)
3. Validated to ensure all nodes and edges exist in PrimeKG
4. Filtered to retain only fully-mapped pathways

The extraction and mapping pipeline is in `Ground_Truth_automation/`.

## Known Limitations

- **Shortest Path is degenerate** — uses `indication` shortcut edges on 100% of pathways, producing 2-node direct paths rather than mechanistic routes
- **Performance degrades on long pathways** — F1 drops from ~0.65 (4 nodes) to ~0.30 (10+ nodes); no algorithm reliably reconstructs long mechanistic chains
- **Side-effect routing** — weighted algorithms frequently route through phenotype/side-effect edges rather than the true molecular mechanism
- **Bidirectional edges** — the graph treats all edges as bidirectional with identical relation labels, erasing biological directionality
- **Learned A\* data leakage** — currently trains and evaluates on the same pathways without cross-validation

## Citation

If you use this benchmark, please cite:

- **PrimeKG:** Chandak, P., Huang, K., & Zitnik, M. (2023). Building a knowledge graph to enable precision medicine. *Scientific Data*, 10(1), 67.
- **DrugMechDB:** Mayers, M., et al. (2022). DrugMechDB: A Curated Database of Drug Mechanisms of Action. *Scientific Data*, 9(1), 648.

## License

This project uses PrimeKG data licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Ground truth pathways are derived from [DrugMechDB](https://drugmechdb.github.io/).
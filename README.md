# PrimeKG Pathfinding Algorithm Benchmark Laboratory

A benchmarking framework for evaluating graph pathfinding algorithms on the task of **drug mechanism-of-action (MoA) discovery** using the **PrimeKG biomedical knowledge graph**.

**Research question:**  
Given a **drug** and a **disease**, can a graph algorithm recover the **mechanistic pathway** — the chain of proteins, biological processes, and anatomical structures that explains how the drug treats the disease?

---

# Overview

We evaluate **8 pathfinding algorithms** across **2 phases** against **150 curated ground truth pathways (4–11 nodes each)** extracted from **DrugMechDB** and mapped onto **PrimeKG**.

---

# Algorithms

## Phase 1 — Edge Weighting

| # | Algorithm | Strategy | Avg Time / Pathway |
|---|---|---|---|
| 1 | Dijkstra | Unweighted baseline | 0.03 ms |
| 2 | Hub-Penalized | Penalizes high-degree hub nodes (w = 1 + α·log(degree)) | 2,049 ms |
| 3 | PageRank-Inverse | Prefers low-centrality nodes (w = 1/(1 + PageRank)) | 4,929 ms |
| 4 | Learned A* | Spectral embeddings + MLP-learned edge weights + A* search | 38,072 ms |
| 5 | Semantic Bridging | TF-IDF cosine similarity edge weighting | 3,394 ms |

---

## Phase 2 — Search Strategy

| # | Algorithm | Strategy |
|---|---|---|
| 6 | Bidirectional ★ | Forward + backward Dijkstra; meets in the middle |
| 7 | K-Shortest + Bio | k = 4 candidate paths, re-ranked by biological plausibility |
| 8 | Bidir + Relation Weighting | Bidirectional search with enrichment-informed edge weights |

---

# Key Results (150 pathways, ≥ 4 nodes)

## Phase 1 — Edge Weighting

| Metric | Dijkstra | Hub-Penalized | PageRank-Inverse | Learned A* | Semantic Bridging |
|---|---|---|---|---|---|
| F1 Score ↑ | 0.545 | 0.540 | 0.547 | 0.460 | **0.557** |
| Edit Distance ↓ | 0.615 | 0.551 | 0.544 | 0.617 | **0.537** |

**Phase 1 F1 spread = 0.023.**  
Edge weighting alone does **not significantly differentiate algorithms** — **graph topology dominates**.

---

## Phase 2 — Search Strategy

| Metric | Bidirectional ★ | K-Shortest + Bio | Bidir + Rel. Wt |
|---|---|---|---|
| F1 Score ↑ | **0.571** | 0.555 | 0.549 |
| Edit Distance ↓ | **0.429** | 0.445 | 0.451 |

**Bidirectional search produces the first statistically significant improvement.**

Edit distance gain vs. Dijkstra:  
```
−0.186 (p < 0.000001, Cohen's d = −1.334)
```

Full results:

```
algos_/evaluation_results_all_algorithms.csv
```

---

# Repository Structure

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
│   ├── raw/                         ← PrimeKG source files (NOT in repo — download below)
│   │   ├── nodes.csv
│   │   ├── edges.csv
│   │   └── ...
│   └── processed/                   ← Ground truth pathways (included in repo)
│       ├── benchmark_pathways_nodes.csv
│       ├── benchmark_pathways_edges.csv
│       └── ...
│
├── notebook/                        ← Legacy multi-file version (reference only)
│   ├── algorithm_benchmark.ipynb
│   ├── Algorithms.py
│   ├── evaluation_helpers.py
│   └── evaluation_metrics.py
│
├── Ground_Truth_automation/         ← Pipeline for extracting pathways
│   ├── automated_pipeline.py
│   ├── pathway_validator.py
│   └── ...
│
├── src/
│   └── prepare_primekg.py
│
├── requirements.txt
└── README.md
```

**Note on `notebook/`:**  
An earlier multi-file implementation where algorithms, metrics, and helpers were separated.  
The primary notebook is now:

```
algos_/primekg_benchmark_.ipynb
```

which is **fully self-contained**.

---

# Setup

## 1. Clone the repository

```bash
git clone https://github.com/maxxyhc/PrimeKG-Pathfinding-Algorithm-Benchmark-Laboratory.git
cd PrimeKG-Pathfinding-Algorithm-Benchmark-Laboratory
```

---

## 2. Install dependencies

```bash
pip install -r requirements.txt
```

Requires **Python 3.9+**

Core dependencies:

```
pandas
numpy
networkx
scikit-learn
scipy
matplotlib
```

---

## 3. Download PrimeKG data

The raw PrimeKG files are **too large to store in the repository**.

Download from **Harvard Dataverse**:

→ PrimeKG Dataset

Download:

```
nodes.csv (~5 MB)
edges.csv (~60 MB)
```

Place them in:

```
data/raw/
```

Directory structure:

```
data/
└── raw/
    ├── nodes.csv
    └── edges.csv
```

---

### Expected Formats

**nodes.csv**

```csv
node_index,node_id,node_type,node_name,node_source
0,9796,gene/protein,PHYHIP,NCBI
1,7918,gene/protein,GPANK1,NCBI
```

**edges.csv**

```csv
relation,display_relation,x_index,y_index
protein_protein,ppi,0,8889
protein_protein,ppi,1,2798
```

---

## 4. Ground truth pathways (included)

No additional download required.

| File | Description |
|---|---|
| `data/processed/benchmark_pathways_nodes.csv` | 343 pathways (1,456 nodes total); **150 with ≥ 4 nodes used for benchmarking** |
| `data/processed/benchmark_pathways_edges.csv` | 1,113 annotated pathway edges |

These were extracted from **DrugMechDB** and mapped onto **PrimeKG node indices**.

Pipeline located in:

```
Ground_Truth_automation/
```

---

## 5. Run the benchmark

```bash
jupyter notebook algos_/primekg_benchmark_.ipynb
```

The notebook will:

1. Load PrimeKG and ground truth data  
2. Filter to pathways with ≥ 4 nodes (150 pathways)  
3. Build the knowledge graph (~129K nodes, ~8M edges)  
4. Run all **8 algorithms**  
5. Evaluate with **9 metrics per algorithm per pathway**  
6. Generate figures and export CSV results

---

### Expected Runtime

```
~90–120 minutes total
```

Learned A* dominates runtime:

```
~38 seconds / pathway
~95 minutes total
```

All other algorithms combined:

```
< 15 minutes
```

---

# Evaluation Metrics

| Category | Metric | Description |
|---|---|---|
| Node Accuracy | Precision | Fraction of predicted nodes in ground truth |
|  | Recall | Fraction of ground truth nodes recovered |
|  | F1 Score | Harmonic mean of precision and recall |
| Target Finding | Hits@1, @3, @5 | Whether disease appears in last k predicted nodes |
| Mechanistic Quality | Relation Accuracy | Fraction of predicted edge types in ground truth |
|  | Edit Distance | Normalized Levenshtein distance between node sequences |
|  | Hub Node Ratio | Fraction of path nodes that are high-degree hubs |
| Efficiency | Path Length MAE | Absolute difference between predicted and true path length |
|  | Speed | Wall-clock time per pathway |

---

# Key Findings

### Edge weighting alone doesn't matter
Phase 1 F1 spread = **0.023** across 5 algorithms.

**0/10 pairwise comparisons are statistically significant.**

Graph topology dominates over weighting strategy.

---

### Bidirectional search is the first real improvement

Searching from both drug and disease simultaneously:

```
Edit distance improvement vs Dijkstra = −0.186
p < 0.000001
```

Adding biological signals afterward **actually worsened results**.

---

### The core problem is path length

```
Average predicted path = 3.4 nodes
Average ground truth = 5.8 nodes
```

68% of bidirectional paths are exactly **3 nodes**.

The task effectively reduces to **choosing one intermediate node**.

---

### Hub shortcuts dominate failures

43% of failures route through **high-degree hubs**  
(e.g., seizure node with **4,218 connections**).

These nodes are structurally close to everything but **biologically irrelevant**.

---

### The first edge determines everything

| First Edge Type | F1 |
|---|---|
| drug → protein | **0.604** |
| drug → effect | **0.437** |

Gap:

```
+0.167 (p = 0.000004)
```

Algorithms often choose **side-effect shortcuts** because they are shorter.

---

### Performance collapses on long pathways

```
4-node pathways → F1 ≈ 0.66
10-node pathways → F1 ≈ 0.26
```

No algorithm reliably reconstructs **long mechanistic chains**.

---

# Ground Truth Construction

Ground truth pathways were sourced from **DrugMechDB**, a curated database of drug mechanism-of-action pathways.

Pipeline steps:

1. Parse DrugMechDB YAML pathways  
2. Map identifiers to PrimeKG nodes  
   - DrugBank → PrimeKG drug index  
   - UniProt → PrimeKG protein index  
   - MESH / MONDO → PrimeKG disease index  
3. Validate node/edge existence in PrimeKG  
4. Retain only fully mapped pathways

Pipeline location:

```
Ground_Truth_automation/
```

---

# Known Limitations

**Shortest path degeneracy**

Shortest path uses **indication shortcut edges** on 100% of pathways, producing trivial **2-node solutions**.

---

**Performance degrades on long pathways**

```
F1 ≈ 0.65 (4 nodes)
F1 ≈ 0.30 (10+ nodes)
```

---

**Side-effect routing**

Weighted algorithms frequently route through **phenotype/side-effect edges** rather than the true molecular mechanism.

---

**Bidirectional edges**

Graph currently treats edges as **bidirectional**, losing biological directionality.

---

**Learned A\* data leakage**

Model trains and evaluates on the **same pathways** without cross-validation.

---

# Citation

If you use this benchmark, please cite:

**PrimeKG**

```
Chandak, P., Huang, K., & Zitnik, M. (2023).
Building a knowledge graph to enable precision medicine.
Scientific Data, 10(1), 67.
```

**DrugMechDB**

```
Mayers, M., et al. (2022).
DrugMechDB: A Curated Database of Drug Mechanisms of Action.
Scientific Data, 9(1), 648.
```

---

# License

This project uses **PrimeKG data licensed under CC BY 4.0**.  
Ground truth pathways are derived from **DrugMechDB**.
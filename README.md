PrimeKG Pathfinding Algorithm Benchmark Laboratory
A benchmarking framework for evaluating graph pathfinding algorithms on the task of drug mechanism-of-action (MoA) discovery using the PrimeKG biomedical knowledge graph.
Given a drug and a disease, can a graph algorithm recover the mechanistic pathway — the chain of proteins, biological processes, and anatomical structures that explains how the drug treats the disease?
Overview
We evaluate 8 pathfinding algorithms across 2 phases against 150 curated ground truth pathways (4–11 nodes each) extracted from DrugMechDB and mapped onto PrimeKG.
Algorithms
Phase 1 — Edge Weighting
#AlgorithmStrategyAvg Time/Pathway1DijkstraUnweighted baseline0.03 ms2Hub-PenalizedPenalizes high-degree hub nodes (w = 1 + α·log(degree))2,049 ms3PageRank-InversePrefers low-centrality nodes (w = 1/(1 + PageRank))4,929 ms4Learned A*Spectral embeddings + MLP-learned edge weights + A* search38,072 ms5Semantic BridgingTF-IDF cosine similarity edge weighting3,394 ms
Phase 2 — Search Strategy
#AlgorithmStrategy6Bidirectional ★Forward + backward Dijkstra; meets in the middle7K-Shortest + Biok=4 candidate paths, re-ranked by biological plausibility8Bidir + Relation WeightingBidirectional search with enrichment-informed edge weights
Key Results (150 pathways, ≥ 4 nodes)
Phase 1 — Edge Weighting
MetricDijkstraHub-PenalizedPageRank-InverseLearned A*Semantic BridgingF1 Score ↑0.5450.5400.5470.4600.557Edit Distance ↓0.6150.5510.5440.6170.537

Phase 1 F1 spread = 0.023. Edge weighting alone does not significantly differentiate algorithms — graph topology dominates.

Phase 2 — Search Strategy
MetricBidirectional ★K-Shortest + BioBidir + Rel. WtF1 Score ↑0.5710.5550.549Edit Distance ↓0.4290.4450.451

Bidirectional search produces the first statistically significant improvement. Edit distance gain vs. Dijkstra: −0.186 (p < 0.000001, Cohen's d = −1.334).

Full results: algos_/evaluation_results_all_algorithms.csv

Repository Structure
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
│       ├── benchmark_pathways_nodes.csv   ← 150 curated pathways
│       ├── benchmark_pathways_edges.csv   ← Pathway edge annotations
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

Note on notebook/: An earlier multi-file version where algorithms, metrics, and helpers lived in separate .py files. Kept for reference only. The primary notebook is algos_/primekg_benchmark_.ipynb, which is fully self-contained.


Setup
1. Clone the repository
bashgit clone https://github.com/maxxyhc/PrimeKG-Pathfinding-Algorithm-Benchmark-Laboratory.git
cd PrimeKG-Pathfinding-Algorithm-Benchmark-Laboratory
2. Install dependencies
bashpip install -r requirements.txt
Requires Python 3.9+. Core dependencies: pandas, numpy, networkx, scikit-learn, scipy, matplotlib.
3. Download PrimeKG data
The raw PrimeKG files are too large to store in the repository. Download them directly from Harvard Dataverse:
→ Harvard Dataverse: PrimeKG
Download these two files:

nodes.csv (~5 MB)
edges.csv (~60 MB)

Place them in data/raw/:
data/
└── raw/
    ├── nodes.csv      ← from Harvard Dataverse
    └── edges.csv      ← from Harvard Dataverse
Expected formats:
nodes.csv:
csvnode_index,node_id,node_type,node_name,node_source
0,9796,gene/protein,PHYHIP,NCBI
1,7918,gene/protein,GPANK1,NCBI
edges.csv:
csvrelation,display_relation,x_index,y_index
protein_protein,ppi,0,8889
protein_protein,ppi,1,2798
4. Ground truth pathways (already in the repo)
The curated ground truth pathways are included — no extra download needed:
FileDescriptiondata/processed/benchmark_pathways_nodes.csv343 pathways (1,456 nodes total); 150 with ≥ 4 nodes used for benchmarkingdata/processed/benchmark_pathways_edges.csv1,113 annotated pathway edges
These were extracted from DrugMechDB and mapped onto PrimeKG node indices using the pipeline in Ground_Truth_automation/.
5. Run the benchmark
bashjupyter notebook algos_/primekg_benchmark_.ipynb
The notebook will:

Load PrimeKG and ground truth data
Filter to pathways with ≥ 4 nodes (150 pathways)
Build the knowledge graph (~129K nodes, ~8M edges)
Run all 8 algorithms on all 150 pathways
Evaluate with 9 metrics per algorithm per pathway
Generate comparison figures and export CSVs


Expected runtime: ~90–120 minutes total. Learned A* dominates at ~38s/pathway (~95 min). All other algorithms finish in under 15 minutes combined.


Evaluation Metrics
CategoryMetricDescriptionNode AccuracyPrecisionFraction of predicted nodes that are in ground truthRecallFraction of ground truth nodes recoveredF1 ScoreHarmonic mean of precision and recallTarget FindingHits@1, @3, @5Whether the disease appears in the last k predicted nodesMechanistic QualityRelation AccuracyFraction of predicted edge types found in ground truthEdit DistanceNormalized Levenshtein distance between node sequences (0 = perfect)Hub Node RatioFraction of path nodes that are high-degree hubs (lower is better)EfficiencyPath Length MAEAbsolute difference between predicted and ground truth path lengthSpeedWall-clock time per pathway in milliseconds

Key Findings

Edge weighting alone doesn't matter. Phase 1 F1 spread = 0.023 across 5 algorithms; 0/10 pairwise comparisons are statistically significant. Graph topology dominates over weighting strategy.
Bidirectional search is the first real improvement. Searching from both drug and disease simultaneously cuts edit distance by 0.186 vs. Dijkstra (p < 0.000001). Adding biological signals on top made results worse — the search strategy already implicitly captures mechanistic grammar.
The core problem is path length. Average predicted path = 3.4 nodes; average ground truth = 5.8 nodes. 68% of bidirectional paths are exactly 3 nodes. The task effectively reduces to picking one intermediate.
Hub shortcuts are the dominant failure mode. 43% of failures involve routing through high-degree hub nodes (e.g., Seizure: 4,218 connections) that are structurally close to everything but mechanistically irrelevant.
The first edge determines everything. Paths starting with drug_protein achieve F1 = 0.604; paths starting with drug_effect achieve F1 = 0.437 (gap: +0.167, p = 0.000004). All 30 drugs that route through side effects have protein edges available — the algorithm chooses the side-effect route because it's shorter. Penalizing shortcuts just reveals the next shortcut layer.
Performance degrades sharply with pathway length. F1 drops from ~0.66 (4-node pathways) to ~0.26 (10-node pathways). No algorithm reliably reconstructs long mechanistic chains.


Ground Truth Construction
Ground truth pathways were sourced from DrugMechDB, a curated database of drug mechanism-of-action pathways. Each pathway was:

Parsed from DrugMechDB's YAML format
Mapped onto PrimeKG using identifier lookups (DrugBank → PrimeKG drug index, UniProt → PrimeKG protein index, MESH/MONDO → PrimeKG disease index)
Validated to ensure all nodes and edges exist in PrimeKG
Filtered to retain only fully-mapped pathways

The extraction and mapping pipeline is in Ground_Truth_automation/.

Known Limitations

Shortest path is degenerate — uses indication shortcut edges on 100% of pathways, producing 2-node direct paths rather than mechanistic routes
Performance degrades on long pathways — F1 drops from ~0.65 (4 nodes) to ~0.30 (10+ nodes); no algorithm reliably reconstructs long mechanistic chains
Side-effect routing — weighted algorithms frequently route through phenotype/side-effect edges rather than the true molecular mechanism
Bidirectional edges — the graph treats all edges as bidirectional with identical relation labels, erasing biological directionality
Learned A* data leakage — currently trains and evaluates on the same pathways without cross-validation


Citation
If you use this benchmark, please cite:

PrimeKG: Chandak, P., Huang, K., & Zitnik, M. (2023). Building a knowledge graph to enable precision medicine. Scientific Data, 10(1), 67.
DrugMechDB: Mayers, M., et al. (2022). DrugMechDB: A Curated Database of Drug Mechanisms of Action. Scientific Data, 9(1), 648.

License
This project uses PrimeKG data licensed under CC BY 4.0. Ground truth pathways are derived from DrugMechDB.
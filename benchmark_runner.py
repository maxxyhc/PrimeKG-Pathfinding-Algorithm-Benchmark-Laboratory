## ============================================
# IMPORTS
# ============================================
# Standard libraries
import os
import sys
import time
import logging
from datetime import datetime
from pathlib import Path

# Data handling
import pandas as pd
import numpy as np
import networkx as nx

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

# Project modules 
from src.evaluation_metrics import *
from src.evaluation_helpers import *
from src.Algorithms import *

# ============================================
# CONFIGURATION
# ============================================
PATHS = {
    'nodes': 'data/nodes.csv', 
    'edges': 'data/edges.csv',  
    'ground_truth_nodes': 'data/benchmark_pathways_nodes.csv',  
    'ground_truth_edges': 'data/benchmark_pathways_edges.csv'   
}
#test mode
#TEST_PATHWAYS = 10 

# ============================================
# LOGGING
# ============================================

# Create logs directory if it doesn't exist
os.makedirs('logs', exist_ok=True)

# Setup logging
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler(f'logs/benchmark_{timestamp}.log'),
        logging.StreamHandler()
    ]
)

logging.info("="*60)
logging.info("BENCHMARK PIPELINE STARTED")
logging.info("="*60)

# ============================================
# STEP 1: LOAD DATA
# ============================================

# Load PrimeKG data
logging.info("Loading PrimeKG data...")
nodes = pd.read_csv(PATHS['nodes'], encoding="latin1")
edges = pd.read_csv(PATHS['edges'], encoding="latin1")

logging.info(f"  ✓ Loaded {len(nodes):,} nodes")
logging.info(f"  ✓ Loaded {len(edges):,} edges")
logging.info(f"  ✓ Node types: {nodes['node_type'].nunique()}")
logging.info(f"  ✓ Edge types: {edges['relation'].nunique()}")

# Load ground truth
logging.info("Loading ground truth pathways...")
ground_truth_nodes = pd.read_csv(PATHS['ground_truth_nodes'], dtype={'node_index': int})
ground_truth_edges = pd.read_csv(PATHS['ground_truth_edges'])

pathways = ground_truth_nodes['pathway_id'].unique()
logging.info(f"  ✓ Loaded {len(pathways)} pathways")

for p in pathways:
    n_nodes = len(ground_truth_nodes[ground_truth_nodes['pathway_id'] == p])
    logging.info(f"    - {p}: {n_nodes} nodes")

# Filter to pathways with 4+ nodes
logging.info("Filtering pathways...")
node_counts = ground_truth_nodes.groupby('pathway_id')['node_index'].count()
long_pathways = node_counts[node_counts >= 4].index
ground_truth_nodes = ground_truth_nodes[ground_truth_nodes['pathway_id'].isin(long_pathways)]
ground_truth_edges = ground_truth_edges[ground_truth_edges['pathway_id'].isin(long_pathways)]
logging.info(f"  ✓ Filtered to {len(long_pathways)} pathways with 4+ nodes")
logging.info(f"  ✓ Nodes: {len(ground_truth_nodes):,}")
logging.info(f"  ✓ Edges: {len(ground_truth_edges):,}")

# # TEST MODE: Limit to subset if TEST_PATHWAYS is set
# if TEST_PATHWAYS is not None:
#     logging.info(f"\n⚠️  TEST MODE: Limiting to {TEST_PATHWAYS} pathways")
#     test_pathway_ids = long_pathways[:TEST_PATHWAYS]
#     ground_truth_nodes = ground_truth_nodes[ground_truth_nodes['pathway_id'].isin(test_pathway_ids)]
#     ground_truth_edges = ground_truth_edges[ground_truth_edges['pathway_id'].isin(test_pathway_ids)]
#     logging.info(f"  ✓ Test subset: {len(test_pathway_ids)} pathways")

# ============================================
# STEP 2: BUILD GRAPH
# ============================================
def build_graph(nodes_df, edges_df, bidirectional=True):
    """
    Build a NetworkX graph from cleaned PrimeKG CSVs.
    
    nodes_df columns:
        node_index, node_id, node_type, node_name, node_source
    edges_df columns:
        relation, display_relation, x_index, y_index
    """
    G = nx.DiGraph()
    
    # ---------- Add nodes ----------
    for _, row in nodes_df.iterrows():
        G.add_node(
            int(row['node_index']),
            node_id=str(row['node_id']),
            node_name=str(row['node_name']),
            node_type=str(row['node_type']),
            node_source=str(row['node_source'])
        )
    
    # ---------- Add edges ----------
    for _, row in edges_df.iterrows():
        G.add_edge(
            int(row['x_index']),
            int(row['y_index']),
            relation=str(row['relation']),
            display_relation=str(row['display_relation'])
        )
        if bidirectional:
            G.add_edge(
                int(row['y_index']),
                int(row['x_index']),
                relation=str(row['relation']),
                display_relation=str(row['display_relation'])
            )
    
    return G

logging.info("Building graph (may take a couple minutes)...")
G = build_graph(nodes, edges, bidirectional=True)
logging.info(f"  ✓ Graph built: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")

# ============================================
# STEP 3: DEFINE & INITIALIZE ALGORITHMS
# ============================================
logging.info("="*60)
logging.info("STEP 3: INITIALIZING ALGORITHMS")
logging.info("="*60)

algorithms = {}

# Algorithm 1: Dijkstra (baseline)
logging.info("Initializing Dijkstra...")
algorithms['Dijkstra'] = DijkstraShortestPath(G)
logging.info("  ✓ Initialized")

# Algorithm 2: MetaPathBFS
logging.info("Initializing MetaPathBFS...")
algorithms['MetaPathBFS'] = MetaPathBFS(G, max_length=10)
logging.info("  ✓ Initialized with default meta-path patterns")

# Algorithm 3: HubPenalizedShortestPath
logging.info("Initializing HubPenalizedShortestPath...")
algorithms['HubPenalized'] = HubPenalizedShortestPath(G, alpha=0.5)
logging.info("  ✓ Edge weights computed (α=0.5)")

# Algorithm 4: PageRankInverseShortestPath
logging.info("Initializing PageRankInverseShortestPath...")
algorithms['PageRankInverse'] = PageRankInverseShortestPath(G, damping=0.85)
logging.info("  ✓ PageRank scores computed")

# Algorithm 5: SemanticBridgingPath
logging.info("Initializing SemanticBridgingPath...")
algorithms['SemanticBridging'] = SemanticBridgingPath(G, beta=0.3)
logging.info("  ✓ Initialized (embeddings computed on first use)")

# Algorithm 6: BidirectionalSearch
logging.info("Initializing BidirectionalSearch...")
algorithms['Bidirectional'] = BidirectionalSearch(G, max_depth=8, max_explore=50000)
logging.info("  ✓ Initialized")

# Algorithm 7: BidirectionalKShortestBio
logging.info("Initializing BidirectionalKShortestBio...")
algorithms['KShortestBio'] = BidirectionalKShortestBio(G, k=4, max_depth=8, max_explore=50000)
logging.info("  ✓ Initialized with k=10 paths")

# Algorithm 8: BidirectionalRelationWeighted
logging.info("Initializing BidirectionalRelationWeighted...")
algorithms['BidirRelationWeighted'] = BidirectionalRelationWeighted(G, max_depth=8, max_explore=50000)
logging.info("  ✓ Relation weights configured")

logging.info(f"\n  ✓ Initialized {len(algorithms)} algorithms")

# ============================================
# STEP 4: RUN ALGORITHMS ON ALL PATHWAYS
# ============================================
logging.info("="*60)
logging.info("STEP 4: RUNNING ALGORITHMS ON PATHWAYS")
logging.info("="*60)

def run_single_pathway(algorithm, graph, source, target, pathway_id, ground_truth_length):
    """
    Run one algorithm on one pathway.
    
    Returns dict with prediction details or NONE if failed.
    """
    try:
        # Call algorithm
        path_nodes, relations, cost = algorithm.find_path(source, target)
        
        # Check if valid path found
        if path_nodes and len(path_nodes) > 0 and path_nodes[-1] == target:
            # Extract node details from graph
            predicted_node_ids = [graph.nodes[idx]['node_id'] for idx in path_nodes]
            predicted_node_names = [graph.nodes[idx]['node_name'] for idx in path_nodes]
            
            return {
                'pathway_id': pathway_id,
                'predicted_node_indices': ','.join(map(str, path_nodes)),
                'predicted_node_ids': ','.join(predicted_node_ids),
                'predicted_node_names': ','.join(predicted_node_names),
                'predicted_relations': ','.join(relations),
                'predicted_length': len(path_nodes),
                'ground_truth_length': ground_truth_length
            }
        else:
            # No valid path found
            return {
                'pathway_id': pathway_id,
                'predicted_node_indices': 'NONE',
                'predicted_node_ids': 'NONE',
                'predicted_node_names': 'NONE',
                'predicted_relations': 'NONE',
                'predicted_length': 0,
                'ground_truth_length': ground_truth_length
            }
    
    except Exception as e:
        # Error during path finding
        logging.debug(f"    Error on {pathway_id}: {str(e)}")
        return {
            'pathway_id': pathway_id,
            'predicted_node_indices': 'NONE',
            'predicted_node_ids': 'NONE',
            'predicted_node_names': 'NONE',
            'predicted_relations': 'NONE',
            'predicted_length': 0,
            'ground_truth_length': ground_truth_length
        }


def run_all_pathways(algorithms_dict, graph, ground_truth_nodes):
    """
    Run all algorithms on all pathways.
    
    Returns dict mapping algorithm_name -> predictions_df
    """
    # Get unique pathways with their endpoints
    pathway_endpoints = ground_truth_nodes.groupby('pathway_id').agg(
        source_index=('node_index', 'first'),
        target_index=('node_index', 'last')
    ).reset_index()
    
    # Get ground truth lengths
    pathway_lengths = ground_truth_nodes.groupby('pathway_id').size().to_dict()
    
    total_pathways = len(pathway_endpoints)
    predictions_dict = {}
    
    # Run each algorithm
    for algo_name, algorithm in algorithms_dict.items():
        logging.info(f"\nRunning {algo_name}...")
        
        predictions = []
        successes = 0
        failures = 0
        start_time = time.time()
        
        for idx, row in pathway_endpoints.iterrows():
            pathway_id = row['pathway_id']
            source = row['source_index']
            target = row['target_index']
            gt_length = pathway_lengths.get(pathway_id, 0)
            
            # Run algorithm on this pathway
            result = run_single_pathway(algorithm, graph, source, target, pathway_id, gt_length)
            predictions.append(result)
            
            # Track success/failure
            if result['predicted_node_indices'] != 'NONE':
                successes += 1
            else:
                failures += 1
            
            # Progress logging every 10 pathways
            if (idx + 1) % 10 == 0:
                elapsed = time.time() - start_time
                rate = (idx + 1) / elapsed if elapsed > 0 else 0
                logging.info(f"  [{idx + 1}/{total_pathways}] {successes} found, {failures} failed | {rate:.1f} pathways/sec")
        
        # Convert to DataFrame
        predictions_df = pd.DataFrame(predictions)
        predictions_dict[algo_name] = predictions_df
        
        # Final stats
        elapsed = time.time() - start_time
        rate = total_pathways / elapsed if elapsed > 0 else 0
        logging.info(f"  ✓ {algo_name} complete:")
        logging.info(f"    - Paths found: {successes}/{total_pathways} ({100*successes/total_pathways:.1f}%)")
        logging.info(f"    - Failed: {failures}/{total_pathways}")
        logging.info(f"    - Time: {elapsed:.2f}s ({rate:.1f} pathways/sec)")
    
    return predictions_dict


# Run all algorithms on all pathways
predictions_dict = run_all_pathways(algorithms, G, ground_truth_nodes)

logging.info("\n" + "="*60)
logging.info("All algorithms complete!")
logging.info("="*60)



# ============================================
# STEP 5: EVALUATE ALL ALGORITHMS
# ============================================
logging.info("="*60)
logging.info("STEP 5: EVALUATING PREDICTIONS")
logging.info("="*60)

def evaluate_algorithm(algorithm_name, predictions, ground_truth_nodes, degree_count, hub_threshold):
    """
    Evaluate all predictions for a single algorithm.
    
    Returns DataFrame with evaluation results for each pathway.
    """
    results = []
    
    for _, row in predictions.iterrows():
        pathway_id = row['pathway_id']
        
        # Get ground truth for this pathway
        gt_nodes = ground_truth_nodes[ground_truth_nodes['pathway_id'] == pathway_id]
        
        if len(gt_nodes) == 0:
            logging.warning(f"  No ground truth found for {pathway_id}")
            continue
        
        # Parse predicted values
        if row['predicted_node_ids'] == 'NONE':
            predicted_ids = []
            predicted_indices = []
        else:
            predicted_ids = row['predicted_node_ids'].split(',')
            predicted_indices = [int(x) for x in row['predicted_node_indices'].split(',')]
        
        predicted_length = row['predicted_length']
        
        # Get ground truth values
        gt_nodes = gt_nodes.sort_values('step_order')
        gt_ids = gt_nodes['node_id'].astype(str).tolist()
        gt_length = len(gt_nodes)
        
        # Call evaluate_pathway to get all 7 metrics
        metrics = evaluate_pathway(
        predicted_ids=predicted_ids,
        predicted_indices=predicted_indices,
        predicted_length=predicted_length,
        ground_truth_ids=gt_ids,
        ground_truth_length=gt_length,
        degree_count=degree_count,
        hub_threshold=hub_threshold,
        algorithm_name=algorithm_name  
    )
        
        # Add metadata
        metrics['pathway_id'] = pathway_id
        metrics['algorithm'] = algorithm_name
        metrics['predicted_length'] = predicted_length
        metrics['ground_truth_length'] = gt_length
        
        results.append(metrics)
    
    return pd.DataFrame(results)


def evaluate_all_algorithms(predictions_dict, ground_truth_nodes, degree_count, hub_threshold):
    """
    Evaluate all algorithms and combine results.
    
    Returns combined DataFrame with all evaluation results.
    """
    all_results = []
    
    for algo_name, predictions in predictions_dict.items():
        logging.info(f"  Evaluating {algo_name}...")
        algo_results = evaluate_algorithm(
            algo_name, predictions,
            ground_truth_nodes,
            degree_count, hub_threshold
        )
        all_results.append(algo_results)
        logging.info(f"    ✓ {len(algo_results)} pathways evaluated")
    
    combined = pd.concat(all_results, ignore_index=True)
    
    # Reorder columns
    col_order = [
        'pathway_id', 'algorithm',
        'precision', 'recall', 'f1_score',
        'edit_distance', 'mrr',
        'hub_node_ratio', 'path_length_accuracy',
        'predicted_length', 'ground_truth_length'
    ]
    combined = combined[[c for c in col_order if c in combined.columns]]
    
    return combined


def generate_summary(results):
    """
    Generate summary statistics (mean) for each algorithm.
    
    Returns summary DataFrame with mean metrics per algorithm.
    """
    metrics = [
        'precision', 'recall', 'f1_score',
        'edit_distance', 'mrr',
        'hub_node_ratio', 'path_length_accuracy'
    ]
    
    # Filter to only metrics that exist in results
    metrics = [m for m in metrics if m in results.columns]
    
    # Calculate mean only
    summary = results.groupby('algorithm')[metrics].mean()
    
    # Round for display
    summary = summary.round(4)
    
    # Add count
    summary['n_pathways'] = results.groupby('algorithm').size()
    
    return summary


def log_comparison_table(summary):
    """Log formatted comparison table."""
    logging.info("\n" + "="*80)
    logging.info("ALGORITHM COMPARISON SUMMARY")
    logging.info("="*80)
    
    logging.info("\n" + summary.to_string())
    logging.info("="*80)
    
    # Find best algorithm for each metric
    logging.info("\nBest Algorithm per Metric:")
    logging.info("-"*40)
    for col in summary.columns:
        if col == 'n_pathways':
            continue
        # For hub_node_ratio and edit_distance, lower is better
        if col in ['hub_node_ratio', 'edit_distance']:
            best = summary[col].idxmin()
            logging.info(f"  {col:<25}: {best} ({summary.loc[best, col]:.4f}) [lower is better]")
        else:
            best = summary[col].idxmax()
            logging.info(f"  {col:<25}: {best} ({summary.loc[best, col]:.4f})")


# Compute degree counts
logging.info("Computing node degrees...")
degree_count = compute_degree_counts(edges)
logging.info(f"  ✓ Computed degrees for {len(degree_count):,} nodes")

# Compute hub threshold
logging.info("Computing hub threshold...")
hub_threshold = compute_hub_threshold(degree_count, percentile=95)
logging.info(f"  ✓ Hub threshold (95th percentile): {hub_threshold:.0f}")

# Evaluate all algorithms
logging.info("Evaluating algorithms...")
detailed_results = evaluate_all_algorithms(
    predictions_dict,
    ground_truth_nodes,
    degree_count,
    hub_threshold
)

# Generate summary
logging.info("Generating summary...")
summary = generate_summary(detailed_results)
logging.info(f"  ✓ Summary generated for {len(summary)} algorithms")

# Log comparison table
log_comparison_table(summary)

logging.info("\n" + "="*60)
logging.info("Evaluation complete!")
logging.info("="*60)

# ============================================
# STEP 6: SAVE RESULTS
# ============================================
logging.info("="*60)
logging.info("STEP 6: SAVING RESULTS")
logging.info("="*60)

# Create results directory
os.makedirs('results', exist_ok=True)

# Combine all predictions into one DataFrame
all_predictions = pd.concat([
    df.assign(algorithm=algo_name) 
    for algo_name, df in predictions_dict.items()
], ignore_index=True)

# Reorder columns for predictions
pred_col_order = [
    'pathway_id', 'algorithm',
    'predicted_node_indices', 'predicted_node_ids', 'predicted_node_names',
    'predicted_relations', 'predicted_length', 'ground_truth_length'
]
all_predictions = all_predictions[[c for c in pred_col_order if c in all_predictions.columns]]

# Save all files
logging.info("Saving results...")
all_predictions.to_csv('results/all_predictions.csv', index=False)
logging.info(f"  ✓ Saved all_predictions.csv ({len(all_predictions)} rows)")

detailed_results.to_csv('results/all_results.csv', index=False)
logging.info(f"  ✓ Saved all_results.csv ({len(detailed_results)} rows)")

summary.to_csv('results/summary_by_algorithm.csv')
logging.info(f"  ✓ Saved summary_by_algorithm.csv ({len(summary)} algorithms)")

logging.info("\n" + "="*60)
logging.info("BENCHMARK COMPLETE!")
logging.info("="*60)
logging.info(f"Results saved to: results/")
logging.info(f"  - all_predictions.csv")
logging.info(f"  - all_results.csv") 
logging.info(f"  - summary_by_algorithm.csv")
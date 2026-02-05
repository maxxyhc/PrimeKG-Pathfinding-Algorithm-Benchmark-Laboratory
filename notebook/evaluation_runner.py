# ============================================
# EVALUATION RUNNER
# ============================================
# 
# This module runs all 6 evaluation metrics on algorithm predictions
# and generates comparison tables.
#
# Usage:
#   from evaluation_runner import evaluate_all_algorithms, generate_summary
#   
#   results = evaluate_all_algorithms(
#       predictions_dict={'Hub-Penalized': hub_predictions, 'PageRank': pr_predictions},
#       ground_truth_nodes=ground_truth_nodes,
#       ground_truth_edges=ground_truth_edges,
#       degree_count=degree_count,
#       hub_threshold=hub_threshold
#   )
#   
#   summary = generate_summary(results)
# ============================================

import pandas as pd
import numpy as np
from typing import Dict, List
from evaluation_metrics import (
    precision, 
    recall_at_k_hops, 
    f1_score, 
    path_length_accuracy,
    hub_node_ratio,
    mrr
)
from evaluation_helpers import (
    compute_degree_counts,
    compute_hub_threshold,
    calculate_edit_distance,
    calculate_relation_accuracy
)


def evaluate_single_pathway(
    predicted_row: pd.Series,
    gt_nodes_df: pd.DataFrame,
    gt_edges_df: pd.DataFrame,
    degree_count: Dict[int, int],
    hub_threshold: float
) -> Dict:
    """
    Evaluate a single pathway prediction against ground truth.
    
    Args:
        predicted_row: Row from predictions DataFrame
        gt_nodes_df: Ground truth nodes for this pathway
        gt_edges_df: Ground truth edges for this pathway
        degree_count: Dict of {node_index: degree}
        hub_threshold: Degree threshold for hub nodes
    
    Returns:
        Dict with all metric values
    """
    pathway_id = predicted_row['pathway_id']
    
    # Parse predicted values
    if predicted_row['predicted_node_ids'] == 'NONE':
        predicted_ids = []
        predicted_indices = []
        predicted_relations = []
    else:
        predicted_ids = predicted_row['predicted_node_ids'].split(',')
        predicted_indices = [int(x) for x in predicted_row['predicted_node_indices'].split(',')]
        predicted_relations = predicted_row['predicted_relations'].split(',') if predicted_row['predicted_relations'] != 'NONE' else []
    
    predicted_length = predicted_row['predicted_length']
    
    # Get ground truth values
    gt_nodes = gt_nodes_df.sort_values('step_order')
    gt_ids = gt_nodes['node_id'].astype(str).tolist()
    gt_indices = gt_nodes['node_index'].tolist()
    gt_length = len(gt_nodes)
    
    # Get ground truth edge types
    gt_edge_types = gt_edges_df['relation'].tolist() if len(gt_edges_df) > 0 else []
    
    # Calculate all metrics
    results = {
        'pathway_id': pathway_id,
        'predicted_length': predicted_length,
        'ground_truth_length': gt_length,
        
        # Node accuracy metrics
        'precision': precision(predicted_ids, gt_ids),
        'recall_at_6_hops': recall_at_k_hops(predicted_ids, gt_ids, k=6),
        'f1_score': f1_score(predicted_ids, gt_ids),
        
        # Path quality metrics
        'path_length_accuracy': path_length_accuracy(predicted_length, gt_length),
        'hub_node_ratio': hub_node_ratio(predicted_indices, degree_count, hub_threshold),
        'mrr': mrr(predicted_ids, gt_ids),
        
        # Additional metrics (optional)
        'edit_distance': calculate_edit_distance(predicted_ids, gt_ids),
        'relation_accuracy': calculate_relation_accuracy(predicted_relations, gt_edge_types)
    }
    
    return results


def evaluate_algorithm(
    algorithm_name: str,
    predictions: pd.DataFrame,
    ground_truth_nodes: pd.DataFrame,
    ground_truth_edges: pd.DataFrame,
    degree_count: Dict[int, int],
    hub_threshold: float
) -> pd.DataFrame:
    """
    Evaluate all predictions for a single algorithm.
    
    Args:
        algorithm_name: Name of the algorithm
        predictions: DataFrame with predictions
        ground_truth_nodes: DataFrame with ground truth nodes
        ground_truth_edges: DataFrame with ground truth edges
        degree_count: Dict of {node_index: degree}
        hub_threshold: Degree threshold for hub nodes
    
    Returns:
        DataFrame with evaluation results for each pathway
    """
    results = []
    
    for _, row in predictions.iterrows():
        pathway_id = row['pathway_id']
        
        # Get ground truth for this pathway
        gt_nodes = ground_truth_nodes[ground_truth_nodes['pathway_id'] == pathway_id]
        gt_edges = ground_truth_edges[ground_truth_edges['pathway_id'] == pathway_id]
        
        if len(gt_nodes) == 0:
            print(f"  Warning: No ground truth found for {pathway_id}")
            continue
        
        # Evaluate this pathway
        eval_result = evaluate_single_pathway(
            row, gt_nodes, gt_edges, degree_count, hub_threshold
        )
        eval_result['algorithm'] = algorithm_name
        results.append(eval_result)
    
    return pd.DataFrame(results)


def evaluate_all_algorithms(
    predictions_dict: Dict[str, pd.DataFrame],
    ground_truth_nodes: pd.DataFrame,
    ground_truth_edges: pd.DataFrame,
    degree_count: Dict[int, int],
    hub_threshold: float
) -> pd.DataFrame:
    """
    Evaluate all algorithms and combine results.
    
    Args:
        predictions_dict: Dict of {algorithm_name: predictions_df}
        ground_truth_nodes: DataFrame with ground truth nodes
        ground_truth_edges: DataFrame with ground truth edges
        degree_count: Dict of {node_index: degree}
        hub_threshold: Degree threshold for hub nodes
    
    Returns:
        Combined DataFrame with all evaluation results
    """
    all_results = []
    
    for algo_name, predictions in predictions_dict.items():
        print(f"Evaluating {algo_name}...")
        algo_results = evaluate_algorithm(
            algo_name, predictions,
            ground_truth_nodes, ground_truth_edges,
            degree_count, hub_threshold
        )
        all_results.append(algo_results)
        print(f"  ✓ {len(algo_results)} pathways evaluated")
    
    combined = pd.concat(all_results, ignore_index=True)
    
    # Reorder columns
    col_order = [
        'algorithm', 'pathway_id', 
        'precision', 'recall_at_6_hops', 'f1_score',
        'path_length_accuracy', 'hub_node_ratio', 'mrr',
        'edit_distance', 'relation_accuracy',
        'predicted_length', 'ground_truth_length'
    ]
    combined = combined[[c for c in col_order if c in combined.columns]]
    
    return combined


def generate_summary(results: pd.DataFrame) -> pd.DataFrame:
    """
    Generate summary statistics for each algorithm.
    
    Args:
        results: DataFrame from evaluate_all_algorithms
    
    Returns:
        Summary DataFrame with mean metrics per algorithm
    """
    metrics = [
        'precision', 'recall_at_6_hops', 'f1_score',
        'path_length_accuracy', 'hub_node_ratio', 'mrr',
        'edit_distance', 'relation_accuracy'
    ]
    
    # Filter to only metrics that exist in results
    metrics = [m for m in metrics if m in results.columns]
    
    summary = results.groupby('algorithm')[metrics].mean()
    
    # Round for display
    summary = summary.round(4)
    
    # Add count
    summary['n_pathways'] = results.groupby('algorithm').size()
    
    return summary


def print_comparison_table(summary: pd.DataFrame):
    """
    Print a formatted comparison table.
    """
    print("\n" + "="*80)
    print("ALGORITHM COMPARISON SUMMARY")
    print("="*80)
    print(summary.to_string())
    print("="*80)
    
    # Find best algorithm for each metric
    print("\nBest Algorithm per Metric:")
    print("-"*40)
    for col in summary.columns:
        if col == 'n_pathways':
            continue
        # For hub_node_ratio and edit_distance, lower is better
        if col in ['hub_node_ratio', 'edit_distance']:
            best = summary[col].idxmin()
            print(f"  {col:<25}: {best} ({summary.loc[best, col]:.4f}) [lower is better]")
        else:
            best = summary[col].idxmax()
            print(f"  {col:<25}: {best} ({summary.loc[best, col]:.4f})")


# ============================================
# QUICK START FUNCTION
# ============================================
def run_evaluation(
    predictions_dict: Dict[str, pd.DataFrame],
    ground_truth_nodes: pd.DataFrame,
    ground_truth_edges: pd.DataFrame,
    edges_df: pd.DataFrame
) -> tuple:
    """
    Quick start: Run full evaluation pipeline.
    
    Args:
        predictions_dict: Dict of {algorithm_name: predictions_df}
        ground_truth_nodes: Ground truth nodes DataFrame
        ground_truth_edges: Ground truth edges DataFrame
        edges_df: PrimeKG edges DataFrame (for computing degree)
    
    Returns:
        (detailed_results, summary)
    
    Example:
        results, summary = run_evaluation(
            predictions_dict={
                'Dijkstra': sp_predictions,
                'Hub-Penalized': hub_predictions,
                'PageRank-Inverse': pr_predictions
            },
            ground_truth_nodes=ground_truth_nodes,
            ground_truth_edges=ground_truth_edges,
            edges_df=edges
        )
    """
    print("="*60)
    print("RUNNING EVALUATION PIPELINE")
    print("="*60)
    
    # Step 1: Compute degree counts
    print("\n1. Computing node degrees...")
    degree_count = compute_degree_counts(edges_df)
    print(f"   ✓ Computed degrees for {len(degree_count):,} nodes")
    
    # Step 2: Compute hub threshold
    print("\n2. Computing hub threshold...")
    hub_threshold = compute_hub_threshold(degree_count, percentile=95)
    print(f"   ✓ Hub threshold (95th percentile): {hub_threshold:.0f}")
    
    # Step 3: Evaluate all algorithms
    print("\n3. Evaluating algorithms...")
    results = evaluate_all_algorithms(
        predictions_dict,
        ground_truth_nodes,
        ground_truth_edges,
        degree_count,
        hub_threshold
    )
    
    # Step 4: Generate summary
    print("\n4. Generating summary...")
    summary = generate_summary(results)
    
    # Step 5: Print comparison
    print_comparison_table(summary)
    
    return results, summary


print("  Use: results, summary = run_evaluation(predictions_dict, ground_truth_nodes, ground_truth_edges, edges)")
# ============================================================
# EFFICIENT HYPERPARAMETER TUNING
# ============================================================
#
# Run all algorithms with all parameter combinations in one go,
# then compare results at the end.
#
# Usage:
#   from hyperparameter_tuning import run_full_grid_search
#   best_configs, all_results = run_full_grid_search(G, ground_truth_nodes, edges)
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
from collections import defaultdict

# Import algorithms
from Algorithms import (
    HubPenalizedShortestPath,
    PageRankInverseShortestPath,
    SemanticBridgingPath
)

from evaluation_helpers import compute_degree_counts, compute_hub_threshold
from evaluation_metrics import precision, recall_at_k_hops, f1_score, path_length_accuracy, hub_node_ratio, mrr


def quick_evaluate(predictions_df, ground_truth_nodes, ground_truth_edges, degree_count, hub_threshold):
    """
    Fast evaluation - only compute key metrics (F1, hub_ratio, relation_accuracy).
    """
    results = []
    
    for _, row in predictions_df.iterrows():
        pathway_id = row['pathway_id']
        gt_nodes = ground_truth_nodes[ground_truth_nodes['pathway_id'] == pathway_id].sort_values('step_order')
        gt_edges = ground_truth_edges[ground_truth_edges['pathway_id'] == pathway_id]
        
        if len(gt_nodes) == 0:
            continue
        
        # Parse predictions
        if row['predicted_node_ids'] == 'NONE':
            pred_ids = []
            pred_indices = []
        else:
            pred_ids = row['predicted_node_ids'].split(',')
            pred_indices = [int(x) for x in row['predicted_node_indices'].split(',')]
        
        gt_ids = gt_nodes['node_id'].astype(str).tolist()
        
        # Calculate key metrics only
        results.append({
            'precision': precision(pred_ids, gt_ids),
            'recall': recall_at_k_hops(pred_ids, gt_ids, k=6),
            'f1_score': f1_score(pred_ids, gt_ids),
            'hub_node_ratio': hub_node_ratio(pred_indices, degree_count, hub_threshold),
            'path_length_accuracy': path_length_accuracy(row['predicted_length'], len(gt_nodes))
        })
    
    # Return averages
    if not results:
        return {'precision': 0, 'recall': 0, 'f1_score': 0, 'hub_node_ratio': 1, 'path_length_accuracy': 0}
    
    return {k: np.mean([r[k] for r in results]) for k in results[0].keys()}


def run_algorithm_on_pathways(algo, graph, ground_truth_df):
    """
    Run any algorithm on all pathways and return predictions DataFrame.
    """
    results = []
    
    for pathway_id in ground_truth_df['pathway_id'].unique():
        pathway_df = ground_truth_df[ground_truth_df['pathway_id'] == pathway_id].sort_values('step_order')
        
        source_idx = int(pathway_df.iloc[0]['node_index'])
        target_idx = int(pathway_df.iloc[-1]['node_index'])
        
        try:
            path, relations, weight = algo.find_path(source_idx, target_idx)
        except:
            path, relations = [], []
        
        if path:
            node_ids = [graph.nodes[idx].get('node_id', str(idx)) for idx in path]
            node_names = [graph.nodes[idx].get('node_name', str(idx)) for idx in path]
            
            results.append({
                'pathway_id': pathway_id,
                'predicted_node_indices': ','.join(map(str, path)),
                'predicted_node_ids': ','.join(node_ids),
                'predicted_node_names': ','.join(node_names),
                'predicted_relations': ','.join(relations),
                'predicted_length': len(path),
                'ground_truth_length': len(pathway_df)
            })
        else:
            results.append({
                'pathway_id': pathway_id,
                'predicted_node_indices': 'NONE',
                'predicted_node_ids': 'NONE',
                'predicted_node_names': 'NONE',
                'predicted_relations': 'NONE',
                'predicted_length': 0,
                'ground_truth_length': len(pathway_df)
            })
    
    return pd.DataFrame(results)


def run_full_grid_search(
    G, 
    ground_truth_nodes, 
    ground_truth_edges,
    edges_df,
    hub_alphas=[0.1, 0.3, 0.5, 0.7, 1.0, 1.5],
    pr_dampings=[0.7, 0.85, 0.9],
    semantic_betas=[0.1, 0.3, 0.5, 0.7]
):
    """
    Run grid search for all tunable algorithms at once.
    
    Returns:
        best_configs: Dict with best parameters for each algorithm
        all_results: DataFrame with all tuning results
    """
    print("="*70)
    print("EFFICIENT HYPERPARAMETER TUNING")
    print("="*70)
    
    # Pre-compute degree counts (only once!)
    print("\n1. Pre-computing degree counts...")
    degree_count = compute_degree_counts(edges_df)
    hub_threshold = compute_hub_threshold(degree_count)
    print(f"   ✓ Done (hub_threshold = {hub_threshold:.0f})")
    
    all_results = []
    
    # ============================================================
    # TUNE HUB-PENALIZED
    # ============================================================
    print(f"\n2. Tuning Hub-Penalized ({len(hub_alphas)} configs)...")
    
    for alpha in hub_alphas:
        print(f"   α = {alpha}...", end=' ')
        
        algo = HubPenalizedShortestPath(G, alpha=alpha)
        predictions = run_algorithm_on_pathways(algo, G, ground_truth_nodes)
        metrics = quick_evaluate(predictions, ground_truth_nodes, ground_truth_edges, degree_count, hub_threshold)
        
        all_results.append({
            'algorithm': 'Hub-Penalized',
            'param_name': 'alpha',
            'param_value': alpha,
            **metrics
        })
        print(f"F1={metrics['f1_score']:.4f}")
    
    # ============================================================
    # TUNE PAGERANK-INVERSE
    # ============================================================
    print(f"\n3. Tuning PageRank-Inverse ({len(pr_dampings)} configs)...")
    
    # Pre-compute PageRank once per damping value
    for damping in pr_dampings:
        print(f"   damping = {damping}...", end=' ')
        
        algo = PageRankInverseShortestPath(G, damping=damping)
        predictions = run_algorithm_on_pathways(algo, G, ground_truth_nodes)
        metrics = quick_evaluate(predictions, ground_truth_nodes, ground_truth_edges, degree_count, hub_threshold)
        
        all_results.append({
            'algorithm': 'PageRank-Inverse',
            'param_name': 'damping',
            'param_value': damping,
            **metrics
        })
        print(f"F1={metrics['f1_score']:.4f}")
    
    # ============================================================
    # TUNE SEMANTIC-BRIDGING
    # ============================================================
    print(f"\n4. Tuning Semantic-Bridging ({len(semantic_betas)} configs)...")
    
    for beta in semantic_betas:
        print(f"   β = {beta}...", end=' ')
        
        algo = SemanticBridgingPath(G, beta=beta)
        algo.compute_embeddings()
        algo.compute_edge_weights()
        predictions = run_algorithm_on_pathways(algo, G, ground_truth_nodes)
        metrics = quick_evaluate(predictions, ground_truth_nodes, ground_truth_edges, degree_count, hub_threshold)
        
        all_results.append({
            'algorithm': 'Semantic-Bridging',
            'param_name': 'beta',
            'param_value': beta,
            **metrics
        })
        print(f"F1={metrics['f1_score']:.4f}")
    
    # Convert to DataFrame
    results_df = pd.DataFrame(all_results)
    
    # ============================================================
    # FIND BEST CONFIGS
    # ============================================================
    print("\n" + "="*70)
    print("BEST CONFIGURATIONS (by F1 Score)")
    print("="*70)
    
    best_configs = {}
    
    for algo in results_df['algorithm'].unique():
        algo_results = results_df[results_df['algorithm'] == algo]
        best_idx = algo_results['f1_score'].idxmax()
        best_row = algo_results.loc[best_idx]
        
        best_configs[algo] = {
            'param_name': best_row['param_name'],
            'param_value': best_row['param_value'],
            'f1_score': best_row['f1_score'],
            'hub_node_ratio': best_row['hub_node_ratio']
        }
        
        print(f"\n{algo}:")
        print(f"   Best {best_row['param_name']} = {best_row['param_value']}")
        print(f"   F1 Score: {best_row['f1_score']:.4f}")
        print(f"   Hub Ratio: {best_row['hub_node_ratio']:.4f}")
    
    return best_configs, results_df


def plot_tuning_results(results_df, save_path='tuning_results.png'):
    """
    Plot tuning results for all algorithms.
    """
    algorithms = results_df['algorithm'].unique()
    
    fig, axes = plt.subplots(1, len(algorithms), figsize=(5*len(algorithms), 5))
    if len(algorithms) == 1:
        axes = [axes]
    
    for ax, algo in zip(axes, algorithms):
        algo_df = results_df[results_df['algorithm'] == algo]
        param_name = algo_df['param_name'].iloc[0]
        
        # Plot F1 Score
        ax.plot(algo_df['param_value'], algo_df['f1_score'], 'b-o', linewidth=2, markersize=8, label='F1 Score')
        
        # Plot Hub Ratio on secondary axis
        ax2 = ax.twinx()
        ax2.plot(algo_df['param_value'], algo_df['hub_node_ratio'], 'r--s', linewidth=2, markersize=8, label='Hub Ratio')
        
        # Mark best F1
        best_idx = algo_df['f1_score'].idxmax()
        best_param = algo_df.loc[best_idx, 'param_value']
        best_f1 = algo_df.loc[best_idx, 'f1_score']
        ax.axvline(x=best_param, color='green', linestyle=':', linewidth=2, alpha=0.7)
        ax.scatter([best_param], [best_f1], color='green', s=150, zorder=5, marker='★')
        
        ax.set_xlabel(param_name, fontsize=11)
        ax.set_ylabel('F1 Score', color='blue', fontsize=11)
        ax2.set_ylabel('Hub Ratio (↓)', color='red', fontsize=11)
        ax.set_title(f'{algo}\n(best {param_name}={best_param})', fontsize=12, fontweight='bold')
        ax.tick_params(axis='y', labelcolor='blue')
        ax2.tick_params(axis='y', labelcolor='red')
        ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved: {save_path}")
    plt.show()


def get_best_predictions(G, ground_truth_nodes, best_configs):
    """
    Get predictions using the best configurations found.
    
    Returns:
        Dict of {algorithm_name: predictions_df}
    """
    print("\n" + "="*70)
    print("GENERATING PREDICTIONS WITH BEST CONFIGS")
    print("="*70)
    
    predictions_dict = {}
    
    for algo_name, config in best_configs.items():
        print(f"\n{algo_name} ({config['param_name']}={config['param_value']})...")
        
        if algo_name == 'Hub-Penalized':
            algo = HubPenalizedShortestPath(G, alpha=config['param_value'])
        elif algo_name == 'PageRank-Inverse':
            algo = PageRankInverseShortestPath(G, damping=config['param_value'])
        elif algo_name == 'Semantic-Bridging':
            algo = SemanticBridgingPath(G, beta=config['param_value'])
            algo.compute_embeddings()
            algo.compute_edge_weights()
        else:
            continue
        
        predictions = run_algorithm_on_pathways(algo, G, ground_truth_nodes)
        predictions_dict[algo_name] = predictions
        print(f"   ✓ {len(predictions)} pathways")
    
    return predictions_dict


print("✓ Hyperparameter tuning module loaded")
print("  Use: best_configs, tuning_results = run_full_grid_search(G, ground_truth_nodes, ground_truth_edges, edges)")

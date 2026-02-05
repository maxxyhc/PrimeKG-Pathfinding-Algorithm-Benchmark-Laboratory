"""
Evaluation Helper Functions for Drug Repurposing Benchmark
===========================================================

This module contains helper functions used by the main evaluation metrics.
Import these into your notebook with: from evaluation_helpers import *
"""

import numpy as np
from collections import Counter


def calculate_set_intersection(pred_set, gt_set):
    """Calculate intersection between predicted and ground truth sets."""
    return len(pred_set & gt_set)


def calculate_edit_distance(predicted_ids, ground_truth_ids):
    """
    Calculate normalized Levenshtein edit distance between two sequences.
    
    Returns value in [0, 1] where 0 = identical, 1 = completely different.
    """
    if not predicted_ids or predicted_ids == ['NONE']:
        return 1.0
    
    m, n = len(predicted_ids), len(ground_truth_ids)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if predicted_ids[i-1] == ground_truth_ids[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    
    return dp[m][n] / max(m, n)


def compute_degree_counts(edges_df):
    """
    Compute degree count for all nodes from edges dataframe.
    
    Returns:
        dict: {node_index: degree_count}
    """
    degree_count = Counter()
    for _, row in edges_df.iterrows():
        degree_count[row['x_index']] += 1
        degree_count[row['y_index']] += 1
    return degree_count


def compute_hub_threshold(degree_count, percentile=95):
    """
    Compute hub threshold based on degree distribution.
    
    Args:
        degree_count: dict of {node_index: degree}
        percentile: percentile cutoff for hub definition (default 95)
    
    Returns:
        float: degree threshold above which nodes are considered hubs
    """
    all_degrees = list(degree_count.values())
    return np.percentile(all_degrees, percentile)


def is_valid_prediction(predicted_ids):
    """Check if prediction is valid (not empty or 'NONE')."""
    return predicted_ids and predicted_ids != ['NONE']


def calculate_hits_at_k(predicted_ids, ground_truth_target, k_values=[1, 3, 5]):
    """
    Check if the target appears in the last k nodes of the predicted path.
    
    For drug repurposing, the disease should be at the END of the path.
    """
    hits = {f'hits_at_{k}': 0 for k in k_values}
    
    if not is_valid_prediction(predicted_ids):
        return hits
    
    for k in k_values:
        last_k = predicted_ids[-k:] if len(predicted_ids) >= k else predicted_ids
        hits[f'hits_at_{k}'] = 1 if ground_truth_target in last_k else 0
    
    return hits


def calculate_relation_accuracy(predicted_relations, ground_truth_edge_types):
    """
    Calculate what percentage of predicted edges use correct relation types.
    """
    if not predicted_relations:
        return 0.0
    
    gt_types = set(ground_truth_edge_types)
    matches = sum(1 for r in predicted_relations if r in gt_types)
    
    return matches / len(predicted_relations)

def calculate_path_length_mae(predicted_length, ground_truth_length):
    return abs(predicted_length - ground_truth_length)



def calculate_hub_node_ratio(predicted_indices, degree_count, hub_threshold):
    if not predicted_indices:
        return 0
    
    hub_count = sum(
        1 for idx in predicted_indices
        if degree_count.get(idx, 0) >= hub_threshold
    )
    return hub_count / len(predicted_indices)

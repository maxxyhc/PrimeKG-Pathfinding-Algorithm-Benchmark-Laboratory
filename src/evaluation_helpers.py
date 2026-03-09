"""
Evaluation Helper Functions for Drug Repurposing Benchmark
===========================================================

This module contains helper functions used by the main evaluation metrics.
Import these into your notebook with: from evaluation_helpers import *
"""

import numpy as np
import time
from collections import Counter


def is_valid_prediction(predicted_ids):
    """Check if prediction is valid (not empty or 'NONE')."""
    return bool(predicted_ids and predicted_ids != ['NONE'])


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


def calculate_relation_accuracy(predicted_relations, ground_truth_edge_types):
    """
    Calculate what percentage of predicted edges use correct relation types.
    """
    if not predicted_relations:
        return 0.0
    
    gt_types = set(ground_truth_edge_types)
    matches = sum(1 for r in predicted_relations if r in gt_types)
    
    return matches / len(predicted_relations)

def speed(run_fn, *args, **kwargs):
    """
    Measure execution time of a function in milliseconds.
    
    Args:
        run_fn: function to time
        *args, **kwargs: arguments to pass to run_fn
    
    Returns:
        tuple: (result, elapsed_time_ms)
    """
    start = time.perf_counter()
    result = run_fn(*args, **kwargs)
    elapsed_ms = (time.perf_counter() - start) * 1000
    
    return result, elapsed_ms


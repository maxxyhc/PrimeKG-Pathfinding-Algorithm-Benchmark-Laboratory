# ============================================
# EVALUATION METRICS
# ============================================
# 
# 7 Main Metrics:
#   1. precision          - What fraction of predicted nodes are correct?
#   2. recall_at_6_hops   - What fraction of GT nodes found within 6 hops?
#   3. f1_score           - Harmonic mean of precision and recall
#   4. path_length_accuracy - How close is predicted length to GT length?
#   5. hub_node_ratio     - What fraction of path nodes are hubs?
#   6. mrr                - Mean Reciprocal Rank of first correct node
#   7. speed              - Execution time in milliseconds
#
# Helper functions are in evaluation_helpers.py
# ============================================
# Define Evaluation Metrics

# We evaluate algorithms using 9 metrics across three categories:

# | Category | Metrics | What It Measures |
# |----------|---------|------------------|
# | **Node Accuracy**  | Precision, Recall, F1 | Are the right nodes in the path? |
# | **Target Finding** | Hits@1, Hits@3, Hits@5 | Does the path reach the disease? |
# | **Mechanistic Quality** | Relation Accuracy, Edit Distance, Hub Ratio | Is the path biologically valid? |

import time
from evaluation_helpers import (
    is_valid_prediction,
    calculate_edit_distance,
    compute_degree_counts,
    compute_hub_threshold,
    calculate_hits_at_k,
    calculate_relation_accuracy
)


# ============================================
# 1. PRECISION
# ============================================
def precision(predicted_ids, ground_truth_ids):
    """
    What fraction of predicted nodes are correct?
    
    Formula: |predicted ∩ ground_truth| / |predicted|
    """
    if not is_valid_prediction(predicted_ids):
        return 0.0
    
    pred_set = set(predicted_ids)
    gt_set = set(ground_truth_ids)
    intersection = len(pred_set & gt_set)
    
    return intersection / len(pred_set)


# ============================================
# 2. RECALL @ 6 HOPS
# ============================================
def recall_at_k_hops(predicted_ids, ground_truth_ids, k=6):
    """
    What fraction of ground truth nodes are found within first k hops?
    
    Formula: |predicted[:k] ∩ ground_truth| / |ground_truth|
    """
    if not is_valid_prediction(predicted_ids):
        return 0.0
    
    truncated = predicted_ids[:k]
    gt_set = set(ground_truth_ids)
    
    return len(set(truncated) & gt_set) / len(gt_set)


# ============================================
# 3. F1 SCORE
# ============================================
def f1_score(predicted_ids, ground_truth_ids):
    """
    Harmonic mean of precision and recall.
    
    Formula: 2 * (precision * recall) / (precision + recall)
    """
    if not is_valid_prediction(predicted_ids):
        return 0.0
    
    pred_set = set(predicted_ids)
    gt_set = set(ground_truth_ids)
    intersection = len(pred_set & gt_set)
    
    prec = intersection / len(pred_set) if pred_set else 0
    rec = intersection / len(gt_set) if gt_set else 0
    
    if prec + rec == 0:
        return 0.0
    
    return 2 * (prec * rec) / (prec + rec)


# ============================================
# 4. PATH LENGTH ACCURACY
# ============================================
def path_length_accuracy(predicted_length, ground_truth_length):
    """
    How close is predicted path length to ground truth?
    
    Formula: 1 - |predicted_length - gt_length| / max(predicted_length, gt_length)
    
    Returns value in [0, 1] where 1 = exact match
    """
    if predicted_length == 0 and ground_truth_length == 0:
        return 1.0
    
    max_len = max(predicted_length, ground_truth_length)
    if max_len == 0:
        return 0.0
    
    return 1 - abs(predicted_length - ground_truth_length) / max_len


# ============================================
# 5. HUB NODE RATIO
# ============================================
def hub_node_ratio(predicted_indices, degree_count, hub_threshold):
    """
    What fraction of predicted nodes are hubs (high-degree nodes)?
    
    Lower is better - means path avoids generic hub shortcuts.
    
    Args:
        predicted_indices: list of node indices in predicted path
        degree_count: dict of {node_index: degree}
        hub_threshold: degree above which a node is considered a hub
    """
    if not predicted_indices:
        return 0.0
    
    hub_count = sum(
        1 for idx in predicted_indices
        if degree_count.get(idx, 0) >= hub_threshold
    )
    
    return hub_count / len(predicted_indices)


# ============================================
# 6. MEAN RECIPROCAL RANK (MRR)
# ============================================
def mrr(predicted_ids, ground_truth_ids):
    """
    Mean Reciprocal Rank - how early does the first correct node appear?
    
    Formula: 1 / rank_of_first_correct_node
    
    Higher is better (1.0 = first node is correct)
    """
    if not is_valid_prediction(predicted_ids):
        return 0.0
    
    gt_set = set(ground_truth_ids)
    
    for rank, node in enumerate(predicted_ids, start=1):
        if node in gt_set:
            return 1 / rank
    
    return 0.0


# ============================================
# 7. SPEED (ms)
# ============================================
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


# ============================================
# COMBINED EVALUATION FUNCTION
# ============================================
def evaluate_pathway(predicted_ids, predicted_indices, predicted_length,
                     ground_truth_ids, ground_truth_length,
                     degree_count, hub_threshold):
    """
    Run all 7 metrics on a single pathway prediction.
    
    Returns:
        dict with all metric values
    """
    return {
        'precision': precision(predicted_ids, ground_truth_ids),
        'recall_at_6_hops': recall_at_k_hops(predicted_ids, ground_truth_ids),
        'f1_score': f1_score(predicted_ids, ground_truth_ids),
        'path_length_accuracy': path_length_accuracy(predicted_length, ground_truth_length),
        'hub_node_ratio': hub_node_ratio(predicted_indices, degree_count, hub_threshold),
        'mrr': mrr(predicted_ids, ground_truth_ids)
        # Note: speed is measured separately when running the algorithm
    }


print("✓ Evaluation metrics loaded:")
print("  1. precision")
print("  2. recall_at_6_hops")
print("  3. f1_score")
print("  4. path_length_accuracy")
print("  5. hub_node_ratio")
print("  6. mrr")
print("  7. speed")

# ============================================
# EVALUATION METRICS
# ============================================
# 
# 7 Main Metrics:
#   1. precision           - What fraction of predicted nodes are correct?
#   2. recall              - What fraction of GT nodes are found?
#   3. f1_score            - Harmonic mean of precision and recall
#   4. edit_distance       - Structural similarity to GT path (lower is better)
#   5. mrr                 - Mean Reciprocal Rank of first correct intermediate node
#   6. hub_node_ratio      - What fraction of path nodes are hubs? (lower is better)
#   7. path_length_accuracy - How close is predicted length to GT length?
#
# Helper functions are in evaluation_helpers.py
# ============================================
# 
# We evaluate algorithms using 7 metrics across three categories:
#
# | Category | Metrics | What It Measures |
# |----------|---------|------------------|
# | **Node Overlap**  | Precision, Recall, F1 | Are the right nodes in the path? |
# | **Sequence Quality** | Edit Distance, MRR | Is the path structure correct? |
# | **Path Characteristics** | Hub Node Ratio, Path Length Accuracy | Does the path have good properties? |
#
# Note: Speed is measured separately in the runner during algorithm execution.
# ============================================

from src.evaluation_helpers import *

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
# 2. RECALL
# ============================================
def recall(predicted_ids, ground_truth_ids):
    """
    What fraction of ground truth nodes are found in the predicted path?
    
    Formula: |predicted ∩ ground_truth| / |ground_truth|
    """
    if not is_valid_prediction(predicted_ids):
        return 0.0
    
    pred_set = set(predicted_ids)
    gt_set = set(ground_truth_ids)
    
    return len(pred_set & gt_set) / len(gt_set) if gt_set else 0.0


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
    
    prec = precision(predicted_ids, ground_truth_ids)
    rec = recall(predicted_ids, ground_truth_ids)
    
    if prec + rec == 0:
        return 0.0
    
    return 2 * (prec * rec) / (prec + rec)

# ============================================
# 4. EDIT DISTANCE 
# ============================================
from difflib import SequenceMatcher

def edit_distance(predicted_ids, ground_truth_ids, algorithm_name=None):
    """
    Calculate edit distance between two sequences.
    
    Returns value in [0, 1] where 0 = identical, 1 = completely different.
    """
    if not predicted_ids or predicted_ids == ['NONE']:
        return 1.0
    
    # Bidirectional algorithms use SequenceMatcher (OLD baseline)
    PHASE_2_ALGOS = ['Bidirectional', 'KShortestBio', 'BidirRelationWeighted']
    
    if algorithm_name in PHASE_2_ALGOS:
        sm = SequenceMatcher(None, ground_truth_ids, predicted_ids)  # Note: gt first!
        return 1 - sm.ratio()
    
    # Other algorithms use Levenshtein with max(m,n)
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
    
    return dp[m][n] / max(m, n) if max(m, n) > 0 else 0.0

# ============================================
# 5. MEAN RECIPROCAL RANK (MRR)
# ============================================
def mrr(predicted_ids, ground_truth_ids):
    """
    Mean Reciprocal Rank - how early does the first correct INTERMEDIATE node appear?
    Excludes source (first) and target (last) nodes.
    
    Formula: 1 / rank_of_first_correct_intermediate_node
    Higher is better (1.0 = first intermediate node is correct)
    """
    if not is_valid_prediction(predicted_ids):
        return 0.0
    
    # Need at least 3 nodes (source, intermediate, target)
    if len(predicted_ids) < 3:
        return 0.0
    
    intermediate_predicted = predicted_ids[1:-1]
    gt_intermediate = set(ground_truth_ids[1:-1])
    
    for rank, node in enumerate(intermediate_predicted, start=1):
        if node in gt_intermediate:
            return 1 / rank
    
    return 0.0


# ============================================
# 6. HUB NODE RATIO
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
# 7. PATH LENGTH ACCURACY
# ============================================
def path_length_accuracy(predicted_length, ground_truth_length):
    """
    How close is predicted path length to ground truth?
    
    Formula: 1 - |predicted_length - gt_length| / max(predicted_length, gt_length)
    
    Returns value in [0, 1] where 1 = exact match
    """
    max_len = max(predicted_length, ground_truth_length)
    
    if max_len == 0:
        return 1.0 # Both lengths are zero, consider it a perfect match
    
    return 1 - abs(predicted_length - ground_truth_length) / max_len

# ============================================
# COMBINED EVALUATION FUNCTION
# ============================================
def evaluate_pathway(predicted_ids, predicted_indices, predicted_length,
                     ground_truth_ids, ground_truth_length,
                     degree_count, hub_threshold, algorithm_name=None):  # ← ADD THIS
    """
    Run all 7 metrics on a single pathway prediction.
    
    Returns:
        dict with all metric values
    """
    return {
        'precision': precision(predicted_ids, ground_truth_ids),
        'recall': recall(predicted_ids, ground_truth_ids),
        'f1_score': f1_score(predicted_ids, ground_truth_ids),
        'edit_distance': edit_distance(predicted_ids, ground_truth_ids, algorithm_name),  # ← PASS IT HERE
        'mrr': mrr(predicted_ids, ground_truth_ids),
        'hub_node_ratio': hub_node_ratio(predicted_indices, degree_count, hub_threshold),
        'path_length_accuracy': path_length_accuracy(predicted_length, ground_truth_length)
    }
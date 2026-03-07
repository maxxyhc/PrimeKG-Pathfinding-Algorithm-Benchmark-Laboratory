"""
Tests for evaluation metrics.
Run with: pytest tests/test_evaluation_metrics.py -v
"""
import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'notebook'))

from evaluation_metrics import (
    precision,
    recall_at_k_hops,
    f1_score,
    path_length_accuracy,
    hub_node_ratio,
    mrr,
    speed,
    evaluate_pathway
)


class TestPrecision:
    def test_perfect_precision(self):
        assert precision([1, 2, 3], [1, 2, 3]) == 1.0

    def test_half_precision(self):
        assert precision([1, 2, 5, 6], [1, 2, 3, 4]) == 0.5

    def test_no_overlap(self):
        assert precision([5, 6, 7], [1, 2, 3]) == 0.0

    def test_invalid_prediction(self):
        assert precision([], [1, 2, 3]) == 0.0

    def test_none_prediction(self):
        assert precision(['NONE'], [1, 2, 3]) == 0.0


class TestRecall:
    def test_perfect_recall(self):
        assert recall_at_k_hops([1, 2, 3], [1, 2, 3]) == 1.0

    def test_partial_recall(self):
        assert recall_at_k_hops([1, 2], [1, 2, 3, 4]) == 0.5

    def test_with_k_limit(self):
        result = recall_at_k_hops([1, 2, 99], [1, 2, 3], k=2)
        assert abs(result - 2/3) < 0.01

    def test_invalid_prediction(self):
        assert recall_at_k_hops([], [1, 2, 3]) == 0.0

    def test_no_overlap(self):
        assert recall_at_k_hops([7, 8, 9], [1, 2, 3]) == 0.0


class TestF1Score:
    def test_perfect_f1(self):
        assert f1_score([1, 2, 3], [1, 2, 3]) == 1.0

    def test_zero_f1(self):
        assert f1_score([4, 5, 6], [1, 2, 3]) == 0.0

    def test_known_f1(self):
        assert f1_score([1, 2, 3, 4], [1, 2, 5, 6]) == 0.5

    def test_invalid_prediction(self):
        assert f1_score([], [1, 2, 3]) == 0.0

    def test_asymmetric(self):
        result = f1_score([1, 2], [1, 2, 3])
        assert abs(result - 0.8) < 0.01


class TestPathLengthAccuracy:
    def test_exact_match(self):
        assert path_length_accuracy(5, 5) == 1.0

    def test_off_by_one(self):
        assert path_length_accuracy(5, 4) == 0.8

    def test_very_different(self):
        assert abs(path_length_accuracy(10, 2) - 0.2) < 0.01

    def test_both_zero(self):
        assert path_length_accuracy(0, 0) == 1.0

    def test_one_zero(self):
        assert path_length_accuracy(0, 5) == 0.0


class TestHubNodeRatio:
    def test_no_hubs(self):
        degree_count = {1: 10, 2: 5, 3: 8}
        assert hub_node_ratio([1, 2, 3], degree_count, hub_threshold=100) == 0.0

    def test_all_hubs(self):
        degree_count = {1: 500, 2: 600}
        assert hub_node_ratio([1, 2], degree_count, hub_threshold=100) == 1.0

    def test_empty_path(self):
        assert hub_node_ratio([], {}, hub_threshold=100) == 0.0

    def test_node_not_in_degree_count(self):
        degree_count = {1: 500}
        assert hub_node_ratio([1, 99], degree_count, hub_threshold=100) == 0.5


class TestMRR:
    def test_first_intermediate_correct(self):
        assert mrr([1, 2, 3], [1, 2, 3]) == 1.0

    def test_second_intermediate_correct(self):
        assert mrr([1, 99, 2, 3], [1, 2, 3]) == 0.5

    def test_no_intermediate_match(self):
        assert mrr([1, 99, 3], [1, 2, 3]) == 0.0

    def test_too_short_for_intermediates(self):
        assert mrr([1, 3], [1, 2, 3]) == 0.0

    def test_invalid_prediction(self):
        assert mrr([], [1, 2, 3]) == 0.0


class TestSpeed:
    def test_returns_result_and_time(self):
        result, elapsed = speed(lambda x: x * 2, 5)
        assert result == 10
        assert elapsed >= 0

    def test_timing_is_reasonable(self):
        import time
        result, elapsed = speed(time.sleep, 0.01)
        assert elapsed >= 10


class TestEvaluatePathway:
    def test_perfect_prediction(self):
        degree_count = {1: 10, 2: 15, 3: 20}
        results = evaluate_pathway(
            predicted_ids=[1, 2, 3],
            predicted_indices=[1, 2, 3],
            predicted_length=3,
            ground_truth_ids=[1, 2, 3],
            ground_truth_length=3,
            degree_count=degree_count,
            hub_threshold=100
        )
        assert results['precision'] == 1.0
        assert results['recall_at_6_hops'] == 1.0
        assert results['f1_score'] == 1.0
        assert results['path_length_accuracy'] == 1.0
        assert results['hub_node_ratio'] == 0.0
        assert results['mrr'] == 1.0

    def test_no_overlap_prediction(self):
        degree_count = {4: 10, 5: 15, 6: 20}
        results = evaluate_pathway(
            predicted_ids=[4, 5, 6],
            predicted_indices=[4, 5, 6],
            predicted_length=3,
            ground_truth_ids=[1, 2, 3],
            ground_truth_length=3,
            degree_count=degree_count,
            hub_threshold=100
        )
        assert results['precision'] == 0.0
        assert results['recall_at_6_hops'] == 0.0
        assert results['f1_score'] == 0.0

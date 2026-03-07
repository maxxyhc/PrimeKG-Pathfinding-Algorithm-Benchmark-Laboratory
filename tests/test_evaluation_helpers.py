"""
Tests for evaluation helper functions.
Run with: pytest tests/test_evaluation_helpers.py -v
"""
import sys
import os
import pytest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'notebook'))

from evaluation_helpers import (
    is_valid_prediction,
    calculate_edit_distance,
    compute_hub_threshold,
    calculate_hits_at_k,
    calculate_relation_accuracy,
    calculate_set_intersection,
    calculate_path_length_mae,
    calculate_hub_node_ratio
)


class TestIsValidPrediction:
    def test_valid_list(self):
        assert is_valid_prediction([1, 2, 3]) == True

    def test_empty_list(self):
        assert not is_valid_prediction([])

    def test_none_string(self):
        assert is_valid_prediction(['NONE']) == False

    def test_none_value(self):
        assert not is_valid_prediction(None)

    def test_single_node(self):
        assert is_valid_prediction([42]) == True


class TestSetIntersection:
    def test_full_overlap(self):
        assert calculate_set_intersection({1, 2, 3}, {1, 2, 3}) == 3

    def test_partial_overlap(self):
        assert calculate_set_intersection({1, 2, 3}, {2, 3, 4}) == 2

    def test_no_overlap(self):
        assert calculate_set_intersection({1, 2}, {3, 4}) == 0

    def test_empty_sets(self):
        assert calculate_set_intersection(set(), set()) == 0


class TestEditDistance:
    def test_identical_sequences(self):
        assert calculate_edit_distance(['A', 'B', 'C'], ['A', 'B', 'C']) == 0.0

    def test_completely_different(self):
        assert calculate_edit_distance(['A', 'B'], ['C', 'D']) == 1.0

    def test_one_substitution(self):
        result = calculate_edit_distance(['A', 'B', 'C'], ['A', 'X', 'C'])
        assert abs(result - 1/3) < 0.01

    def test_different_lengths(self):
        result = calculate_edit_distance(['A', 'B'], ['A', 'B', 'C'])
        assert abs(result - 1/3) < 0.01

    def test_empty_prediction(self):
        assert calculate_edit_distance([], ['A', 'B']) == 1.0

    def test_none_prediction(self):
        assert calculate_edit_distance(['NONE'], ['A', 'B']) == 1.0


class TestHitsAtK:
    def test_target_at_end(self):
        hits = calculate_hits_at_k(['drug', 'protein', 'disease'], 'disease')
        assert hits['hits_at_1'] == 1
        assert hits['hits_at_3'] == 1
        assert hits['hits_at_5'] == 1

    def test_target_not_present(self):
        hits = calculate_hits_at_k(['drug', 'protein', 'gene'], 'disease')
        assert hits['hits_at_1'] == 0
        assert hits['hits_at_3'] == 0
        assert hits['hits_at_5'] == 0

    def test_target_second_from_end(self):
        hits = calculate_hits_at_k(['drug', 'disease', 'gene'], 'disease')
        assert hits['hits_at_1'] == 0
        assert hits['hits_at_3'] == 1

    def test_invalid_prediction(self):
        hits = calculate_hits_at_k([], 'disease')
        assert hits['hits_at_1'] == 0
        assert hits['hits_at_3'] == 0
        assert hits['hits_at_5'] == 0


class TestRelationAccuracy:
    def test_all_correct(self):
        predicted = ['ppi', 'drug_protein', 'disease_protein']
        gt = ['ppi', 'drug_protein', 'disease_protein']
        assert calculate_relation_accuracy(predicted, gt) == 1.0

    def test_none_correct(self):
        predicted = ['ppi', 'ppi']
        gt = ['drug_protein', 'disease_protein']
        assert calculate_relation_accuracy(predicted, gt) == 0.0

    def test_partial_match(self):
        predicted = ['ppi', 'drug_protein', 'fake_relation']
        gt = ['ppi', 'drug_protein']
        assert abs(calculate_relation_accuracy(predicted, gt) - 2/3) < 0.01

    def test_empty_predictions(self):
        assert calculate_relation_accuracy([], ['ppi']) == 0.0


class TestPathLengthMAE:
    def test_exact_match(self):
        assert calculate_path_length_mae(5, 5) == 0

    def test_off_by_two(self):
        assert calculate_path_length_mae(3, 5) == 2

    def test_predicted_longer(self):
        assert calculate_path_length_mae(8, 5) == 3


class TestHubNodeRatio:
    def test_no_hubs(self):
        degree_count = {1: 10, 2: 5, 3: 8}
        assert calculate_hub_node_ratio([1, 2, 3], degree_count, hub_threshold=100) == 0.0

    def test_all_hubs(self):
        degree_count = {1: 500, 2: 600, 3: 700}
        assert calculate_hub_node_ratio([1, 2, 3], degree_count, hub_threshold=100) == 1.0

    def test_mixed(self):
        degree_count = {1: 500, 2: 10, 3: 600}
        result = calculate_hub_node_ratio([1, 2, 3], degree_count, hub_threshold=100)
        assert abs(result - 2/3) < 0.01

    def test_empty_path(self):
        assert calculate_hub_node_ratio([], {}, hub_threshold=100) == 0


class TestComputeHubThreshold:
    def test_basic_threshold(self):
        degree_count = {i: i for i in range(1, 101)}
        threshold = compute_hub_threshold(degree_count, percentile=95)
        assert threshold >= 95

    def test_uniform_degrees(self):
        degree_count = {i: 50 for i in range(100)}
        threshold = compute_hub_threshold(degree_count, percentile=95)
        assert threshold == 50.0

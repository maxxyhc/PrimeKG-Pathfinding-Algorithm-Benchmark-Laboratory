"""
Tests for evaluation metrics.
Run with: pytest tests/test_evaluation_metrics.py -v
"""
import pytest
from evaluation_metrics import (
    precision,
    recall,
    f1_score,
    edit_distance,
    mrr,
    hub_node_ratio,
    path_length_accuracy,
    evaluate_pathway
)


class TestPrecision:
    """Test precision: |predicted ∩ ground_truth| / |predicted|"""
    
    def test_perfect_match(self):
        """All predicted nodes are correct"""
        assert precision(['A', 'B', 'C'], ['A', 'B', 'C']) == 1.0
    
    def test_no_match(self):
        """No predicted nodes are correct"""
        assert precision(['X', 'Y', 'Z'], ['A', 'B', 'C']) == 0.0
    
    def test_partial_match(self):
        """Half of predicted nodes are correct"""
        assert precision(['A', 'B', 'X', 'Y'], ['A', 'B', 'C', 'D']) == 0.5
    
    def test_empty_prediction(self):
        """Empty prediction returns 0"""
        assert precision([], ['A', 'B', 'C']) == 0.0
    
    def test_none_prediction(self):
        """NONE prediction returns 0"""
        assert precision(['NONE'], ['A', 'B', 'C']) == 0.0


class TestRecall:
    """Test recall: |predicted ∩ ground_truth| / |ground_truth|"""
    
    def test_perfect_match(self):
        """All GT nodes are found"""
        assert recall(['A', 'B', 'C'], ['A', 'B', 'C']) == 1.0
    
    def test_no_match(self):
        """No GT nodes are found"""
        assert recall(['X', 'Y', 'Z'], ['A', 'B', 'C']) == 0.0
    
    def test_partial_match(self):
        """Half of GT nodes are found"""
        assert recall(['A', 'B'], ['A', 'B', 'C', 'D']) == 0.5
    
    def test_empty_prediction(self):
        """Empty prediction returns 0"""
        assert recall([], ['A', 'B', 'C']) == 0.0
    
    def test_superset_prediction(self):
        """Predicted has all GT nodes plus extras"""
        assert recall(['A', 'B', 'C', 'X'], ['A', 'B', 'C']) == 1.0


class TestF1Score:
    """Test F1: harmonic mean of precision and recall"""
    
    def test_perfect_match(self):
        """Perfect precision and recall"""
        assert f1_score(['A', 'B', 'C'], ['A', 'B', 'C']) == 1.0
    
    def test_no_match(self):
        """No overlap between predicted and GT"""
        assert f1_score(['X', 'Y', 'Z'], ['A', 'B', 'C']) == 0.0
    
    def test_balanced(self):
        """Precision=0.5, Recall=0.5, F1=0.5"""
        assert f1_score(['A', 'B', 'X', 'Y'], ['A', 'B', 'C', 'D']) == 0.5
    
    def test_empty_prediction(self):
        """Empty prediction returns 0"""
        assert f1_score([], ['A', 'B', 'C']) == 0.0
    
    def test_asymmetric(self):
        """Different precision and recall"""
        # pred=['A','B'], gt=['A','B','C']
        # precision=2/2=1.0, recall=2/3=0.667
        # f1 = 2*(1.0*0.667)/(1.0+0.667) = 0.8
        result = f1_score(['A', 'B'], ['A', 'B', 'C'])
        assert pytest.approx(result, rel=0.01) == 0.8


class TestEditDistance:
    """Test edit distance: normalized Levenshtein distance"""
    
    def test_identical_sequences(self):
        """Identical sequences have distance 0"""
        assert edit_distance(['A', 'B', 'C'], ['A', 'B', 'C']) == 0.0
    
    def test_completely_different(self):
        """Completely different sequences have distance 1"""
        assert edit_distance(['A', 'B', 'C'], ['X', 'Y', 'Z']) == 1.0
    
    def test_one_substitution(self):
        """One substitution needed"""
        # ['A','B','C'] -> ['A','X','C'] requires 1 edit, length=3
        result = edit_distance(['A', 'B', 'C'], ['A', 'X', 'C'])
        assert pytest.approx(result, rel=0.01) == 1/3
    
    def test_empty_prediction(self):
        """Empty prediction has max distance"""
        assert edit_distance([], ['A', 'B', 'C']) == 1.0
    
    def test_none_prediction(self):
        """NONE prediction has max distance"""
        assert edit_distance(['NONE'], ['A', 'B', 'C']) == 1.0


class TestMRR:
    """Test Mean Reciprocal Rank of first correct intermediate node"""
    
    def test_first_intermediate_correct(self):
        """First intermediate node is correct (rank=1)"""
        # Intermediates: pred=['A'], gt=['A','X']
        assert mrr(['Drug', 'A', 'Disease'], ['Drug', 'A', 'X', 'Disease']) == 1.0
    
    def test_second_intermediate_correct(self):
        """Second intermediate is first correct (rank=2)"""
        # Intermediates: pred=['X','A'], gt=['A','B']
        assert mrr(['Drug', 'X', 'A', 'Disease'], ['Drug', 'A', 'B', 'Disease']) == 0.5
    
    def test_no_intermediate_match(self):
        """No intermediate nodes match"""
        # Intermediates: pred=['X','Y'], gt=['A','B']
        assert mrr(['Drug', 'X', 'Y', 'Disease'], ['Drug', 'A', 'B', 'Disease']) == 0.0
    
    def test_too_short_no_intermediates(self):
        """Path has only source and target (no intermediates)"""
        assert mrr(['Drug', 'Disease'], ['Drug', 'A', 'Disease']) == 0.0
    
    def test_empty_prediction(self):
        """Empty prediction returns 0"""
        assert mrr([], ['Drug', 'A', 'Disease']) == 0.0


class TestHubNodeRatio:
    """Test fraction of nodes that are hubs"""
    
    def test_no_hubs(self):
        """No nodes exceed hub threshold"""
        degree_count = {1: 10, 2: 20, 3: 30}
        assert hub_node_ratio([1, 2, 3], degree_count, hub_threshold=100) == 0.0
    
    def test_all_hubs(self):
        """All nodes are hubs"""
        degree_count = {1: 150, 2: 200, 3: 180}
        assert hub_node_ratio([1, 2, 3], degree_count, hub_threshold=100) == 1.0
    
    def test_half_hubs(self):
        """Half of nodes are hubs"""
        degree_count = {1: 50, 2: 150, 3: 75, 4: 200}
        assert hub_node_ratio([1, 2, 3, 4], degree_count, hub_threshold=100) == 0.5
    
    def test_empty_path(self):
        """Empty path returns 0"""
        assert hub_node_ratio([], {1: 100}, hub_threshold=50) == 0.0
    
    def test_node_not_in_degree_count(self):
        """Missing nodes treated as degree 0"""
        degree_count = {1: 150}
        # Node 999 not in degree_count, treated as degree 0 < threshold
        assert hub_node_ratio([1, 999], degree_count, hub_threshold=100) == 0.5


class TestPathLengthAccuracy:
    """Test path length accuracy: 1 - |pred-gt|/max(pred,gt)"""
    
    def test_exact_match(self):
        """Identical lengths"""
        assert path_length_accuracy(5, 5) == 1.0
    
    def test_off_by_one(self):
        """One node difference"""
        # |5-4|/5 = 1/5 = 0.2, accuracy = 1-0.2 = 0.8
        assert path_length_accuracy(5, 4) == 0.8
    
    def test_very_different(self):
        """Large difference"""
        # |10-2|/10 = 8/10 = 0.8, accuracy = 1-0.8 = 0.2
        assert pytest.approx(path_length_accuracy(10, 2), rel=0.01) == 0.2
    
    def test_both_zero(self):
        """Both lengths are zero"""
        assert path_length_accuracy(0, 0) == 1.0
    
    def test_one_zero(self):
        """One length is zero"""
        # |0-5|/5 = 1, accuracy = 1-1 = 0
        assert path_length_accuracy(0, 5) == 0.0


class TestEvaluatePathway:
    """Test combined evaluation function"""
    
    def test_returns_all_metrics(self):
        """Returns all 7 metrics"""
        degree_count = {0: 10, 1: 20, 2: 30}
        results = evaluate_pathway(
            predicted_ids=['Drug', 'A', 'Disease'],
            predicted_indices=[0, 1, 2],
            predicted_length=3,
            ground_truth_ids=['Drug', 'A', 'Disease'],
            ground_truth_length=3,
            degree_count=degree_count,
            hub_threshold=100
        )
        
        expected_keys = ['precision', 'recall', 'f1_score', 'edit_distance',
                        'mrr', 'hub_node_ratio', 'path_length_accuracy']
        assert set(results.keys()) == set(expected_keys)
    
    def test_perfect_prediction(self):
        """Perfect match on all metrics"""
        degree_count = {0: 10, 1: 20, 2: 30}
        results = evaluate_pathway(
            predicted_ids=['Drug', 'A', 'Disease'],
            predicted_indices=[0, 1, 2],
            predicted_length=3,
            ground_truth_ids=['Drug', 'A', 'Disease'],
            ground_truth_length=3,
            degree_count=degree_count,
            hub_threshold=100
        )
        
        assert results['precision'] == 1.0
        assert results['recall'] == 1.0
        assert results['f1_score'] == 1.0
        assert results['edit_distance'] == 0.0
        assert results['mrr'] == 1.0
        assert results['hub_node_ratio'] == 0.0
        assert results['path_length_accuracy'] == 1.0
    
    def test_no_overlap(self):
        """No overlap between predicted and GT"""
        degree_count = {3: 10, 4: 20, 5: 30}
        results = evaluate_pathway(
            predicted_ids=['Drug2', 'X', 'Disease2'],
            predicted_indices=[3, 4, 5],
            predicted_length=3,
            ground_truth_ids=['Drug', 'A', 'Disease'],
            ground_truth_length=3,
            degree_count=degree_count,
            hub_threshold=100
        )
        
        assert results['precision'] == 0.0
        assert results['recall'] == 0.0
        assert results['f1_score'] == 0.0
        assert results['edit_distance'] == 1.0
        assert results['mrr'] == 0.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
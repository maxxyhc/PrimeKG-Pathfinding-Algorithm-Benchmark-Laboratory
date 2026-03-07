"""
Tests for evaluation helper functions.
Run with: pytest tests/test_evaluation_helpers.py -v
"""
import pytest
import pandas as pd
import time as time_module
from collections import Counter
from evaluation_helpers import (
    is_valid_prediction,
    compute_degree_counts,
    compute_hub_threshold,
    calculate_relation_accuracy,
    speed
)


class TestIsValidPrediction:
    """Test prediction validation"""
    
    def test_valid_prediction(self):
        """Valid prediction with nodes"""
        assert is_valid_prediction(['A', 'B', 'C']) == True
    
    def test_empty_list(self):
        """Empty list is invalid"""
        assert is_valid_prediction([]) == False
    
    def test_none_prediction(self):
        """['NONE'] is invalid"""
        assert is_valid_prediction(['NONE']) == False
    
    def test_single_node(self):
        """Single node is valid"""
        assert is_valid_prediction(['A']) == True
    
    def test_none_value(self):
        """None value is invalid"""
        assert is_valid_prediction(None) == False


class TestComputeDegreeCounts:
    """Test degree counting from edge dataframe"""
    
    def test_simple_triangle(self):
        """Simple 3-node triangle graph"""
        edges = pd.DataFrame({
            'x_index': [1, 2, 3],
            'y_index': [2, 3, 1]
        })
        degree_count = compute_degree_counts(edges)
        
        assert degree_count[1] == 2
        assert degree_count[2] == 2
        assert degree_count[3] == 2
    
    def test_hub_node(self):
        """One hub node connected to many"""
        edges = pd.DataFrame({
            'x_index': [1, 1, 1],
            'y_index': [2, 3, 4]
        })
        degree_count = compute_degree_counts(edges)
        
        assert degree_count[1] == 3  # Hub
        assert degree_count[2] == 1
        assert degree_count[3] == 1
        assert degree_count[4] == 1
    
    def test_empty_edges(self):
        """Empty edge dataframe"""
        edges = pd.DataFrame({
            'x_index': [],
            'y_index': []
        })
        degree_count = compute_degree_counts(edges)
        
        assert len(degree_count) == 0
    
    def test_returns_counter(self):
        """Returns Counter object"""
        edges = pd.DataFrame({
            'x_index': [1],
            'y_index': [2]
        })
        degree_count = compute_degree_counts(edges)
        
        assert isinstance(degree_count, Counter)


class TestComputeHubThreshold:
    """Test hub threshold calculation"""
    
    def test_95th_percentile_default(self):
        """Default 95th percentile"""
        degree_count = Counter({
            1: 10, 2: 20, 3: 30, 4: 40, 5: 50,
            6: 60, 7: 70, 8: 80, 9: 90, 10: 100
        })
        threshold = compute_hub_threshold(degree_count)
        
        # 95th percentile of [10,20,30,40,50,60,70,80,90,100] = 95
        assert threshold >= 90
        assert threshold <= 100
    
    def test_custom_percentile(self):
        """Custom percentile value"""
        degree_count = Counter({1: 10, 2: 20, 3: 30, 4: 40})
        threshold = compute_hub_threshold(degree_count, percentile=50)
        
        # 50th percentile (median) of [10,20,30,40] = 25
        assert threshold == 25.0
    
    def test_100th_percentile(self):
        """100th percentile returns max"""
        degree_count = Counter({1: 10, 2: 50, 3: 100})
        threshold = compute_hub_threshold(degree_count, percentile=100)
        
        assert threshold == 100.0
    
    def test_0th_percentile(self):
        """0th percentile returns min"""
        degree_count = Counter({1: 10, 2: 50, 3: 100})
        threshold = compute_hub_threshold(degree_count, percentile=0)
        
        assert threshold == 10.0


class TestCalculateRelationAccuracy:
    """Test relation type accuracy"""
    
    def test_perfect_match(self):
        """All predicted relations are in GT"""
        pred_relations = ['binds', 'regulates']
        gt_edge_types = ['binds', 'regulates', 'treats']
        
        assert calculate_relation_accuracy(pred_relations, gt_edge_types) == 1.0
    
    def test_no_match(self):
        """No predicted relations are in GT"""
        pred_relations = ['X', 'Y']
        gt_edge_types = ['A', 'B']
        
        assert calculate_relation_accuracy(pred_relations, gt_edge_types) == 0.0
    
    def test_partial_match(self):
        """Half of predicted relations match"""
        pred_relations = ['binds', 'X', 'regulates', 'Y']
        gt_edge_types = ['binds', 'regulates', 'treats']
        
        assert calculate_relation_accuracy(pred_relations, gt_edge_types) == 0.5
    
    def test_empty_prediction(self):
        """Empty prediction returns 0"""
        pred_relations = []
        gt_edge_types = ['binds', 'regulates']
        
        assert calculate_relation_accuracy(pred_relations, gt_edge_types) == 0.0
    
    def test_duplicate_relations(self):
        """Duplicate relations counted separately"""
        pred_relations = ['binds', 'binds', 'binds']
        gt_edge_types = ['binds', 'regulates']
        
        # All 3 'binds' match
        assert calculate_relation_accuracy(pred_relations, gt_edge_types) == 1.0


class TestSpeed:
    """Test timing function"""
    
    def test_returns_result_and_time(self):
        """Returns both result and elapsed time"""
        def sample_func(x):
            return x * 2
        
        result, elapsed = speed(sample_func, 5)
        
        assert result == 10
        assert isinstance(elapsed, float)
        assert elapsed >= 0
    
    def test_measures_time_correctly(self):
        """Measures time for sleep"""
        def slow_func():
            time_module.sleep(0.01)  # 10ms sleep
            return 42
        
        result, elapsed = speed(slow_func)
        
        assert result == 42
        assert elapsed >= 10  # At least 10ms
        assert elapsed < 100  # But not crazy long
    
    def test_forwards_args(self):
        """Forwards arguments correctly"""
        def add(a, b):
            return a + b
        
        result, elapsed = speed(add, 3, 5)
        
        assert result == 8
    
    def test_forwards_kwargs(self):
        """Forwards keyword arguments correctly"""
        def greet(name, greeting="Hello"):
            return f"{greeting}, {name}"
        
        result, elapsed = speed(greet, "Alice", greeting="Hi")
        
        assert result == "Hi, Alice"
    
    def test_very_fast_function(self):
        """Handles very fast functions"""
        def instant():
            return 1
        
        result, elapsed = speed(instant)
        
        assert result == 1
        assert elapsed >= 0
        assert elapsed < 1  # Less than 1ms


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
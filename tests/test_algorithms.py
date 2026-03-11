import pytest
import networkx as nx
import numpy as np
from Algorithms import (
    allowed_transition,
    find_path_engine,
    MetaPathBFS,
    HubPenalizedShortestPath,
    PageRankInverseShortestPath,
    SemanticBridgingPath,
    BidirectionalSearch,
    BidirectionalKShortestBio,
    BidirectionalRelationWeighted
)


@pytest.fixture
def test_graph():
    """
    Create a simple test graph for algorithm testing.
    
    Structure:
        Drug(0) -> Protein(1) -> Protein(2) -> Disease(3)
                -> Protein(4) -> Protein(5) -> Disease(3)
        Drug(6) -> Drug(7) -> Disease(3)  [should be blocked]
    """
    G = nx.DiGraph()
    
    # Add nodes with types
    G.add_node(0, node_type='drug', node_name='Drug_A')
    G.add_node(1, node_type='protein', node_name='Protein_B')
    G.add_node(2, node_type='protein', node_name='Protein_C')
    G.add_node(3, node_type='disease', node_name='Disease_D')
    G.add_node(4, node_type='protein', node_name='Protein_E')
    G.add_node(5, node_type='protein', node_name='Protein_F')
    G.add_node(6, node_type='drug', node_name='Drug_G')
    G.add_node(7, node_type='drug', node_name='Drug_H')
    
    # Add edges with relations and weights
    edges = [
        (0, 1, {'relation': 'drug_protein', 'weight': 1.0}),
        (1, 2, {'relation': 'protein_protein', 'weight': 1.0}),
        (2, 3, {'relation': 'disease_protein', 'weight': 1.0}),
        (0, 4, {'relation': 'drug_protein', 'weight': 1.5}),
        (4, 5, {'relation': 'protein_protein', 'weight': 1.0}),
        (5, 3, {'relation': 'disease_protein', 'weight': 1.0}),
        (6, 7, {'relation': 'drug_drug', 'weight': 1.0}),
        (7, 3, {'relation': 'indication', 'weight': 1.0}),
    ]
    
    for u, v, data in edges:
        G.add_edge(u, v, **data)
    
    return G


# ============================================================
# Test allowed_transition constraint function
# ============================================================

def test_drug_to_disease_blocked(test_graph):
    """Drug -> Disease should be blocked."""
    assert not allowed_transition(test_graph, 0, 0, 3)


def test_drug_to_protein_allowed(test_graph):
    """Drug -> Protein should be allowed."""
    assert allowed_transition(test_graph, 0, 0, 1)


def test_protein_to_disease_allowed(test_graph):
    """Protein -> Disease should be allowed."""
    assert allowed_transition(test_graph, 0, 2, 3)


def test_drug_to_drug_from_drug_source_blocked(test_graph):
    """Drug -> Drug when source is drug should be blocked."""
    assert not allowed_transition(test_graph, 6, 6, 7)


def test_drug_to_drug_from_protein_source_allowed(test_graph):
    """Drug -> Drug when source is not drug should be allowed."""
    assert allowed_transition(test_graph, 1, 6, 7)


# ============================================================
# Test find_path_engine (Dijkstra)
# ============================================================

def test_find_path_engine_simple_path(test_graph):
    """Test finding a simple valid path."""
    path, relations, cost = find_path_engine(test_graph, test_graph, 0, 3, allowed_transition)
    
    assert path[0] == 0
    assert path[-1] == 3
    assert len(path) == 4
    assert cost == pytest.approx(3.0)


def test_find_path_engine_no_path(test_graph):
    """Test when no path exists."""
    path, relations, cost = find_path_engine(test_graph, test_graph, 3, 0, allowed_transition)
    
    assert path == []
    assert relations == []
    assert cost == float('inf')


def test_find_path_engine_same_source_target(test_graph):
    """Test when source equals target."""
    path, relations, cost = find_path_engine(test_graph, test_graph, 0, 0, allowed_transition)
    assert 0 in path


# ============================================================
# Test MetaPathBFS
# ============================================================

def test_metapath_initialization_default(test_graph):
    """Test initialization with default meta-paths."""
    algo = MetaPathBFS(test_graph)
    
    assert algo.valid_metapaths is not None
    assert len(algo.valid_metapaths) > 0
    assert algo.max_length == 10


def test_metapath_initialization_custom(test_graph):
    """Test initialization with custom meta-paths."""
    custom_paths = [
        ['drug_protein', 'disease_protein'],
        ['drug_protein', 'protein_protein', 'disease_protein']
    ]
    algo = MetaPathBFS(test_graph, valid_metapaths=custom_paths, max_length=5)
    
    assert algo.valid_metapaths == custom_paths
    assert algo.max_length == 5


def test_metapath_find_path(test_graph):
    """Test finding path that matches valid meta-path."""
    algo = MetaPathBFS(test_graph)
    path, relations, cost = algo.find_path(0, 3)
    
    if len(path) > 0:
        assert path[0] == 0
        assert path[-1] == 3
        assert algo._is_valid_metapath(relations)


def test_metapath_same_source_target(test_graph):
    """Test when source equals target."""
    algo = MetaPathBFS(test_graph)
    path, relations, cost = algo.find_path(0, 0)
    
    assert path == [0]
    assert relations == []
    assert cost == pytest.approx(0.0)


def test_metapath_no_valid_path(test_graph):
    """Test when no valid meta-path exists."""
    custom_paths = [['nonexistent_relation']]
    algo = MetaPathBFS(test_graph, valid_metapaths=custom_paths)
    path, relations, cost = algo.find_path(0, 3)
    
    assert path == []
    assert cost == float('inf')


def test_metapath_is_valid(test_graph):
    """Test meta-path validation."""
    algo = MetaPathBFS(test_graph)
    
    assert algo._is_valid_metapath(['drug_protein', 'disease_protein'])
    assert not algo._is_valid_metapath(['invalid', 'path'])


def test_metapath_could_match(test_graph):
    """Test partial meta-path matching."""
    algo = MetaPathBFS(test_graph)
    
    assert algo._could_match_metapath(['drug_protein'])
    assert not algo._could_match_metapath(['invalid'])


# ============================================================
# Test HubPenalizedShortestPath
# ============================================================

def test_hub_penalized_initialization(test_graph):
    """Test algorithm initializes correctly."""
    algo = HubPenalizedShortestPath(test_graph, alpha=0.5)
    
    assert algo.graph is not None
    assert algo.weighted_graph is not None
    assert algo.alpha == 0.5


def test_hub_penalized_find_path(test_graph):
    """Test finding a path with hub penalties."""
    algo = HubPenalizedShortestPath(test_graph, alpha=0.5)
    path, relations, cost = algo.find_path(0, 3)
    
    assert path[0] == 0
    assert path[-1] == 3
    assert len(path) > 0
    assert cost < float('inf')


def test_hub_penalized_has_weights(test_graph):
    """Test that weighted graph has weight attributes."""
    algo = HubPenalizedShortestPath(test_graph, alpha=0.5)
    
    for u, v in algo.weighted_graph.edges():
        assert 'weight' in algo.weighted_graph[u][v]
        assert algo.weighted_graph[u][v]['weight'] > 0


# ============================================================
# Test PageRankInverseShortestPath
# ============================================================

def test_pagerank_initialization_no_precomputed(test_graph):
    """Test initialization without precomputed PageRank."""
    algo = PageRankInverseShortestPath(test_graph)
    
    assert algo.pagerank_scores is not None
    assert len(algo.pagerank_scores) > 0


def test_pagerank_initialization_with_precomputed(test_graph):
    """Test initialization with precomputed PageRank."""
    precomputed = {i: 0.125 for i in range(8)}
    algo = PageRankInverseShortestPath(test_graph, precomputed_pagerank=precomputed)
    
    assert algo.pagerank_scores == precomputed


def test_pagerank_find_path(test_graph):
    """Test finding a path with PageRank weighting."""
    algo = PageRankInverseShortestPath(test_graph)
    path, relations, cost = algo.find_path(0, 3)
    
    assert path[0] == 0
    assert path[-1] == 3
    assert len(path) > 0


# ============================================================
# Test SemanticBridgingPath
# ============================================================

def test_semantic_initialization(test_graph):
    """Test algorithm initializes correctly."""
    algo = SemanticBridgingPath(test_graph, beta=0.3)
    
    assert algo.beta == 0.3
    assert algo.embeddings is None
    assert algo.weighted_graph is None


def test_semantic_compute_embeddings(test_graph):
    """Test embedding computation."""
    algo = SemanticBridgingPath(test_graph, beta=0.3)
    embeddings = algo.compute_embeddings()
    
    assert embeddings is not None
    assert len(embeddings) == 8
    
    for node, emb in embeddings.items():
        assert isinstance(emb, np.ndarray)
        assert len(emb) > 0


def test_semantic_find_path(test_graph):
    """Test finding a semantically coherent path."""
    algo = SemanticBridgingPath(test_graph, beta=0.3)
    path, relations, cost = algo.find_path(0, 3)
    
    assert path[0] == 0
    assert path[-1] == 3
    assert len(path) > 0


# ============================================================
# Test BidirectionalSearch
# ============================================================

def test_bidirectional_initialization(test_graph):
    """Test algorithm initializes correctly."""
    algo = BidirectionalSearch(test_graph, max_depth=8, max_explore=50000)
    
    assert algo.max_depth == 8
    assert algo.max_explore == 50000


def test_bidirectional_find_path(test_graph):
    """Test finding a simple path."""
    algo = BidirectionalSearch(test_graph)
    path, relations, cost = algo.find_path(0, 3)
    
    assert path[0] == 0
    assert path[-1] == 3
    assert len(path) == 4


def test_bidirectional_same_source_target(test_graph):
    """Test when source equals target."""
    algo = BidirectionalSearch(test_graph)
    path, relations, cost = algo.find_path(0, 0)
    
    assert path == [0]
    assert relations == []
    assert cost == pytest.approx(0.0)


def test_bidirectional_no_path(test_graph):
    """Test when no path exists."""
    algo = BidirectionalSearch(test_graph)
    path, relations, cost = algo.find_path(3, 0)
    
    assert path == []
    assert cost == float('inf')


def test_bidirectional_respects_constraints(test_graph):
    """Test that algorithm respects allowed_transition."""
    algo = BidirectionalSearch(test_graph)
    path, relations, cost = algo.find_path(6, 3)
    
    if len(path) > 0:
        for i in range(len(path) - 1):
            assert allowed_transition(test_graph, 6, path[i], path[i+1])


# ============================================================
# Test BidirectionalKShortestBio
# ============================================================

def test_k_shortest_initialization(test_graph):
    """Test algorithm initializes correctly."""
    algo = BidirectionalKShortestBio(test_graph, k=4)
    
    assert algo.k == 4
    assert algo.max_depth == 8
    assert algo.degrees is not None
    assert algo.degree_p90 > 0


def test_k_shortest_find_path(test_graph):
    """Test finding best-scored path from k alternatives."""
    algo = BidirectionalKShortestBio(test_graph, k=3)
    path, relations, cost = algo.find_path(0, 3)
    
    assert path[0] == 0
    assert path[-1] == 3
    assert len(path) > 0


def test_k_shortest_score_path(test_graph):
    """Test path scoring function."""
    algo = BidirectionalKShortestBio(test_graph, k=4)
    test_path = [0, 1, 2, 3]
    
    score = algo._score_path(test_path)
    assert isinstance(score, float)


def test_k_shortest_no_path(test_graph):
    """Test when no path exists."""
    algo = BidirectionalKShortestBio(test_graph, k=3)
    path, relations, cost = algo.find_path(3, 0)
    
    assert path == []
    assert cost == float('inf')


def test_k_shortest_finds_multiple(test_graph):
    """Test that algorithm attempts to find k paths."""
    algo = BidirectionalKShortestBio(test_graph, k=2)
    candidates = algo._find_k_paths(0, 3)
    
    assert len(candidates) > 0
    assert len(candidates) <= 2


# ============================================================
# Test BidirectionalRelationWeighted
# ============================================================

def test_relation_weighted_initialization(test_graph):
    """Test algorithm initializes with hardcoded weights."""
    algo = BidirectionalRelationWeighted(test_graph)
    
    assert algo.relation_weights is not None
    assert 'drug_protein' in algo.relation_weights
    assert 'disease_protein' in algo.relation_weights
    assert algo.relation_weights['drug_protein'] == 0.100


def test_relation_weighted_get_weight(test_graph):
    """Test edge weight retrieval."""
    algo = BidirectionalRelationWeighted(test_graph)
    
    weight = algo._get_weight(0, 1)
    assert weight == pytest.approx(0.100)  # drug_protein
    
    weight = algo._get_weight(1, 2)
    assert weight == pytest.approx(0.227)  # protein_protein


def test_relation_weighted_find_path(test_graph):
    """Test finding weighted path."""
    algo = BidirectionalRelationWeighted(test_graph)
    path, relations, cost = algo.find_path(0, 3)
    
    assert path[0] == 0
    assert path[-1] == 3
    assert len(path) > 0
    assert cost < float('inf')


def test_relation_weighted_prefers_low_weights(test_graph):
    """Test that algorithm prefers edges with lower weights."""
    algo = BidirectionalRelationWeighted(test_graph)
    path, relations, cost = algo.find_path(0, 3)
    
    # Path should prefer drug_protein and disease_protein (weight 0.1)
    assert 'drug_protein' in relations or 'disease_protein' in relations


def test_relation_weighted_same_source_target(test_graph):
    """Test when source equals target."""
    algo = BidirectionalRelationWeighted(test_graph)
    path, relations, cost = algo.find_path(0, 0)
    
    assert path == [0]
    assert relations == []
    assert cost == pytest.approx(0.0)


def test_relation_weighted_no_path(test_graph):
    """Test when no path exists."""
    algo = BidirectionalRelationWeighted(test_graph)
    path, relations, cost = algo.find_path(3, 0)
    
    assert path == []
    assert cost == float('inf')


# ============================================================
# Test algorithm consistency
# ============================================================

@pytest.fixture
def all_algorithms(test_graph):
    """Fixture providing all algorithm instances."""
    return [
        MetaPathBFS(test_graph),
        HubPenalizedShortestPath(test_graph),
        PageRankInverseShortestPath(test_graph),
        SemanticBridgingPath(test_graph),
        BidirectionalSearch(test_graph),
        BidirectionalKShortestBio(test_graph),
        BidirectionalRelationWeighted(test_graph),
    ]


def test_all_return_same_format(all_algorithms):
    """Test that all algorithms return (path, relations, cost) format."""
    for algo in all_algorithms:
        result = algo.find_path(0, 3)
        
        assert isinstance(result, tuple)
        assert len(result) == 3
        
        path, relations, cost = result
        assert isinstance(path, list)
        assert isinstance(relations, list)
        assert isinstance(cost, (int, float))


def test_all_find_valid_paths(all_algorithms):
    """Test that all algorithms find valid paths."""
    for algo in all_algorithms:
        path, relations, cost = algo.find_path(0, 3)
        
        if len(path) > 0:
            assert path[0] == 0
            assert path[-1] == 3
            assert len(relations) == len(path) - 1


def test_all_respect_constraints(all_algorithms, test_graph):
    """Test that all algorithms respect allowed_transition."""
    for algo in all_algorithms:
        path, relations, cost = algo.find_path(0, 3)
        
        # Verify no drug->disease transitions
        for i in range(len(path) - 1):
            u, v = path[i], path[i+1]
            u_type = test_graph.nodes[u].get('node_type', '')
            v_type = test_graph.nodes[v].get('node_type', '')
            
            assert not (u_type == 'drug' and v_type == 'disease'), \
                f"{algo.__class__.__name__} violated drug->disease constraint"
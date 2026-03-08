import heapq
import networkx as nx
import numpy as np
from typing import Dict, List, Tuple
from scipy.sparse.linalg import eigsh
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler



def allowed_transition(G, src, u, v) -> bool:
    """
    Return False if the step u->v would create:
      (1) drug -> disease
      (2) drug -> drug -> disease   (only when starting node is drug and we're at 2nd hop)
    Assumes node attributes: node_type ('drug','disease',...)
    """
    u_type = G.nodes[u].get("node_type", "")
    v_type = G.nodes[v].get("node_type", "")

    # (1) ban drug -> disease (anywhere)
    if u_type == "drug" and v_type == "disease":
        return False

    # (2) ban drug -> drug when the path starts from a drug (prevents drug->drug->disease)
    # You can enforce only at hop-1 from the source:
    src_type = G.nodes[src].get("node_type", "")
    if src_type == "drug" and u == src and v_type == "drug":
        return False

    return True

def find_path_engine(graph: nx.DiGraph, weighted_graph: nx.DiGraph, source: int, target: int, allowed_transition_fn ) -> Tuple[List[int], List[str], float]:
    """
    Generic constrained shortest-path engine (Dijkstra).

    Parameters
    ----------
    graph : nx.DiGraph
        Original graph (used for node_type, relation lookup).
    weighted_graph : nx.DiGraph
        Graph with edge attribute 'weight'.
    source : int
    target : int
    allowed_transition_fn : function(G, source, u, v) -> bool

    Returns
    -------
    path_nodes : List[int]
    relations  : List[str]
    total_cost : float
    """

    dist = {source: 0.0}
    parent = {source: None}
    pq = [(0.0, source)]

    while pq:
        cur_cost, u = heapq.heappop(pq)

        if cur_cost != dist.get(u, float('inf')):
            continue

        if u == target:
            break

        for v in graph.successors(u):
            # ✅ enforce global bans
            if not allowed_transition_fn(graph, source, u, v):
                continue

            w = weighted_graph[u][v].get("weight", 1.0)
            new_cost = cur_cost + w

            if new_cost < dist.get(v, float('inf')):
                dist[v] = new_cost
                parent[v] = u
                heapq.heappush(pq, (new_cost, v))

    if target not in dist:
        return [], [], float("inf")

    # --- reconstruct path ---
    path = []
    cur = target
    while cur is not None:
        path.append(cur)
        cur = parent[cur]
    path.reverse()

    # --- reconstruct relations ---
    relations = []
    for i in range(len(path) - 1):
        edge_data = graph.get_edge_data(path[i], path[i + 1]) or {}
        relations.append(edge_data.get("relation", "unknown"))

    return path, relations, dist[target]


# ============================================================
# ALGORITHM 1: Dijkstra Implementation
# ============================================================
# ============================================================
# ALGORITHM 1B: Dijkstra Baseline (Class Wrapper)
# ============================================================

class DijkstraShortestPath:
    """
    Baseline Dijkstra shortest path.
    Wrapper for find_path_engine with unweighted edges (all weights = 1.0).
    """
    
    def __init__(self, graph: nx.DiGraph):
        self.graph = graph
        # Create unweighted graph - all edges weight 1.0
        self.weighted_graph = graph.copy()
        for u, v in self.weighted_graph.edges():
            self.weighted_graph[u][v]['weight'] = 1.0
    
    def find_path(self, source: int, target: int) -> Tuple[List[int], List[str], float]:
        """Find shortest path using Dijkstra."""
        return find_path_engine(
            self.graph,
            self.weighted_graph,
            source,
            target,
            allowed_transition
        )

# ============================================================
# ALGORITHM 2: Meta-Path Constrained BFS
# ============================================================

class MetaPathBFS:
    """
    BFS that only explores paths matching predefined meta-path patterns.
    Enforces specific edge type sequences (e.g., drug_protein → protein_protein → disease_protein).
    """
    
    def __init__(self, graph, valid_metapaths=None, max_length=10):
        self.graph = graph
        self.max_length = max_length
        
        if valid_metapaths is None:
            self.valid_metapaths = [
                ['drug_protein', 'disease_protein'],
                ['drug_protein', 'protein_protein', 'disease_protein'],
                ['drug_protein', 'protein_protein', 'protein_protein', 'disease_protein'],
                ['drug_protein', 'pathway_protein', 'disease_protein'],
                ['drug_protein', 'pathway_protein', 'pathway_protein', 'disease_protein'],
                ['drug_protein', 'pathway_protein', 'pathway_pathway', 'pathway_protein', 'disease_protein'],
                ['drug_protein', 'anatomy_protein_present', 'anatomy_protein_present', 'disease_protein'],
                ['drug_protein', 'protein_protein', 'pathway_protein', 'disease_protein'],
                ['drug_protein', 'pathway_protein', 'pathway_protein', 'pathway_protein', 'disease_protein'],
            ]
        else:
            self.valid_metapaths = valid_metapaths
    
    def _is_valid_metapath(self, relations):
        """Check if a relation sequence matches any valid meta-path pattern."""
        return relations in self.valid_metapaths
    
    def _could_match_metapath(self, relations):
        """Check if the current relation sequence could potentially lead to a valid path."""
        for pattern in self.valid_metapaths:
            if len(relations) <= len(pattern):
                if relations == pattern[:len(relations)]:
                    return True
        return False
    
    def find_path(self, source, target):
        """
        Find shortest path that follows valid meta-path patterns using BFS.
        
        Returns
        -------
        path : List[int]
        relations : List[str]
        cost : float
        """
        from collections import deque
        
        if source == target:
            return [source], [], 0.0
        
        queue = deque([(source, [source], [])])
        visited = {source: []}
        
        while queue:
            current, path, relations = queue.popleft()
            
            if current == target:
                if self._is_valid_metapath(relations):
                    return path, relations, float(len(path) - 1)
            
            if len(path) >= self.max_length:
                continue
            
            for neighbor in self.graph.successors(current):
                edge_data = self.graph.get_edge_data(current, neighbor) or {}
                new_relation = edge_data.get('relation', 'unknown')
                new_relations = relations + [new_relation]
                
                if self._could_match_metapath(new_relations):
                    state_key = (neighbor, tuple(new_relations))
                    
                    if neighbor not in visited or visited[neighbor] != new_relations:
                        visited[neighbor] = new_relations
                        queue.append((neighbor, path + [neighbor], new_relations))
        
        return [], [], float('inf')
    
# ============================================================
# ALGORITHM 3: Hub-Penalized Weighted Shortest Path
# ============================================================


class HubPenalizedShortestPath:
    """
    Weighted Dijkstra that penalizes high-degree (hub) nodes.
    Weight formula: weight[u,v] = 1 + α * log(degree[v])
    """
    
    def __init__(self, graph: nx.DiGraph, alpha: float = 0.5):
        self.graph = graph
        self.alpha = alpha
        # self.allowed_metapaths = VALID_METAPATHS
        self.weighted_graph = self._compute_weights()
    
    def _compute_weights(self) -> nx.DiGraph:
        G_weighted = self.graph.copy()
        degrees = dict(G_weighted.degree())
        
        for u, v in G_weighted.edges():
            target_degree = degrees.get(v, 1)
            weight = 1.0 + self.alpha * np.log(max(target_degree, 1))
            G_weighted[u][v]['weight'] = weight
        
        return G_weighted
    
    def find_path(self, source: int, target: int):
        return find_path_engine(
            self.graph,
            self.weighted_graph,
            source,
            target,
            allowed_transition
        )

# ============================================================
# ALGORITHM 4: PageRank-Inverse Weighted Shortest Path
# ============================================================

class PageRankInverseShortestPath:
    """
    Weighted Dijkstra using inverse PageRank.
    Weight formula: weight[u,v] = 1 / (1 + pagerank[v])
    Enforces shortcut bans via allowed_transition().
    """

    def __init__(self, graph: nx.DiGraph, damping: float = 0.85,
                 precomputed_pagerank: Dict[int, float] = None):
        self.graph = graph

        if precomputed_pagerank is not None:
            self.pagerank_scores = precomputed_pagerank
        else:
            print("  Computing PageRank (this may take a minute)...")
            self.pagerank_scores = nx.pagerank(graph, alpha=damping)
            print(f"  PageRank computed for {len(self.pagerank_scores):,} nodes")

        self.weighted_graph = self._compute_weights()

    def _compute_weights(self) -> nx.DiGraph:
        G_weighted = self.graph.copy()

        max_pr = max(self.pagerank_scores.values())
        min_pr = min(self.pagerank_scores.values())
        pr_range = max_pr - min_pr if max_pr > min_pr else 1.0

        for u, v in G_weighted.edges():
            normalized_pr = (self.pagerank_scores.get(v, 0) - min_pr) / pr_range
            weight = 1.0 / (1.0 + normalized_pr)
            G_weighted[u][v]['weight'] = weight

        return G_weighted

    def find_path(self, source: int, target: int) -> Tuple[List[int], List[str], float]:
        return find_path_engine(
            self.graph,
            self.weighted_graph,
            source,
            target,
            allowed_transition
        )

    

# ============================================================
# ALGORITHM 5: Semantic Bridging with Intermediate Node Scoring
# ============================================================

class SemanticBridgingPath:
    """
    Pathfinding weighted by semantic coherence.
    Weight formula: weight[u,v] = 1 - β * cosine_sim(embedding[u], embedding[v])
    """
    
    def __init__(self, graph: nx.DiGraph, beta: float = 0.3):
        self.graph = graph
        self.beta = beta
        self.embeddings = None
        self.weighted_graph = None
        
        self.descriptions = {
            n: graph.nodes[n].get('node_name', str(n)) 
            for n in graph.nodes()
        }
    
    def compute_embeddings(self) -> Dict[int, np.ndarray]:
        """Compute TF-IDF embeddings for node descriptions."""
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.decomposition import TruncatedSVD
        
        print("  Computing TF-IDF embeddings...")
        
        nodes = list(self.graph.nodes())
        texts = [self.descriptions[n] for n in nodes]
        
        vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(texts)
        
        n_components = min(64, tfidf_matrix.shape[1] - 1)
        svd = TruncatedSVD(n_components=n_components, random_state=42)
        embeddings_matrix = svd.fit_transform(tfidf_matrix)
        
        self.embeddings = {node: embeddings_matrix[i] for i, node in enumerate(nodes)}
        print(f"  Embeddings computed for {len(self.embeddings):,} nodes")
        
        return self.embeddings
    
    def _cosine_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        norm1, norm2 = np.linalg.norm(emb1), np.linalg.norm(emb2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return np.dot(emb1, emb2) / (norm1 * norm2)
    
    def compute_edge_weights(self) -> nx.DiGraph:
        """Compute semantic similarity-based edge weights."""
        if self.embeddings is None:
            self.compute_embeddings()
        
        print("  Computing edge weights...")
        self.weighted_graph = self.graph.copy()
        
        for u, v in self.weighted_graph.edges():
            emb_u = self.embeddings.get(u)
            emb_v = self.embeddings.get(v)
            
            if emb_u is not None and emb_v is not None:
                sim = self._cosine_similarity(emb_u, emb_v)
                weight = 1.0 - self.beta * max(0, sim)
            else:
                weight = 1.0
            
            self.weighted_graph[u][v]['weight'] = weight
        
        print(f"  Edge weights computed for {self.weighted_graph.number_of_edges():,} edges")
        return self.weighted_graph
    
    def find_path(self, source: int, target: int) -> Tuple[List[int], List[str], float]:
        """Find semantically coherent path."""
        if self.weighted_graph is None:
            self.compute_edge_weights()
        return find_path_engine(
            self.graph,
            self.weighted_graph,
            source,
            target,
            allowed_transition
        )


# ============================================================
# ALGORITHM 6: Bidirectional Shortest Path
# ============================================================
    
class BidirectionalSearch:
    """
    Bidirectional unweighted shortest path search.
    Searches simultaneously from source (forward) and target (backward)
    until the searches meet, reducing exploration compared to forward-only search.
    """
    def __init__(self, graph, max_depth=8, max_explore=50000):
        self.graph = graph
        self.max_depth = max_depth
        self.max_explore = max_explore

    def find_path(self, source, target):
        if source == target:
            return [source], [], 0.0

        # Forward state
        dist_f = {source: 0.0}
        parent_f = {source: None}
        pq_f = [(0.0, source)]
        visited_f = set()

        # Backward state
        dist_b = {target: 0.0}
        parent_b = {target: None}
        pq_b = [(0.0, target)]
        visited_b = set()

        best_cost = float('inf')
        meeting_node = None
        explored = 0

        while pq_f or pq_b:
            explored += 1
            if explored > self.max_explore:
                break

            # Forward step
            if pq_f:
                cost_f, u_f = heapq.heappop(pq_f)
                if cost_f > self.max_depth:
                    pq_f = []
                elif cost_f == dist_f.get(u_f, float('inf')) and cost_f <= best_cost:
                    visited_f.add(u_f)
                    for v in self.graph.successors(u_f):
                        if not allowed_transition(self.graph, source, u_f, v):
                            continue
                        new_cost = cost_f + 1.0
                        if new_cost < dist_f.get(v, float('inf')):
                            dist_f[v] = new_cost
                            parent_f[v] = u_f
                            heapq.heappush(pq_f, (new_cost, v))
                            if v in dist_b:
                                total = new_cost + dist_b[v]
                                if total < best_cost:
                                    best_cost = total
                                    meeting_node = v

            # Backward step
            if pq_b:
                cost_b, u_b = heapq.heappop(pq_b)
                if cost_b > self.max_depth:
                    pq_b = []
                elif cost_b == dist_b.get(u_b, float('inf')) and cost_b <= best_cost:
                    visited_b.add(u_b)
                    for v in self.graph.predecessors(u_b):
                        new_cost = cost_b + 1.0
                        if new_cost < dist_b.get(v, float('inf')):
                            dist_b[v] = new_cost
                            parent_b[v] = u_b
                            heapq.heappush(pq_b, (new_cost, v))
                            if v in dist_f:
                                total = dist_f[v] + new_cost
                                if total < best_cost:
                                    best_cost = total
                                    meeting_node = v

            # Early termination
            min_f = pq_f[0][0] if pq_f else float('inf')
            min_b = pq_b[0][0] if pq_b else float('inf')
            if min_f + min_b >= best_cost:
                break

        if meeting_node is None:
            return [], [], float('inf')

        # Reconstruct forward half: source → meeting
        fwd = []
        cur = meeting_node
        while cur is not None:
            fwd.append(cur)
            cur = parent_f.get(cur)
        fwd.reverse()

        # Reconstruct backward half: meeting → target
        bwd = []
        cur = parent_b.get(meeting_node)
        while cur is not None:
            bwd.append(cur)
            cur = parent_b.get(cur)

        full_path = fwd + bwd

        # Validate edges exist and constraints hold
        valid = [full_path[0]]
        for i in range(len(full_path) - 1):
            u, v = full_path[i], full_path[i + 1]
            if self.graph.has_edge(u, v) and allowed_transition(self.graph, source, u, v):
                valid.append(v)
            else:
                # Path broken — fall back to find_path_engine
                return find_path_engine(self.graph, self.graph, source, target, allowed_transition)

        if valid[-1] != target:
            return find_path_engine(self.graph, self.graph, source, target, allowed_transition)

        relations = []
        for i in range(len(valid) - 1):
            ed = self.graph.get_edge_data(valid[i], valid[i + 1]) or {}
            relations.append(ed.get("relation", "unknown"))

        return valid, relations, float(len(valid) - 1)

# ============================================================
# ALGORITHM 7: Bidirectional: K-Shortest Paths with Biological Scoring
# ============================================================

class BidirectionalKShortestBio:
    """
    Finds k diverse paths using Yen's algorithm, then returns the path
    with the highest biological quality score. Scores favor node type diversity,
    high-value relations (drug_protein, disease_protein), and penalize hub nodes.
    """
    def __init__(self, graph, k=4, max_depth=8, max_explore=50000):
        self.graph = graph
        self.k = k
        self.max_depth = max_depth
        self.max_explore = max_explore
        self.degrees = dict(graph.degree())
        degree_vals = list(self.degrees.values())
        self.degree_p90 = np.percentile(degree_vals, 90)
        self.degree_p99 = np.percentile(degree_vals, 99)

    def _bidir_shortest(self, source, target, excluded_edges=None, excluded_nodes=None):
        if excluded_edges is None:
            excluded_edges = set()
        if excluded_nodes is None:
            excluded_nodes = set()

        dist_f = {source: 0.0}
        parent_f = {source: None}
        pq_f = [(0.0, source)]

        dist_b = {target: 0.0}
        parent_b = {target: None}
        pq_b = [(0.0, target)]

        best_cost = float('inf')
        meeting = None
        explored = 0

        while pq_f or pq_b:
            explored += 1
            if explored > self.max_explore:
                break

            if pq_f:
                cf, uf = heapq.heappop(pq_f)
                if cf > self.max_depth:
                    pq_f = []
                elif cf == dist_f.get(uf, float('inf')) and cf <= best_cost:
                    for v in self.graph.successors(uf):
                        if v in excluded_nodes or (uf, v) in excluded_edges:
                            continue
                        if not allowed_transition(self.graph, source, uf, v):
                            continue
                        nc = cf + 1.0
                        if nc < dist_f.get(v, float('inf')):
                            dist_f[v] = nc
                            parent_f[v] = uf
                            heapq.heappush(pq_f, (nc, v))
                            if v in dist_b and nc + dist_b[v] < best_cost:
                                best_cost = nc + dist_b[v]
                                meeting = v

            if pq_b:
                cb, ub = heapq.heappop(pq_b)
                if cb > self.max_depth:
                    pq_b = []
                elif cb == dist_b.get(ub, float('inf')) and cb <= best_cost:
                    for v in self.graph.predecessors(ub):
                        if v in excluded_nodes or (v, ub) in excluded_edges:
                            continue
                        nc = cb + 1.0
                        if nc < dist_b.get(v, float('inf')):
                            dist_b[v] = nc
                            parent_b[v] = ub
                            heapq.heappush(pq_b, (nc, v))
                            if v in dist_f and dist_f[v] + nc < best_cost:
                                best_cost = dist_f[v] + nc
                                meeting = v

            min_f = pq_f[0][0] if pq_f else float('inf')
            min_b = pq_b[0][0] if pq_b else float('inf')
            if min_f + min_b >= best_cost:
                break

        if meeting is None:
            return None, float('inf')

        fwd = []
        cur = meeting
        while cur is not None:
            fwd.append(cur)
            cur = parent_f.get(cur)
        fwd.reverse()

        bwd = []
        cur = parent_b.get(meeting)
        while cur is not None:
            bwd.append(cur)
            cur = parent_b.get(cur)

        path = fwd + bwd

        for i in range(len(path) - 1):
            if not self.graph.has_edge(path[i], path[i + 1]):
                return None, float('inf')
            if not allowed_transition(self.graph, source, path[i], path[i + 1]):
                return None, float('inf')

        if path[-1] != target:
            return None, float('inf')

        return path, float(len(path) - 1)

    def _fwd_shortest(self, source, target, excluded_edges=None, excluded_nodes=None):
        """Forward-only Dijkstra fallback when bidirectional fails."""
        if excluded_edges is None:
            excluded_edges = set()
        if excluded_nodes is None:
            excluded_nodes = set()

        dist = {source: 0.0}
        parent = {source: None}
        pq = [(0.0, source)]

        while pq:
            cost, u = heapq.heappop(pq)
            if cost != dist.get(u, float('inf')):
                continue
            if cost > self.max_depth:
                break
            if u == target:
                break
            for v in self.graph.successors(u):
                if v in excluded_nodes or (u, v) in excluded_edges:
                    continue
                if not allowed_transition(self.graph, source, u, v):
                    continue
                nc = cost + 1.0
                if nc < dist.get(v, float('inf')):
                    dist[v] = nc
                    parent[v] = u
                    heapq.heappush(pq, (nc, v))

        if target not in dist:
            return None, float('inf')

        path = []
        cur = target
        while cur is not None:
            path.append(cur)
            cur = parent[cur]
        path.reverse()
        return path, dist[target]

    def _find_k_paths(self, source, target):
        """Find k diverse paths using Yen's algorithm with bidirectional base search."""
        # Try bidirectional first, fall back to forward
        first_path, first_cost = self._bidir_shortest(source, target)
        if first_path is None:
            first_path, first_cost = self._fwd_shortest(source, target)
        if first_path is None:
            return []

        A = [(first_path, first_cost)]
        B = []
        seen = {tuple(first_path)}
        counter = 0

        for _ in range(1, self.k):
            prev_path = A[-1][0]
            max_spur = min(len(prev_path) - 1, 4)

            for i in range(max_spur):
                spur_node = prev_path[i]
                root_path = prev_path[:i + 1]

                excl_edges = set()
                for (p, _) in A:
                    if len(p) > i and p[:i + 1] == root_path and i + 1 < len(p):
                        excl_edges.add((p[i], p[i + 1]))

                excl_nodes = set(root_path[:-1])

                # Try bidirectional, fall back to forward
                spur_path, _ = self._bidir_shortest(spur_node, target, excl_edges, excl_nodes)
                if spur_path is None:
                    spur_path, _ = self._fwd_shortest(spur_node, target, excl_edges, excl_nodes)

                if spur_path is not None:
                    total_path = root_path[:-1] + spur_path
                    pt = tuple(total_path)
                    if pt not in seen and total_path[-1] == target:
                        seen.add(pt)
                        counter += 1
                        heapq.heappush(B, (float(len(total_path) - 1), counter, total_path))

            if not B:
                break
            cost, _, path = heapq.heappop(B)
            A.append((path, cost))

        return A

    def _score_path(self, path):
        if len(path) < 2:
            return -999
        score = 0.0

        ntypes = set()
        for node in path[1:-1]:
            ntypes.add(self.graph.nodes[node].get('node_type', 'unknown'))
        score += len(ntypes) * 0.4

        rels = []
        for i in range(len(path) - 1):
            ed = self.graph.get_edge_data(path[i], path[i + 1]) or {}
            rels.append(ed.get('relation', 'unknown'))

        if rels:
            if rels[0] == 'drug_protein':
                score += 0.8
            if rels[-1] == 'disease_protein':
                score += 0.8
            for r in rels:
                if r in ('drug_protein', 'disease_protein', 'bioprocess_protein'):
                    score += 0.3
                elif r == 'ppi':
                    score += 0.1

        for node in path[1:-1]:
            deg = self.degrees.get(node, 0)
            if deg > self.degree_p99:
                score -= 1.5
            elif deg > self.degree_p90:
                score -= 0.3

        score -= 0.3 * abs(len(path) - 5)
        return score

    def find_path(self, source, target):
        candidates = self._find_k_paths(source, target)
        if not candidates:
            return [], [], float('inf')

        scored = [(self._score_path(p), p, c) for p, c in candidates]
        scored.sort(key=lambda x: x[0], reverse=True)
        _, best_path, best_cost = scored[0]

        relations = []
        for i in range(len(best_path) - 1):
            ed = self.graph.get_edge_data(best_path[i], best_path[i + 1]) or {}
            relations.append(ed.get("relation", "unknown"))
        return best_path, relations, best_cost


# ============================================================
# ALGORITHM 8: Relation-Weighted Bidirectional Search
# ============================================================

class BidirectionalRelationWeighted:
    """
    Bidirectional search with edge weights based on relation type enrichment
    in ground truth pathways. Biologically meaningful relations (drug_protein,
    disease_protein, bioprocess_protein) have lower costs, guiding search
    toward mechanistically relevant paths.
    """
    def __init__(self, graph, max_depth=8, max_explore=50000):
        self.graph = graph
        self.max_depth = max_depth
        self.max_explore = max_explore
        
        # Computed from ground truth data
        self.relation_weights = {
            'off-label use': 0.184,
            'pathway_pathway': 1.500,
            'drug_protein': 0.100,
            'phenotype_protein': 0.230,
            'exposure_exposure': 1.500,
            'bioprocess_protein': 0.100,
            'disease_phenotype_negative': 1.500,
            'indication': 0.100,
            'drug_effect': 0.255,
            'exposure_disease': 1.500,
            'disease_disease': 1.500,
            'contraindication': 0.733,
            'disease_protein': 0.100,
            'molfunc_protein': 0.308,
            'drug_drug': 0.909,
            'anatomy_protein_absent': 1.500,
            'disease_phenotype_positive': 0.871,
            'anatomy_protein_present': 0.861,
            'protein_protein': 0.227,
            'exposure_protein': 1.500,
            'phenotype_phenotype': 1.500,
            'bioprocess_bioprocess': 0.826,
            'molfunc_molfunc': 1.500,
            'pathway_protein': 0.193,
            'exposure_bioprocess': 1.500,
            'cellcomp_protein': 1.500,
            'cellcomp_cellcomp': 1.500,
            'anatomy_anatomy': 1.500,
        }

    def _get_weight(self, u, v):
        ed = self.graph.get_edge_data(u, v) or {}
        rel = ed.get('relation', 'unknown')
        return self.relation_weights.get(rel, 1.0)

    def _fallback(self, source, target):
        """Forward-only weighted Dijkstra when bidirectional fails."""
        dist = {source: 0.0}
        parent = {source: None}
        pq = [(0.0, source)]

        while pq:
            cost, u = heapq.heappop(pq)
            if cost != dist.get(u, float('inf')):
                continue
            if cost > self.max_depth * 2:
                break
            if u == target:
                break
            for v in self.graph.successors(u):
                if not allowed_transition(self.graph, source, u, v):
                    continue
                w = self._get_weight(u, v)
                nc = cost + w
                if nc < dist.get(v, float('inf')):
                    dist[v] = nc
                    parent[v] = u
                    heapq.heappush(pq, (nc, v))

        if target not in dist:
            return [], [], float('inf')

        # Reconstruct path
        path = []
        cur = target
        while cur is not None:
            path.append(cur)
            cur = parent[cur]
        path.reverse()

        # Extract relations
        relations = []
        for i in range(len(path) - 1):
            ed = self.graph.get_edge_data(path[i], path[i + 1]) or {}
            relations.append(ed.get("relation", "unknown"))
        
        return path, relations, dist[target]

    def find_path(self, source, target):
        if source == target:
            return [source], [], 0.0

        dist_f = {source: 0.0}
        parent_f = {source: None}
        pq_f = [(0.0, source)]

        dist_b = {target: 0.0}
        parent_b = {target: None}
        pq_b = [(0.0, target)]

        best_cost = float('inf')
        meeting = None
        explored = 0

        while pq_f or pq_b:
            explored += 1
            if explored > self.max_explore:
                break

            if pq_f:
                cf, uf = heapq.heappop(pq_f)
                if cf > self.max_depth * 2:
                    pq_f = []
                elif cf == dist_f.get(uf, float('inf')) and cf <= best_cost:
                    for v in self.graph.successors(uf):
                        if not allowed_transition(self.graph, source, uf, v):
                            continue
                        w = self._get_weight(uf, v)
                        nc = cf + w
                        if nc < dist_f.get(v, float('inf')):
                            dist_f[v] = nc
                            parent_f[v] = uf
                            heapq.heappush(pq_f, (nc, v))
                            if v in dist_b and nc + dist_b[v] < best_cost:
                                best_cost = nc + dist_b[v]
                                meeting = v

            if pq_b:
                cb, ub = heapq.heappop(pq_b)
                if cb > self.max_depth * 2:
                    pq_b = []
                elif cb == dist_b.get(ub, float('inf')) and cb <= best_cost:
                    for v in self.graph.predecessors(ub):
                        w = self._get_weight(v, ub)
                        nc = cb + w
                        if nc < dist_b.get(v, float('inf')):
                            dist_b[v] = nc
                            parent_b[v] = ub
                            heapq.heappush(pq_b, (nc, v))
                            if v in dist_f and dist_f[v] + nc < best_cost:
                                best_cost = dist_f[v] + nc
                                meeting = v

            min_f = pq_f[0][0] if pq_f else float('inf')
            min_b = pq_b[0][0] if pq_b else float('inf')
            if min_f + min_b >= best_cost:
                break

        if meeting is None:
            return self._fallback(source, target)

        fwd = []
        cur = meeting
        while cur is not None:
            fwd.append(cur)
            cur = parent_f.get(cur)
        fwd.reverse()

        bwd = []
        cur = parent_b.get(meeting)
        while cur is not None:
            bwd.append(cur)
            cur = parent_b.get(cur)

        path = fwd + bwd

        for i in range(len(path) - 1):
            if not self.graph.has_edge(path[i], path[i + 1]):
                return self._fallback(source, target)
            if not allowed_transition(self.graph, source, path[i], path[i + 1]):
                return self._fallback(source, target)

        if path[-1] != target:
            return self._fallback(source, target)

        relations = []
        for i in range(len(path) - 1):
            ed = self.graph.get_edge_data(path[i], path[i + 1]) or {}
            relations.append(ed.get("relation", "unknown"))

        return path, relations, best_cost
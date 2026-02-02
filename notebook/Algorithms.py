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
# ALGORITHM 2: Hub-Penalized Weighted Shortest Path
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
# ALGORITHM 3: PageRank-Inverse Weighted Shortest Path
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
# ALGORITHM 4: Learned Embeddings + A* with Supervised Edge Weights
# ============================================================



class LearnedEmbeddingsAStar:
    """
    A* search with learned edge weights from embeddings.
    """
    
    def __init__(self, graph: nx.DiGraph, embedding_dim: int = 64):
        self.graph = graph
        self.embedding_dim = embedding_dim
        self.embeddings = None
        self.edge_weights = None
        self.scaler = None
        self.mlp = None
        self.degrees = dict(graph.degree())
    
    def train_embeddings(self) -> Dict[int, np.ndarray]:
        """Train embeddings using sparse methods (memory efficient)."""
        
        
        print("Computing spectral embeddings (sparse method)...")
        
        G_undirected = self.graph.to_undirected()
        largest_cc = max(nx.connected_components(G_undirected), key=len)
        G_sub = G_undirected.subgraph(largest_cc)
        
        L = nx.normalized_laplacian_matrix(G_sub)
        
        # Use a sparse eigensolver and compute only the top k eigenvectors
        k = min(self.embedding_dim + 1, L.shape[0] - 2)
        
        # eigsh is much more memory-efficient than eigh
        eigenvalues, eigenvectors = eigsh(L, k=k, which='SM')  # SM = smallest magnitude
        
        node_list = list(G_sub.nodes())
        self.embeddings = {}
        for i, node in enumerate(node_list):
            # Skip the first eigenvector (trivial solution)
            self.embeddings[node] = eigenvectors[i, 1:]
        
        # Assign random embeddings to nodes not in the largest connected component
        for node in self.graph.nodes():
            if node not in self.embeddings:
                self.embeddings[node] = np.random.randn(k - 1) * 0.01
        
        print(f"  Embeddings computed for {len(self.embeddings):,} nodes")
        return self.embeddings
    
    def _edge_features(self, u: int, v: int) -> np.ndarray:
        features = []
        
        if self.embeddings:
            emb_u = self.embeddings.get(u, np.zeros(self.embedding_dim))
            emb_v = self.embeddings.get(v, np.zeros(self.embedding_dim))
            
            norm_u, norm_v = np.linalg.norm(emb_u), np.linalg.norm(emb_v)
            cos_sim = np.dot(emb_u, emb_v) / (norm_u * norm_v) if norm_u > 0 and norm_v > 0 else 0.0
            features.append(cos_sim)
            features.append(np.linalg.norm(emb_u - emb_v))
        
        features.append(np.log1p(self.degrees.get(u, 0)))
        features.append(np.log1p(self.degrees.get(v, 0)))
        features.append(np.log1p(self.degrees.get(u, 1) / max(self.degrees.get(v, 1), 1)))
        
        return np.array(features)
    
    def train_edge_weights(self, training_pathways: List[Dict], negative_ratio: float = 3.0):
        """Train MLP to predict edge weights."""
        
        if self.embeddings is None:
            self.train_embeddings()
        
        print("  Training edge weight MLP...")
        
        X_train, y_train = [], []
        positive_edges = set()
        
        for pathway in training_pathways:
            path = pathway['path_nodes']
            for i in range(len(path) - 1):
                u, v = path[i], path[i+1]
                if self.graph.has_edge(u, v):
                    positive_edges.add((u, v))
                    X_train.append(self._edge_features(u, v))
                    y_train.append(0.1)
        
        all_edges = list(self.graph.edges())
        np.random.shuffle(all_edges)
        
        n_negative = int(len(positive_edges) * negative_ratio)
        for u, v in all_edges[:n_negative * 2]:
            if (u, v) not in positive_edges and len(X_train) < len(positive_edges) + n_negative:
                X_train.append(self._edge_features(u, v))
                y_train.append(1.0)
        
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_train)
        
        self.mlp = MLPRegressor(
            hidden_layer_sizes=(32, 16),
            activation='relu',
            max_iter=500,
            early_stopping=True,
            random_state=42
        )
        self.mlp.fit(X_scaled, y_train)
        
        print(f"  MLP trained on {len(X_train)} samples (R²={self.mlp.score(X_scaled, y_train):.3f})")
        self._precompute_edge_weights()
    
    def _precompute_edge_weights(self):
        """Precompute weights for all edges."""
        print("  Precomputing edge weights...")
        self.edge_weights = {}
        edges = list(self.graph.edges())
        
        X = np.array([self._edge_features(u, v) for u, v in edges])
        X_scaled = self.scaler.transform(X)
        weights = np.clip(self.mlp.predict(X_scaled), 0.01, 2.0)
        
        for (u, v), w in zip(edges, weights):
            self.edge_weights[(u, v)] = w
        
        print(f"  Edge weights computed for {len(self.edge_weights):,} edges")
    
    def _heuristic(self, node: int, target: int) -> float:
        if self.embeddings is None:
            return 0.0
        emb_node = self.embeddings.get(node, np.zeros(self.embedding_dim))
        emb_target = self.embeddings.get(target, np.zeros(self.embedding_dim))
        return np.linalg.norm(emb_node - emb_target) * 0.1
    
    def find_path(self, source: int, target: int) -> Tuple[List[int], List[str], float]:
        """Find path using A* with learned edge weights."""
        if self.edge_weights is None:
            self.edge_weights = {(u, v): 1.0 for u, v in self.graph.edges()}
        
        counter = 0
        open_set = [(self._heuristic(source, target), counter, source, [source], 0.0)]
        visited = set()
        
        while open_set:
            f_score, _, current, path, g_score = heapq.heappop(open_set)
            
            if current == target:
                relations = []
                for i in range(len(path) - 1):
                    edge_data = self.graph.get_edge_data(path[i], path[i+1])
                    relations.append(edge_data.get('relation', 'unknown'))
                return path, relations, g_score
            
            if current in visited:
                continue
            visited.add(current)
            
            for neighbor in self.graph.neighbors(current):
                if neighbor in visited:
                    continue

                # ✅ your bans
                if not allowed_transition(self.graph, source, current, neighbor):
                    continue

                edge_weight = self.edge_weights.get((current, neighbor), 1.0)
                new_g = g_score + edge_weight
                new_f = new_g + self._heuristic(neighbor, target)
                counter += 1
                heapq.heappush(open_set, (new_f, counter, neighbor, path + [neighbor], new_g))
        
        return [], [], float('inf')
    

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
"""
Automated Pathway Validation for Drug Repurposing in PrimeKG
============================================================
This script automates the validation of whether known drug repurposing pathways
exist in PrimeKG before using them as benchmark test cases.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Set, Optional
from dataclasses import dataclass
from collections import defaultdict
import json


@dataclass
class PathwayValidationResult:
    """Results from validating a single pathway"""
    drug: str
    disease: str
    expected_pathway: List[str]
    found_in_kg: bool
    missing_nodes: List[str]
    missing_edges: List[Tuple[str, str]]
    alternative_nodes: Dict[str, List[str]]
    coverage_score: float
    validation_details: Dict


class PathwayValidator:
    """
    Validates whether documented drug repurposing pathways exist in PrimeKG
    """
    
    def __init__(self, primekg_path: str):
        """
        Initialize validator with PrimeKG data
        
        Args:
            primekg_path: Path to PrimeKG CSV file
        """
        print("Loading PrimeKG...")
        self.kg = pd.read_csv(primekg_path)
        print(f"Loaded {len(self.kg)} edges from PrimeKG")
        
        # Build efficient lookup structures
        self._build_indexes()
        
    def _build_indexes(self):
        """Build indexes for fast lookup"""
        print("Building indexes...")
        
        # Node index: all unique entities
        all_x = set(self.kg['x_name'].str.lower())
        all_y = set(self.kg['y_name'].str.lower())
        self.all_nodes = all_x.union(all_y)
        
        # Edge index: (source, target) -> relation types
        self.edge_index = defaultdict(set)
        for _, row in self.kg.iterrows():
            source = row['x_name'].lower()
            target = row['y_name'].lower()
            relation = row['relation']
            self.edge_index[(source, target)].add(relation)
            
        # Type index: entity -> type
        self.node_types = {}
        for _, row in self.kg.iterrows():
            self.node_types[row['x_name'].lower()] = row['x_type']
            self.node_types[row['y_name'].lower()] = row['y_type']
            
        # Neighbor index: node -> connected nodes
        self.neighbors = defaultdict(set)
        for _, row in self.kg.iterrows():
            source = row['x_name'].lower()
            target = row['y_name'].lower()
            self.neighbors[source].add(target)
            self.neighbors[target].add(source)
            
        print(f"Index built: {len(self.all_nodes)} unique nodes")
        
    def validate_pathway(self, 
                        drug: str, 
                        disease: str, 
                        expected_pathway: List[str],
                        check_alternatives: bool = True) -> PathwayValidationResult:
        """
        Validate if a documented pathway exists in PrimeKG
        
        Args:
            drug: Drug name (e.g., "Thalidomide")
            disease: Disease name (e.g., "multiple myeloma")
            expected_pathway: Ordered list of entities in the pathway
                             e.g., ["Thalidomide", "CRBN", "protein ubiquitination", "disease"]
            check_alternatives: Whether to search for alternative node names
            
        Returns:
            PathwayValidationResult with detailed validation info
        """
        print(f"\n{'='*60}")
        print(f"Validating pathway: {drug} → {disease}")
        print(f"Expected pathway: {' → '.join(expected_pathway)}")
        print(f"{'='*60}")
        
        # Normalize names
        pathway_normalized = [node.lower() for node in expected_pathway]
        
        # Check node existence
        missing_nodes = []
        alternative_nodes = {}
        
        for node in pathway_normalized:
            if node not in self.all_nodes:
                missing_nodes.append(node)
                if check_alternatives:
                    alternatives = self._find_alternative_names(node)
                    if alternatives:
                        alternative_nodes[node] = alternatives
                        
        # Check edge existence
        missing_edges = []
        for i in range(len(pathway_normalized) - 1):
            source = pathway_normalized[i]
            target = pathway_normalized[i + 1]
            
            # Check if this edge exists (in either direction)
            if (source, target) not in self.edge_index and \
               (target, source) not in self.edge_index:
                missing_edges.append((source, target))
                
        # Calculate coverage score
        total_components = len(pathway_normalized) + (len(pathway_normalized) - 1)
        missing_components = len(missing_nodes) + len(missing_edges)
        coverage_score = (total_components - missing_components) / total_components
        
        # Determine if pathway is found
        found_in_kg = len(missing_nodes) == 0 and len(missing_edges) == 0
        
        # Detailed validation info
        validation_details = {
            'total_nodes': len(pathway_normalized),
            'existing_nodes': len(pathway_normalized) - len(missing_nodes),
            'total_edges': len(pathway_normalized) - 1,
            'existing_edges': len(pathway_normalized) - 1 - len(missing_edges),
            'node_types': {node: self.node_types.get(node, 'UNKNOWN') 
                          for node in pathway_normalized if node in self.all_nodes}
        }
        
        result = PathwayValidationResult(
            drug=drug,
            disease=disease,
            expected_pathway=expected_pathway,
            found_in_kg=found_in_kg,
            missing_nodes=missing_nodes,
            missing_edges=missing_edges,
            alternative_nodes=alternative_nodes,
            coverage_score=coverage_score,
            validation_details=validation_details
        )
        
        self._print_validation_result(result)
        return result
        
    def _find_alternative_names(self, node: str, top_k: int = 5) -> List[str]:
        """
        Find alternative names for a missing node using fuzzy matching
        
        Args:
            node: Node name to search for
            top_k: Number of top matches to return
            
        Returns:
            List of similar node names found in KG
        """
        # Simple substring matching (can be enhanced with fuzzy matching libraries)
        candidates = []
        node_lower = node.lower()
        
        for kg_node in self.all_nodes:
            # Check if node is substring of kg_node or vice versa
            if node_lower in kg_node or kg_node in node_lower:
                candidates.append(kg_node)
                
        return candidates[:top_k]
        
    def _print_validation_result(self, result: PathwayValidationResult):
        """Pretty print validation results"""
        print(f"\n✓ VALIDATION RESULT:")
        print(f"  Coverage Score: {result.coverage_score:.1%}")
        print(f"  Pathway Found: {'✓ YES' if result.found_in_kg else '✗ NO'}")
        
        if result.missing_nodes:
            print(f"\n  Missing Nodes ({len(result.missing_nodes)}):")
            for node in result.missing_nodes:
                print(f"    ✗ {node}")
                if node in result.alternative_nodes:
                    print(f"      Alternatives: {', '.join(result.alternative_nodes[node][:3])}")
                    
        if result.missing_edges:
            print(f"\n  Missing Edges ({len(result.missing_edges)}):")
            for source, target in result.missing_edges:
                print(f"    ✗ {source} → {target}")
                
        print(f"\n  Node Details:")
        for node, node_type in result.validation_details['node_types'].items():
            print(f"    ✓ {node} ({node_type})")
            
    
        
    def discover_actual_pathway(self, 
                               drug: str, 
                               disease: str,
                               max_depth: int = 4) -> List[List[str]]:
        """
        Discover what pathways actually exist in KG between drug and disease
        
        Args:
            drug: Drug name
            disease: Disease name
            max_depth: Maximum path length to search
            
        Returns:
            List of paths found (each path is a list of nodes)
        """
        print(f"\n🔍 Discovering pathways: {drug} → {disease}")
        
        drug_lower = drug.lower()
        disease_lower = disease.lower()
        
        # Check if both endpoints exist
        if drug_lower not in self.all_nodes:
            print(f"  ✗ Drug '{drug}' not found in KG")
            return []
            
        if disease_lower not in self.all_nodes:
            print(f"  ✗ Disease '{disease}' not found in KG")
            return []
            
        # BFS to find paths
        paths = []
        queue = [(drug_lower, [drug_lower])]
        visited_at_depth = {0: {drug_lower}}
        
        for depth in range(1, max_depth + 1):
            visited_at_depth[depth] = set()
            next_queue = []
            
            while queue:
                current, path = queue.pop(0)
                
                # Check all neighbors
                for neighbor in self.neighbors[current]:
                    if neighbor == disease_lower:
                        # Found a path!
                        complete_path = path + [neighbor]
                        paths.append(complete_path)
                        print(f"  ✓ Found path (length {len(complete_path)}): {' → '.join(complete_path)}")
                        
                    elif neighbor not in visited_at_depth[depth]:
                        visited_at_depth[depth].add(neighbor)
                        next_queue.append((neighbor, path + [neighbor]))
                        
            queue = next_queue
            
            if not queue:
                break
                
        if not paths:
            print(f"  ✗ No paths found within {max_depth} hops")
            
        return paths
        
    

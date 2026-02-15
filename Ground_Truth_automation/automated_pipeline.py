"""
Automated Drug Repurposing Pathway Validation Pipeline
======================================================
Complete pipeline for validating benchmark pathways against PrimeKG
"""

import pandas as pd
import yaml
import json
from pathlib import Path
from typing import List, Dict
import argparse
from datetime import datetime


class AutomatedPathwayBenchmark:
    """
    Input: pathway_config.yaml (your 50 candidate pathways)
    ↓
    [AutomatedPathwayBenchmark does these things]
    ↓
    1️⃣ Read all candidate pathways
    2️⃣ Validate each pathway one by one
    3️⃣ Automatically exclude incomplete ones (like Minoxidil)
    4️⃣ Discover alternative paths for excluded pathways
    5️⃣ Generate benchmark dataset
    6️⃣ Generate detailed reports
    
    """
    
    def __init__(self, primekg_path: str, config_path: str):
        """
        Initialize benchmark pipeline
        
        Args:
            primekg_path: Path to PrimeKG CSV
            config_path: Path to YAML config with pathway definitions
        """
        from pathway_validator import PathwayValidator
        
        print("="*70)
        print("DRUG REPURPOSING PATHWAY BENCHMARK - AUTOMATED VALIDATION")
        print("="*70)
        
        # Load PrimeKG
        self.validator = PathwayValidator(primekg_path)
        
        # Load pathway configurations
        print(f"\nLoading pathway configurations from: {config_path}")
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        self.pathways = self.config['pathways']
        self.settings = self.config.get('validation_settings', {})
        
        print(f"✓ Loaded {len(self.pathways)} pathway configurations")
        
    def validate_all_pathways(self) -> Dict:
        """
        Validate all pathways defined in config
        
        Returns:
            Dictionary with comprehensive validation results
        """
        print("\n" + "="*70)
        print("PHASE 1: PATHWAY VALIDATION")
        print("="*70)
        
        results = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'total_pathways': len(self.pathways),
                'settings': self.settings
            },
            'pathways': [],
            'summary': {}
        }
        
        valid_count = 0
        partial_count = 0
        missing_count = 0
        
        for idx, pathway_config in enumerate(self.pathways, 1):
            print(f"\n[{idx}/{len(self.pathways)}] Validating: {pathway_config['name']}")
            
            # Extract pathway nodes
            pathway_nodes = [node['node'] for node in pathway_config['expected_pathway']]
            
            # Validate
            validation_result = self.validator.validate_pathway(
                drug=pathway_config['drug'],
                disease=pathway_config['disease'],
                expected_pathway=pathway_nodes,
                check_alternatives=self.settings.get('check_alternative_names', True)
            )
            
            # Classify result
            if validation_result.found_in_kg:
                valid_count += 1
                status = "VALID"
            elif validation_result.coverage_score >= self.settings.get('min_coverage_threshold', 0.7):
                partial_count += 1
                status = "PARTIAL"
            else:
                missing_count += 1
                status = "MISSING"
                
            # Store result
            pathway_result = {
                'name': pathway_config['name'],
                'drug': pathway_config['drug'],
                'disease': pathway_config['disease'],
                'mechanism_type': pathway_config.get('mechanism_type'),
                'documented_source': pathway_config.get('documented_source'),
                'status': status,
                'found_in_kg': validation_result.found_in_kg,
                'coverage_score': validation_result.coverage_score,
                'expected_pathway': pathway_nodes,
                'missing_nodes': validation_result.missing_nodes,
                'missing_edges': [f"{s}→{t}" for s, t in validation_result.missing_edges],
                'alternative_nodes': validation_result.alternative_nodes,
                'node_types': validation_result.validation_details['node_types']
            }
            
            results['pathways'].append(pathway_result)
            
        # Summary statistics
        results['summary'] = {
            'valid': valid_count,
            'partial': partial_count,
            'missing': missing_count,
            'valid_percentage': valid_count / len(self.pathways) * 100,
            'usable_percentage': (valid_count + partial_count) / len(self.pathways) * 100
        }
        
        return results
        
    def discover_missing_pathways(self, results: Dict) -> Dict:
        """
        For pathways not found, discover what actually exists
        
        Args:
            results: Results from validate_all_pathways()
            
        Returns:
            Updated results with discovered pathways
        """
        print("\n" + "="*70)
        print("PHASE 2: DISCOVERING ALTERNATIVE PATHWAYS")
        print("="*70)
        
        for pathway in results['pathways']:
            if pathway['status'] in ['PARTIAL', 'MISSING']:
                print(f"\nSearching alternatives for: {pathway['name']}")
                
                actual_paths = self.validator.discover_actual_pathway(
                    drug=pathway['drug'],
                    disease=pathway['disease'],
                    max_depth=self.settings.get('max_search_depth', 5)
                )
                
                pathway['discovered_pathways'] = [
                    {'length': len(path), 'nodes': path}
                    for path in actual_paths[:5]  # Top 5 shortest paths
                ]
                
        return results
        
    def generate_benchmark_dataset(self, results: Dict, output_dir: str):
        """
        Generate benchmark dataset files for algorithm testing
        
        Args:
            results: Validation results
            output_dir: Directory to save benchmark files
        """
        print("\n" + "="*70)
        print("PHASE 3: GENERATING BENCHMARK DATASET")
        print("="*70)
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 1. Valid pathways for testing (ground truth)
        valid_pathways = [p for p in results['pathways'] if p['status'] == 'VALID']
        
        ground_truth = []
        for p in valid_pathways:
            ground_truth.append({
                'query_id': p['name'],
                'drug': p['drug'],
                'disease': p['disease'],
                'mechanism': p['mechanism_type'],
                'ground_truth_path': p['expected_pathway'],
                'path_length': len(p['expected_pathway']),
                'source': p['documented_source']
            })
            
        with open(output_path / 'ground_truth_pathways.json', 'w') as f:
            json.dump(ground_truth, f, indent=2)
        print(f"✓ Saved {len(ground_truth)} ground truth pathways")
        
        # 2. Test queries (drug-disease pairs without paths)
        test_queries = []
        for p in valid_pathways:
            test_queries.append({
                'query_id': p['name'],
                'drug': p['drug'],
                'disease': p['disease']
            })
            
        with open(output_path / 'test_queries.json', 'w') as f:
            json.dump(test_queries, f, indent=2)
        print(f"✓ Saved {len(test_queries)} test queries")
        
        # 3. Negative examples (pathways not in KG)
        negative_examples = [p for p in results['pathways'] if p['status'] == 'MISSING']
        
        with open(output_path / 'negative_examples.json', 'w') as f:
            json.dump(negative_examples, f, indent=2)
        print(f"✓ Saved {len(negative_examples)} negative examples")
        
        # 4. Full validation report
        with open(output_path / 'full_validation_report.json', 'w') as f:
            json.dump(results, f, indent=2)
        print(f"✓ Saved full validation report")
        
        # 5. CSV summary for easy inspection
        df_summary = pd.DataFrame([
            {
                'pathway_name': p['name'],
                'drug': p['drug'],
                'disease': p['disease'],
                'status': p['status'],
                'coverage': f"{p['coverage_score']:.1%}",
                'path_length': len(p['expected_pathway']),
                'missing_nodes': len(p['missing_nodes']),
                'missing_edges': len(p['missing_edges'])
            }
            for p in results['pathways']
        ])
        
        df_summary.to_csv(output_path / 'validation_summary.csv', index=False)
        print(f"✓ Saved validation summary CSV")
        
        return output_path
        
    def print_summary_report(self, results: Dict):
        """Print human-readable summary"""
        print("\n" + "="*70)
        print("VALIDATION SUMMARY REPORT")
        print("="*70)
        
        summary = results['summary']
        total = results['metadata']['total_pathways']
        
        print(f"\nTotal Pathways Tested: {total}")
        print(f"\n✓ Valid (100% found):     {summary['valid']:2d} ({summary['valid_percentage']:.1f}%)")
        print(f"◐ Partial (>70% found):  {summary['partial']:2d}")
        print(f"✗ Missing (<70% found):  {summary['missing']:2d}")
        print(f"\nUsable for Benchmark:    {summary['valid'] + summary['partial']} ({summary['usable_percentage']:.1f}%)")
        
        print("\n" + "-"*70)
        print("RECOMMENDATION:")
        if summary['valid_percentage'] >= 80:
            print("✓ Excellent! Most pathways are in PrimeKG - proceed with benchmark")
        elif summary['usable_percentage'] >= 60:
            print("◐ Good. Consider using partial matches with relaxed constraints")
        else:
            print("✗ Warning: Many pathways missing. Consider:")
            print("  1. Alternative node names/synonyms")
            print("  2. Using discovered pathways instead")
            print("  3. Adding edges to PrimeKG from other sources")
            
        print("\n" + "="*70)
        
    def run_full_pipeline(self, output_dir: str = '/mnt/user-data/outputs'):
        """
        Run complete validation and benchmark generation pipeline
        
        Args:
            output_dir: Where to save all outputs
        """
        # Phase 1: Validate all pathways
        results = self.validate_all_pathways()
        
        # Phase 2: Discover alternatives for missing pathways
        results = self.discover_missing_pathways(results)
        
        # Phase 3: Generate benchmark dataset
        output_path = self.generate_benchmark_dataset(results, output_dir)
        
        # Print summary
        self.print_summary_report(results)
        
        print(f"\n✓ All outputs saved to: {output_path}")
        
        return results, output_path


def main():
    """Command-line interface"""
    parser = argparse.ArgumentParser(
        description='Automated Drug Repurposing Pathway Validation'
    )
    parser.add_argument(
        '--primekg',
        default='kg.csv',
        help='Path to PrimeKG CSV file'
    )
    parser.add_argument(
        '--config',
        default='pathway_config.yaml',
        help='Path to pathway configuration YAML'
    )
    parser.add_argument(
        '--output',
        default='/mnt/user-data/outputs',
        help='Output directory for benchmark files'
    )
    
    args = parser.parse_args()
    
    # Run pipeline
    pipeline = AutomatedPathwayBenchmark(
        primekg_path=args.primekg,
        config_path=args.config
    )
    
    results, output_path = pipeline.run_full_pipeline(output_dir=args.output)
    
    print("\n✓ Pipeline complete!")


if __name__ == "__main__":
    main()

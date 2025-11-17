import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr, spearmanr
from scipy.spatial.distance import jaccard
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import combinations

class GRNSimilarityAnalyzer:
    def __init__(self, dict_filtered_GRNs):
        """
        Initialize with the GRN dictionary
        dict_filtered_GRNs: {timepoint: {celltype: GRN_matrix}}
        """
        self.grns = dict_filtered_GRNs
        self.timepoints = list(dict_filtered_GRNs.keys())
        self.celltypes = list(dict_filtered_GRNs[self.timepoints[0]].keys())
        print(f"Found {len(self.timepoints)} timepoints: {self.timepoints}")
        print(f"Found {len(self.celltypes)} celltypes")
        
    def extract_tf_gene_pairs(self, grn_matrix):
        """Extract TF-gene pairs from a GRN matrix (handles both wide and long formats)"""
        if grn_matrix is None or grn_matrix.empty:
            return set()
        
        pairs = set()
        
        # Check if this is long format (has 'source', 'target' columns)
        if 'source' in grn_matrix.columns and 'target' in grn_matrix.columns:
            # Long format: source, target, coef_mean (or similar)
            coef_col = 'coef_mean' if 'coef_mean' in grn_matrix.columns else grn_matrix.columns[-1]
            
            for _, row in grn_matrix.iterrows():
                source = row['source']  # This is the TF
                target = row['target']  # This is the target gene
                val = row[coef_col]
                
                # Check if value is non-zero
                try:
                    if pd.notna(val) and float(val) != 0:
                        pairs.add(f"{target}_{source}")  # target_TF format
                except (ValueError, TypeError):
                    continue
                    
        else:
            # Wide format: matrix with targets as rows, TFs as columns
            grn_stacked = grn_matrix.stack()
            for (target, tf), val in grn_stacked.items():
                try:
                    if pd.notna(val) and float(val) != 0:
                        pairs.add(f"{target}_{tf}")
                except (ValueError, TypeError):
                    continue
        
        return pairs
    
    def get_superset_pairs(self, grn_dict):
        """
        Get union (superset) of all TF-gene pairs from a dictionary of GRNs
        grn_dict: {identifier: GRN_matrix}
        """
        all_pairs = set()
        valid_grns = 0
        
        for identifier, grn in grn_dict.items():
            if grn is not None and not grn.empty:
                pairs = self.extract_tf_gene_pairs(grn)
                all_pairs.update(pairs)
                valid_grns += 1
        
        print(f"  Superset contains {len(all_pairs):,} unique TF-gene pairs from {valid_grns} valid GRNs")
        return sorted(list(all_pairs))  # Sort for consistent ordering
    
    def grn_to_vector_superset(self, grn, superset_pairs):
        """
        Convert GRN matrix to feature vector using superset of pairs
        Missing pairs are filled with 0 (no regulation)
        Handles both wide and long format GRN matrices
        """
        if grn is None or grn.empty:
            return np.zeros(len(superset_pairs))
        
        feature_dict = {}
        
        # Check if this is long format (has 'source', 'target' columns)
        if 'source' in grn.columns and 'target' in grn.columns:
            # Long format: source, target, coef_mean (or similar)
            coef_col = 'coef_mean' if 'coef_mean' in grn.columns else grn.columns[-1]
            
            for _, row in grn.iterrows():
                source = row['source']  # This is the TF
                target = row['target']  # This is the target gene
                val = row[coef_col]
                
                # Convert to numeric
                try:
                    val = float(val) if pd.notna(val) else 0.0
                except (ValueError, TypeError):
                    val = 0.0
                
                feature_dict[f"{target}_{source}"] = val
                
        else:
            # Wide format: matrix with targets as rows, TFs as columns
            grn_stacked = grn.stack()
            for (target, tf), val in grn_stacked.items():
                # Convert to numeric
                try:
                    val = float(val) if pd.notna(val) else 0.0
                except (ValueError, TypeError):
                    val = 0.0
                
                feature_dict[f"{target}_{tf}"] = val
        
        # Create vector for superset: existing pairs get their values, missing pairs get 0
        vector = np.array([feature_dict.get(pair, 0.0) for pair in superset_pairs], dtype=np.float64)
        return vector
    
    def compute_similarity_matrix(self, vectors_dict, metric='pearson'):
        """
        Compute similarity matrix from feature vectors dictionary
        vectors_dict: {identifier: vector}
        """
        identifiers = list(vectors_dict.keys())
        vectors = list(vectors_dict.values())
        n = len(vectors)
        
        similarity_matrix = np.full((n, n), np.nan)
        
        for i in range(n):
            for j in range(i, n):
                if np.all(np.isnan(vectors[i])) or np.all(np.isnan(vectors[j])):
                    continue
                    
                if metric == 'pearson':
                    # Handle constant vectors
                    if np.std(vectors[i]) == 0 or np.std(vectors[j]) == 0:
                        corr = 1.0 if np.allclose(vectors[i], vectors[j]) else 0.0
                    else:
                        corr, _ = pearsonr(vectors[i], vectors[j])
                        if np.isnan(corr):
                            corr = 0.0
                            
                elif metric == 'cosine':
                    corr = cosine_similarity([vectors[i]], [vectors[j]])[0, 0]
                    
                elif metric == 'spearman':
                    # Handle constant vectors
                    if np.std(vectors[i]) == 0 or np.std(vectors[j]) == 0:
                        corr = 1.0 if np.allclose(vectors[i], vectors[j]) else 0.0
                    else:
                        corr, _ = spearmanr(vectors[i], vectors[j])
                        if np.isnan(corr):
                            corr = 0.0
                
                elif metric == 'jaccard':
                    # Jaccard on binary vectors (non-zero elements)
                    binary_i = (vectors[i] != 0).astype(int)
                    binary_j = (vectors[j] != 0).astype(int)
                    
                    intersection = np.sum(binary_i & binary_j)
                    union = np.sum(binary_i | binary_j)
                    
                    corr = intersection / union if union > 0 else 0.0
                
                similarity_matrix[i, j] = similarity_matrix[j, i] = corr
        
        return similarity_matrix, identifiers
    
    def compute_data_characteristics(self, vectors_dict, context=""):
        """Compute and report data characteristics"""
        vectors = list(vectors_dict.values())
        identifiers = list(vectors_dict.keys())
        
        if not vectors:
            return {}
        
        # Ensure all vectors are numeric
        numeric_vectors = []
        for v in vectors:
            if v.dtype.kind not in 'biufc':  # not a numeric type
                print(f"Warning: Non-numeric vector detected, converting to float64")
                v = np.array([float(x) if x != '' else 0.0 for x in v], dtype=np.float64)
            numeric_vectors.append(v)
        
        vectors = numeric_vectors
        
        # Basic statistics
        total_pairs = len(vectors[0]) if vectors else 0
        sparsity_per_grn = [np.mean(v == 0) for v in vectors]
        nonzero_per_grn = [np.sum(v != 0) for v in vectors]
        
        # Compute mean values only for non-zero elements
        mean_values = []
        for v in vectors:
            nonzero_vals = v[v != 0]
            if len(nonzero_vals) > 0:
                mean_values.append(np.mean(nonzero_vals))
            else:
                mean_values.append(0.0)
        
        stats = {
            'context': context,
            'total_pairs': total_pairs,
            'n_grns': len(vectors),
            'mean_sparsity': np.mean(sparsity_per_grn),
            'std_sparsity': np.std(sparsity_per_grn),
            'mean_nonzero_pairs': np.mean(nonzero_per_grn),
            'std_nonzero_pairs': np.std(nonzero_per_grn),
            'mean_interaction_strength': np.mean(mean_values),
            'sparsity_per_grn': dict(zip(identifiers, sparsity_per_grn)),
            'nonzero_per_grn': dict(zip(identifiers, nonzero_per_grn))
        }
        
        # Print summary
        print(f"\n📊 Data characteristics for {context}:")
        print(f"  Total TF-gene pairs in superset: {total_pairs:,}")
        print(f"  Number of GRNs: {len(vectors)}")
        print(f"  Mean sparsity (% zeros): {stats['mean_sparsity']:.3f} ± {stats['std_sparsity']:.3f}")
        print(f"  Mean active pairs per GRN: {stats['mean_nonzero_pairs']:,.0f} ± {stats['std_nonzero_pairs']:,.0f}")
        print(f"  Mean interaction strength: {stats['mean_interaction_strength']:.3f}")
        
        # Warnings
        if stats['mean_sparsity'] > 0.95:
            print("  Very sparse data (>95% zeros) - consider Jaccard similarity")
        if stats['mean_nonzero_pairs'] < 100:
            print("  Very few active pairs per GRN - limited statistical power")
            
        return stats
    
    def analyze_timepoint_similarity(self, metrics=['pearson', 'cosine', 'spearman', 'jaccard'],
                                   comparison_method='superset', min_overlap_threshold=0.2):
        """
        Analyze how similar celltypes are within each timepoint using superset approach
        
        Parameters:
        -----------
        metrics : list
            Similarity metrics to compute
        comparison_method : str
            'superset' - use union of all pairs, fill missing with 0
            'intersection' - use only common pairs
            'adaptive' - use intersection if overlap > threshold, else superset
        min_overlap_threshold : float
            Minimum overlap fraction required for intersection method (0.0-1.0)
        """
        results = {
            'timepoints': [],
            'data_stats': {},
            'comparison_method_used': []
        }
        
        # Initialize results for each metric
        for metric in metrics:
            results[f'{metric}_mean'] = []
            results[f'{metric}_std'] = []
            results[f'{metric}_sem'] = []
        
        for tp in self.timepoints:
            print(f"\n🔍 Analyzing timepoint: {tp}")
            
            # Get all valid GRNs for this timepoint
            tp_grns = {}
            for ct in self.celltypes:
                grn = self.grns[tp].get(ct)
                if grn is not None and not grn.empty:
                    tp_grns[ct] = grn
            
            if len(tp_grns) < 2:
                print(f"  ❌ Skipping {tp} - insufficient data ({len(tp_grns)} valid GRNs)")
                continue
            
            print(f"  Valid celltypes: {len(tp_grns)}/{len(self.celltypes)}")
            
            # Determine which comparison method to use
            if comparison_method == 'adaptive':
                # Check overlap between GRNs to decide method
                all_pair_sets = []
                for grn in tp_grns.values():
                    pairs = self.extract_tf_gene_pairs(grn)
                    all_pair_sets.append(pairs)
                
                # Calculate average pairwise overlap
                total_overlaps = 0
                total_comparisons = 0
                for i in range(len(all_pair_sets)):
                    for j in range(i+1, len(all_pair_sets)):
                        intersection_size = len(all_pair_sets[i] & all_pair_sets[j])
                        union_size = len(all_pair_sets[i] | all_pair_sets[j])
                        if union_size > 0:
                            total_overlaps += intersection_size / union_size
                            total_comparisons += 1
                
                avg_overlap = total_overlaps / total_comparisons if total_comparisons > 0 else 0
                method_used = 'intersection' if avg_overlap >= min_overlap_threshold else 'superset'
                print(f"  Average overlap: {avg_overlap:.3f}, using {method_used} method")
            else:
                method_used = comparison_method
                print(f"  Using {method_used} method")
            
            results['comparison_method_used'].append(method_used)
            
            # Apply the chosen method
            if method_used == 'intersection':
                # Find common pairs across all GRNs
                common_pairs = None
                for grn in tp_grns.values():
                    pairs = self.extract_tf_gene_pairs(grn)
                    if common_pairs is None:
                        common_pairs = pairs
                    else:
                        common_pairs = common_pairs.intersection(pairs)
                
                common_pairs = sorted(list(common_pairs))
                print(f"  Using {len(common_pairs)} common pairs")
                
                if len(common_pairs) < 10:
                    print(f"  ⚠️ Very few common pairs, results may not be reliable")
                
                # Convert to vectors using only common pairs
                vectors_dict = {}
                for ct, grn in tp_grns.items():
                    vector = np.array([self._get_pair_value(grn, pair) for pair in common_pairs])
                    vectors_dict[ct] = vector
                    
            else:  # superset method
                # Get superset of all pairs
                superset_pairs = self.get_superset_pairs(tp_grns)
                print(f"  Using {len(superset_pairs)} total pairs (superset)")
                
                # Convert to vectors using superset (missing = 0)
                vectors_dict = {}
                for ct, grn in tp_grns.items():
                    vectors_dict[ct] = self.grn_to_vector_superset(grn, superset_pairs)
            
            # Compute data characteristics
            data_stats = self.compute_data_characteristics(vectors_dict, f"timepoint {tp}")
            results['data_stats'][tp] = data_stats
            
            # Compute similarities for each requested metric
            metric_results = {}
            for metric in metrics:
                sim_matrix, identifiers = self.compute_similarity_matrix(vectors_dict, metric)
                
                # Extract upper triangular values (unique pairs)
                mask = np.triu(np.ones_like(sim_matrix, dtype=bool), k=1)
                sim_values = sim_matrix[mask]
                sim_values = sim_values[~np.isnan(sim_values)]
                
                if len(sim_values) > 0:
                    metric_results[metric] = {
                        'mean': np.mean(sim_values),
                        'std': np.std(sim_values),
                        'values': sim_values,
                        'n_comparisons': len(sim_values)
                    }
                    print(f"  {metric.capitalize()} similarity: {np.mean(sim_values):.3f} ± {np.std(sim_values):.3f} ({len(sim_values)} pairs)")
                else:
                    metric_results[metric] = {'mean': np.nan, 'std': np.nan, 'values': [], 'n_comparisons': 0}
                    print(f"  {metric.capitalize()} similarity: No valid comparisons")
            
            # Store results
            results['timepoints'].append(tp)
            for metric in metrics:
                results[f'{metric}_mean'].append(metric_results[metric]['mean'])
                results[f'{metric}_std'].append(metric_results[metric]['std'])
                # Calculate SEM (standard error of the mean)
                n_comparisons = metric_results[metric]['n_comparisons']
                sem = metric_results[metric]['std'] / np.sqrt(n_comparisons) if n_comparisons > 0 else 0
                results[f'{metric}_sem'].append(sem)
        
        return pd.DataFrame({k: v for k, v in results.items() if k not in ['data_stats', 'comparison_method_used']}), results['data_stats']
    
    def _get_pair_value(self, grn, pair):
        """Helper function to get value for a TF-gene pair from GRN"""
        target, source = pair.split('_', 1)
        
        # Handle long format (source, target, coef_mean columns)
        if 'source' in grn.columns and 'target' in grn.columns:
            coef_col = 'coef_mean' if 'coef_mean' in grn.columns else grn.columns[-1]
            match = grn[(grn['source'] == source) & (grn['target'] == target)]
            if not match.empty:
                try:
                    return float(match.iloc[0][coef_col])
                except (ValueError, TypeError):
                    return 0.0
            return 0.0
        else:
            # Handle wide format
            try:
                if target in grn.index and source in grn.columns:
                    val = grn.loc[target, source]
                    return float(val) if pd.notna(val) else 0.0
            except (KeyError, ValueError, TypeError):
                pass
            return 0.0
        """
        Analyze how similar celltypes are within each timepoint using superset approach
        """
        results = {
            'timepoints': [],
            'data_stats': {}
        }
        
        # Initialize results for each metric
        for metric in metrics:
            results[f'{metric}_mean'] = []
            results[f'{metric}_std'] = []
        
        for tp in self.timepoints:
            print(f"\n🔍 Analyzing timepoint: {tp}")
            
            # Get all valid GRNs for this timepoint
            tp_grns = {}
            for ct in self.celltypes:
                grn = self.grns[tp].get(ct)
                if grn is not None and not grn.empty:
                    tp_grns[ct] = grn
            
            if len(tp_grns) < 2:
                print(f"  ❌ Skipping {tp} - insufficient data ({len(tp_grns)} valid GRNs)")
                continue
            
            print(f"  Valid celltypes: {len(tp_grns)}/{len(self.celltypes)}")
            
            # Get superset of all TF-gene pairs for this timepoint
            superset_pairs = self.get_superset_pairs(tp_grns)
            
            # Convert each GRN to vector using superset (missing = 0)
            vectors_dict = {}
            for ct, grn in tp_grns.items():
                vectors_dict[ct] = self.grn_to_vector_superset(grn, superset_pairs)
            
            # Compute data characteristics
            data_stats = self.compute_data_characteristics(vectors_dict, f"timepoint {tp}")
            results['data_stats'][tp] = data_stats
            
            # Compute similarities for each requested metric
            metric_results = {}
            for metric in metrics:
                sim_matrix, identifiers = self.compute_similarity_matrix(vectors_dict, metric)
                
                # Extract upper triangular values (unique pairs)
                mask = np.triu(np.ones_like(sim_matrix, dtype=bool), k=1)
                sim_values = sim_matrix[mask]
                sim_values = sim_values[~np.isnan(sim_values)]
                
                if len(sim_values) > 0:
                    metric_results[metric] = {
                        'mean': np.mean(sim_values),
                        'std': np.std(sim_values),
                        'values': sim_values,
                        'n_comparisons': len(sim_values)
                    }
                    print(f"  {metric.capitalize()} similarity: {np.mean(sim_values):.3f} ± {np.std(sim_values):.3f} ({len(sim_values)} pairs)")
                else:
                    metric_results[metric] = {'mean': np.nan, 'std': np.nan, 'values': [], 'n_comparisons': 0}
                    print(f"  {metric.capitalize()} similarity: No valid comparisons")
            
            # Store results
            results['timepoints'].append(tp)
            for metric in metrics:
                results[f'{metric}_mean'].append(metric_results[metric]['mean'])
                results[f'{metric}_std'].append(metric_results[metric]['std'])
                # Calculate SEM (standard error of the mean)
                n_comparisons = metric_results[metric]['n_comparisons']
                sem = metric_results[metric]['std'] / np.sqrt(n_comparisons) if n_comparisons > 0 else 0
                results[f'{metric}_sem'] = results.get(f'{metric}_sem', [])
                results[f'{metric}_sem'].append(sem)
        
    def analyze_celltype_similarity(self, metrics=['pearson', 'cosine', 'spearman', 'jaccard'],
                                  comparison_method='superset', min_overlap_threshold=0.2):
        """
        Analyze how similar timepoints are within each celltype using superset approach
        
        Parameters:
        -----------
        metrics : list
            Similarity metrics to compute  
        comparison_method : str
            'superset' - use union of all pairs, fill missing with 0
            'intersection' - use only common pairs
            'adaptive' - use intersection if overlap > threshold, else superset
        min_overlap_threshold : float
            Minimum overlap fraction required for intersection method (0.0-1.0)
        """
        results = {
            'celltypes': [],
            'data_stats': {},
            'comparison_method_used': []
        }
        
        # Initialize results for each metric
        for metric in metrics:
            results[f'{metric}_mean'] = []
            results[f'{metric}_std'] = []
            results[f'{metric}_sem'] = []
        
        for ct in self.celltypes:
            print(f"\n🔍 Analyzing celltype: {ct}")
            
            # Get all valid GRNs for this celltype across timepoints
            ct_grns = {}
            for tp in self.timepoints:
                grn = self.grns[tp].get(ct)
                if grn is not None and not grn.empty:
                    ct_grns[tp] = grn
            
            if len(ct_grns) < 2:
                print(f"  ❌ Skipping {ct} - insufficient data ({len(ct_grns)} valid GRNs)")
                continue
            
            print(f"  Valid timepoints: {len(ct_grns)}/{len(self.timepoints)}")
            
            # Determine which comparison method to use
            if comparison_method == 'adaptive':
                # Check overlap between GRNs
                all_pair_sets = []
                for grn in ct_grns.values():
                    pairs = self.extract_tf_gene_pairs(grn)
                    all_pair_sets.append(pairs)
                
                # Calculate average pairwise overlap
                total_overlaps = 0
                total_comparisons = 0
                for i in range(len(all_pair_sets)):
                    for j in range(i+1, len(all_pair_sets)):
                        intersection_size = len(all_pair_sets[i] & all_pair_sets[j])
                        union_size = len(all_pair_sets[i] | all_pair_sets[j])
                        if union_size > 0:
                            total_overlaps += intersection_size / union_size
                            total_comparisons += 1
                
                avg_overlap = total_overlaps / total_comparisons if total_comparisons > 0 else 0
                method_used = 'intersection' if avg_overlap >= min_overlap_threshold else 'superset'
                print(f"  Average overlap: {avg_overlap:.3f}, using {method_used} method")
            else:
                method_used = comparison_method
                print(f"  Using {method_used} method")
            
            results['comparison_method_used'].append(method_used)
            
            # Apply the chosen method
            if method_used == 'intersection':
                # Find common pairs across all timepoints for this celltype
                common_pairs = None
                for grn in ct_grns.values():
                    pairs = self.extract_tf_gene_pairs(grn)
                    if common_pairs is None:
                        common_pairs = pairs
                    else:
                        common_pairs = common_pairs.intersection(pairs)
                
                common_pairs = sorted(list(common_pairs))
                print(f"  Using {len(common_pairs)} common pairs")
                
                if len(common_pairs) < 10:
                    print(f"  ⚠️ Very few common pairs, results may not be reliable")
                
                # Convert to vectors using only common pairs
                vectors_dict = {}
                for tp, grn in ct_grns.items():
                    vector = np.array([self._get_pair_value(grn, pair) for pair in common_pairs])
                    vectors_dict[tp] = vector
                    
            else:  # superset method
                # Get superset of all pairs
                superset_pairs = self.get_superset_pairs(ct_grns)
                print(f"  Using {len(superset_pairs)} total pairs (superset)")
                
                # Convert to vectors using superset (missing = 0)
                vectors_dict = {}
                for tp, grn in ct_grns.items():
                    vectors_dict[tp] = self.grn_to_vector_superset(grn, superset_pairs)
            
            # Compute data characteristics
            data_stats = self.compute_data_characteristics(vectors_dict, f"celltype {ct}")
            results['data_stats'][ct] = data_stats
            
            # Compute similarities for each requested metric
            metric_results = {}
            for metric in metrics:
                sim_matrix, identifiers = self.compute_similarity_matrix(vectors_dict, metric)
                
                # Extract upper triangular values
                mask = np.triu(np.ones_like(sim_matrix, dtype=bool), k=1)
                sim_values = sim_matrix[mask]
                sim_values = sim_values[~np.isnan(sim_values)]
                
                if len(sim_values) > 0:
                    metric_results[metric] = {
                        'mean': np.mean(sim_values),
                        'std': np.std(sim_values),
                        'values': sim_values,
                        'n_comparisons': len(sim_values)
                    }
                    print(f"  {metric.capitalize()} similarity: {np.mean(sim_values):.3f} ± {np.std(sim_values):.3f} ({len(sim_values)} pairs)")
                else:
                    metric_results[metric] = {'mean': np.nan, 'std': np.nan, 'values': [], 'n_comparisons': 0}
                    print(f"  {metric.capitalize()} similarity: No valid comparisons")
            
            # Store results
            results['celltypes'].append(ct)
            for metric in metrics:
                results[f'{metric}_mean'].append(metric_results[metric]['mean'])
                results[f'{metric}_std'].append(metric_results[metric]['std'])
                # Calculate SEM (standard error of the mean)
                n_comparisons = metric_results[metric]['n_comparisons']
                sem = metric_results[metric]['std'] / np.sqrt(n_comparisons) if n_comparisons > 0 else 0
                results[f'{metric}_sem'].append(sem)
        
        return pd.DataFrame({k: v for k, v in results.items() if k not in ['data_stats', 'comparison_method_used']}), results['data_stats']
    
   
    def plot_similarity_trends(self, tp_results, ct_results, tp_stats=None, ct_stats=None, 
                              metrics=['pearson', 'cosine', 'spearman'], save_path=None, 
                              sort_celltypes=True, include_violins=True, save_individual_panels=False):
        """
        Create comprehensive similarity plots with multiple metrics
        
        Parameters:
        -----------
        sort_celltypes : bool
            Whether to sort celltypes by similarity magnitude (high to low)
        include_violins : bool
            Whether to include violin plots showing distributions
        save_path : str or bool
            If string: save HTML with this path
            If True: save individual panels as PDF
            If None: don't save
        save_individual_panels : bool
            Whether to save each panel as separate PDF (requires matplotlib)
        """
        n_metrics = len(metrics)
        n_rows = 4 if include_violins else 3
        
        # Create subplot layout
        subplot_titles = []
        subplot_titles.extend([f'{m.capitalize()} - Celltype Similarity Within Timepoints' for m in metrics])
        subplot_titles.extend([f'{m.capitalize()} - Timepoint Similarity Within Celltypes' for m in metrics])
        subplot_titles.extend([f'{m.capitalize()} - Data Characteristics' for m in metrics])
        if include_violins:
            subplot_titles.extend([f'{m.capitalize()} - Similarity Distributions' for m in metrics])
        
        fig = make_subplots(
            rows=n_rows, cols=n_metrics,
            subplot_titles=subplot_titles,
            specs=[[{"secondary_y": False} for _ in range(n_metrics)] for _ in range(n_rows)],
            vertical_spacing=0.08
        )
        
        colors = ['blue', 'red', 'green', 'purple', 'orange', 'brown']
        
        # Row 1: Timepoint analysis (celltype similarity over time)
        for i, metric in enumerate(metrics):
            if f'{metric}_mean' in tp_results.columns and f'{metric}_sem' in tp_results.columns:
                fig.add_trace(
                    go.Scatter(
                        x=tp_results['timepoints'],
                        y=tp_results[f'{metric}_mean'],
                        error_y=dict(type='data', array=tp_results[f'{metric}_sem'], visible=True),
                        mode='lines+markers',
                        name=f'{metric.capitalize()} (TP)',
                        line=dict(color=colors[i % len(colors)], width=3),
                        marker=dict(size=8)
                    ),
                    row=1, col=i+1
                )
        
        # Row 2: Celltype analysis (timepoint similarity over celltypes) - SORTED
        for i, metric in enumerate(metrics):
            if f'{metric}_mean' in ct_results.columns and f'{metric}_sem' in ct_results.columns:
                
                # Sort celltypes by similarity magnitude if requested
                if sort_celltypes:
                    # Sort by mean similarity (descending) 
                    ct_sorted = ct_results.copy().sort_values(f'{metric}_mean', ascending=False)
                    x_values = list(range(len(ct_sorted)))
                    x_labels = ct_sorted['celltypes'].tolist()
                    y_values = ct_sorted[f'{metric}_mean']
                    error_values = ct_sorted[f'{metric}_sem']
                else:
                    x_values = list(range(len(ct_results)))
                    x_labels = ct_results['celltypes'].tolist()
                    y_values = ct_results[f'{metric}_mean']
                    error_values = ct_results[f'{metric}_sem']
                
                fig.add_trace(
                    go.Scatter(
                        x=x_values,
                        y=y_values,
                        error_y=dict(type='data', array=error_values, visible=True),
                        mode='markers',
                        name=f'{metric.capitalize()} (CT)',
                        marker=dict(color=colors[i % len(colors)], size=8, opacity=0.7),
                        text=x_labels,
                        hovertemplate='%{text}<br>' + f'{metric.capitalize()}: %{{y:.3f}} ± %{{error_y.array:.4f}}<extra></extra>',
                        showlegend=False
                    ),
                    row=2, col=i+1
                )
                
                # Update x-axis to show celltype names
                fig.update_xaxes(
                    ticktext=x_labels,
                    tickvals=x_values,
                    tickangle=45,
                    row=2, col=i+1
                )
        
        # Row 3: Data characteristics (if provided)
        if tp_stats:
            for i, metric in enumerate(metrics):
                # Plot sparsity over timepoints
                sparsity_data = [tp_stats[tp]['mean_sparsity'] for tp in tp_results['timepoints'] 
                               if tp in tp_stats]
                
                fig.add_trace(
                    go.Scatter(
                        x=tp_results['timepoints'],
                        y=sparsity_data,
                        mode='lines+markers',
                        name='Sparsity',
                        line=dict(color='gray', dash='dash'),
                        showlegend=i == 0
                    ),
                    row=3, col=i+1
                )
        
        # Row 4: Violin plots (if requested and data available)
        if include_violins and hasattr(self, '_last_detailed_results'):
            tp_raw_data = self._last_detailed_results['timepoint_analysis']['raw_similarities']
            
            if not tp_raw_data.empty:
                for i, metric in enumerate(metrics):
                    metric_data = tp_raw_data[tp_raw_data['metric'] == metric]
                    
                    if not metric_data.empty:
                        # Create violin plot data for this metric
                        timepoint_order = ['TDR126', 'TDR127', 'TDR128', 'TDR118', 'TDR125', 'TDR124']
                        hpf_labels = {'TDR126': '10hpf', 'TDR127': '12hpf', 'TDR128': '14hpf', 
                                     'TDR118': '16hpf', 'TDR125': '19hpf', 'TDR124': '24hpf'}
                        
                        for j, tp in enumerate(timepoint_order):
                            if tp in metric_data['timepoint'].values:
                                tp_metric_data = metric_data[metric_data['timepoint'] == tp]
                                
                                fig.add_trace(
                                    go.Violin(
                                        y=tp_metric_data['similarity'],
                                        x=[hpf_labels[tp]] * len(tp_metric_data),
                                        name=hpf_labels[tp],
                                        box_visible=True,
                                        meanline_visible=True,
                                        fillcolor=colors[j % len(colors)],
                                        opacity=0.5,
                                        showlegend=False,
                                        side='positive' if j % 2 == 0 else 'negative'
                                    ),
                                    row=4, col=i+1
                                )
        
        # Update layout
        for i in range(n_metrics):
            fig.update_xaxes(title_text="Timepoints", tickangle=45, row=1, col=i+1)
            fig.update_xaxes(title_text="Celltypes (Sorted by Similarity)", tickangle=45, row=2, col=i+1)
            fig.update_xaxes(title_text="Timepoints", tickangle=45, row=3, col=i+1)
            if include_violins:
                fig.update_xaxes(title_text="Timepoints", tickangle=45, row=4, col=i+1)
            
            fig.update_yaxes(title_text="Similarity", row=1, col=i+1)
            fig.update_yaxes(title_text="Similarity", row=2, col=i+1)
            fig.update_yaxes(title_text="Sparsity", row=3, col=i+1)
            if include_violins:
                fig.update_yaxes(title_text="Similarity Distribution", row=4, col=i+1)
        
        fig.update_layout(
            height=300 * n_rows,
            width=400 * n_metrics,
            title_text="GRN Similarity Analysis: Superset Approach with Multiple Metrics<br><sub>Error bars show Standard Error of the Mean (SEM)</sub>",
            showlegend=True,
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(showgrid=False)
        
        # Save files based on save_path parameter
        if save_path == True or save_individual_panels:
            self._save_individual_panels(fig, tp_results, ct_results, metrics, sort_celltypes)
        elif isinstance(save_path, str):
            fig.write_html(save_path)
        
        return fig
    
    def _save_individual_panels(self, main_fig, tp_results, ct_results, metrics, sort_celltypes):
        """
        Save individual panels as separate PDF files using matplotlib
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Set publication-quality style
        plt.rcParams.update({
            'figure.dpi': 300,
            'savefig.dpi': 300,
            'font.size': 10,
            'axes.labelsize': 12,
            'axes.titlesize': 14,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10
        })
        
        # Color palette
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        hpf_labels = {'TDR126': '10hpf', 'TDR127': '12hpf', 'TDR128': '14hpf', 
                     'TDR118': '16hpf', 'TDR125': '19hpf', 'TDR124': '24hpf'}
        
        print("💾 Saving individual panels as PDF files...")
        
        # Panel 1: Timepoint trends (all metrics together)
        fig1, ax1 = plt.subplots(1, 1, figsize=(8, 6))
        
        for i, metric in enumerate(metrics):
            if f'{metric}_mean' in tp_results.columns:
                x_vals = [hpf_labels.get(tp, tp) for tp in tp_results['timepoints']]
                y_vals = tp_results[f'{metric}_mean']
                err_vals = tp_results[f'{metric}_sem'] if f'{metric}_sem' in tp_results.columns else None
                
                ax1.errorbar(x_vals, y_vals, yerr=err_vals, 
                           marker='o', linewidth=2, markersize=6,
                           label=metric.capitalize(), color=colors[i % len(colors)])
        
        ax1.set_xlabel('Developmental Stage')
        ax1.set_ylabel('Mean Similarity Score')
        ax1.set_title('GRN Similarity Trends Across Development\n(Celltype Similarity Within Timepoints)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('grn_timepoint_trends.pdf', bbox_inches='tight')
        plt.close()
        
        # Panel 2: Individual metric timepoint trends
        for i, metric in enumerate(metrics):
            if f'{metric}_mean' in tp_results.columns:
                fig2, ax2 = plt.subplots(1, 1, figsize=(8, 6))
                
                x_vals = [hpf_labels.get(tp, tp) for tp in tp_results['timepoints']]
                y_vals = tp_results[f'{metric}_mean']
                err_vals = tp_results[f'{metric}_sem'] if f'{metric}_sem' in tp_results.columns else None
                
                ax2.errorbar(x_vals, y_vals, yerr=err_vals,
                           marker='o', linewidth=3, markersize=8,
                           color=colors[i % len(colors)], capsize=5)
                
                ax2.set_xlabel('Developmental Stage')
                ax2.set_ylabel(f'{metric.capitalize()} Similarity')
                ax2.set_title(f'{metric.capitalize()} Similarity Across Development')
                ax2.grid(True, alpha=0.3)
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(f'grn_{metric}_timepoint_trend.pdf', bbox_inches='tight')
                plt.close()
        
    def _save_individual_panels(self, main_fig, tp_results, ct_results, metrics, sort_celltypes):
        """
        Save individual panels as separate PDF files using matplotlib
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Set publication-quality style - clean white background
        plt.rcParams.update({
            'figure.dpi': 300,
            'savefig.dpi': 300,
            'font.size': 10,
            'axes.labelsize': 12,
            'axes.titlesize': 14,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'figure.facecolor': 'white',
            'axes.facecolor': 'white',
            'axes.grid': False,  # No grids
            'axes.spines.top': False,
            'axes.spines.right': False
        })
        
        # Color palette
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        hpf_labels = {'TDR126': '10hpf', 'TDR127': '12hpf', 'TDR128': '14hpf', 
                     'TDR118': '16hpf', 'TDR125': '19hpf', 'TDR124': '24hpf'}
        
        print("💾 Saving individual panels as PDF files...")
        
        # Panel 1: Timepoint trends (all metrics together)
        fig1, ax1 = plt.subplots(1, 1, figsize=(8, 6), facecolor='white')
        ax1.set_facecolor('white')
        
        for i, metric in enumerate(metrics):
            if f'{metric}_mean' in tp_results.columns:
                x_vals = [hpf_labels.get(tp, tp) for tp in tp_results['timepoints']]
                y_vals = tp_results[f'{metric}_mean']
                err_vals = tp_results[f'{metric}_sem'] if f'{metric}_sem' in tp_results.columns else None
                
                ax1.errorbar(x_vals, y_vals, yerr=err_vals, 
                           marker='o', linewidth=2, markersize=6,
                           label=metric.capitalize(), color=colors[i % len(colors)],
                           capsize=4, capthick=1.5)
        
        ax1.set_xlabel('Developmental Stage')
        ax1.set_ylabel('Mean Similarity Score')
        ax1.set_title('GRN Similarity Trends Across Development\n(Celltype Similarity Within Timepoints)')
        ax1.legend(frameon=False)
        ax1.grid(False)  # No grid
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('grn_timepoint_trends.pdf', bbox_inches='tight', facecolor='white')
        plt.close()
        
        # Panel 2: Individual metric timepoint trends
        for i, metric in enumerate(metrics):
            if f'{metric}_mean' in tp_results.columns:
                fig2, ax2 = plt.subplots(1, 1, figsize=(8, 6), facecolor='white')
                ax2.set_facecolor('white')
                
                x_vals = [hpf_labels.get(tp, tp) for tp in tp_results['timepoints']]
                y_vals = tp_results[f'{metric}_mean']
                err_vals = tp_results[f'{metric}_sem'] if f'{metric}_sem' in tp_results.columns else None
                
                ax2.errorbar(x_vals, y_vals, yerr=err_vals,
                           marker='o', linewidth=3, markersize=8,
                           color=colors[i % len(colors)], capsize=5, capthick=2)
                
                ax2.set_xlabel('Developmental Stage')
                ax2.set_ylabel(f'{metric.capitalize()} Similarity')
                ax2.set_title(f'{metric.capitalize()} Similarity Across Development')
                ax2.grid(False)  # No grid
                ax2.spines['top'].set_visible(False)
                ax2.spines['right'].set_visible(False)
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(f'grn_{metric}_timepoint_trend.pdf', bbox_inches='tight', facecolor='white')
                plt.close()
        
        # Panel 3: Celltype similarity (sorted) with SEM
        for i, metric in enumerate(metrics):
            if f'{metric}_mean' in ct_results.columns:
                fig3, ax3 = plt.subplots(1, 1, figsize=(12, 6), facecolor='white')
                ax3.set_facecolor('white')
                
                # Sort celltypes by similarity if requested
                if sort_celltypes:
                    ct_sorted = ct_results.copy().sort_values(f'{metric}_mean', ascending=False)
                else:
                    ct_sorted = ct_results.copy()
                
                x_vals = range(len(ct_sorted))
                y_vals = ct_sorted[f'{metric}_mean']
                # Use SEM for error bars
                if f'{metric}_sem' in ct_sorted.columns:
                    err_vals = ct_sorted[f'{metric}_sem']
                elif f'{metric}_std' in ct_sorted.columns:
                    # Calculate SEM from std if not available
                    counts = ct_sorted.get(f'{metric}_count', pd.Series([1]*len(ct_sorted)))
                    err_vals = ct_sorted[f'{metric}_std'] / np.sqrt(counts)
                else:
                    err_vals = None
                
                celltype_names = ct_sorted['celltypes']
                
                ax3.errorbar(x_vals, y_vals, yerr=err_vals,
                           marker='o', linewidth=0, markersize=6,
                           color=colors[i % len(colors)], capsize=3,
                           linestyle='none')
                
                ax3.set_xlabel('Cell Types (Sorted by Temporal Variability)')
                ax3.set_ylabel(f'{metric.capitalize()} Similarity')
                ax3.set_title(f'{metric.capitalize()} Temporal Dynamics by Cell Type\n(High = Stable, Low = Dynamic)')
                ax3.set_xticks(x_vals)
                ax3.set_xticklabels(celltype_names, rotation=45, ha='right')
                ax3.grid(False)  # No grid
                ax3.spines['top'].set_visible(False)
                ax3.spines['right'].set_visible(False)
                plt.tight_layout()
                plt.savefig(f'grn_{metric}_celltype_sorted.pdf', bbox_inches='tight', facecolor='white')
                plt.close()
        
        # Panel 4: Violin plots (if detailed results available)
        if hasattr(self, '_last_detailed_results'):
            tp_raw_data = self._last_detailed_results['timepoint_analysis']['raw_similarities']
            
            if not tp_raw_data.empty:
                for i, metric in enumerate(metrics):
                    metric_data = tp_raw_data[tp_raw_data['metric'] == metric]
                    
                    if not metric_data.empty:
                        fig4, ax4 = plt.subplots(1, 1, figsize=(10, 6), facecolor='white')
                        ax4.set_facecolor('white')
                        
                        # Prepare data for seaborn
                        plot_data = metric_data.copy()
                        plot_data['hpf'] = plot_data['timepoint'].map(hpf_labels)
                        
                        sns.violinplot(data=plot_data, x='hpf', y='similarity', 
                                     palette='viridis', inner='box', ax=ax4)
                        
                        ax4.set_xlabel('Developmental Stage')
                        ax4.set_ylabel(f'{metric.capitalize()} Similarity')
                        ax4.set_title(f'{metric.capitalize()} Similarity Distributions\n(All Celltype Pairs Within Each Timepoint)')
                        ax4.grid(False)  # No grid
                        ax4.spines['top'].set_visible(False)
                        ax4.spines['right'].set_visible(False)
                        plt.xticks(rotation=45)
                        plt.tight_layout()
                        plt.savefig(f'grn_{metric}_violin_distributions.pdf', bbox_inches='tight', facecolor='white')
                        plt.close()
        
        print("✅ Individual panels saved as PDF files:")
        print("  - grn_timepoint_trends.pdf (all metrics combined)")
        for metric in metrics:
            print(f"  - grn_{metric}_timepoint_trend.pdf (individual metric)")
            print(f"  - grn_{metric}_celltype_sorted.pdf (sorted celltypes)")
            if hasattr(self, '_last_detailed_results'):
                print(f"  - grn_{metric}_violin_distributions.pdf (violin plots)")
    
    def plot_similarity_violins(self, detailed_results=None, metrics=['pearson', 'cosine', 'spearman'], 
                               save_path=None, lineage_groups=None):
        """
        Create violin plots showing similarity distributions
        
        Parameters:
        -----------
        detailed_results : dict, optional
            Results from compute_detailed_similarities(). If None, will compute automatically
        metrics : list
            Metrics to plot
        save_path : str, optional
            Path to save HTML file
        lineage_groups : dict, optional
            Lineage groupings for coloring
        """
        
        # If no detailed results provided, compute them
        if detailed_results is None:
            print("🔍 Computing detailed similarities for violin plots...")
            detailed_results = self.compute_detailed_similarities(metrics=metrics, lineage_groups=lineage_groups)
        
        tp_raw_data = detailed_results['timepoint_analysis']['raw_similarities']
        
        if tp_raw_data.empty:
            print("❌ No timepoint data available for violin plots")
            return None
        
        # Create violin plots
        n_metrics = len(metrics)
        fig = make_subplots(
            rows=2, cols=n_metrics,
            subplot_titles=[f'{m.capitalize()} - Similarity Distributions by Timepoint' for m in metrics] +
                          [f'{m.capitalize()} - Similarity by Lineage Interactions' for m in metrics],
            specs=[[{"secondary_y": False} for _ in range(n_metrics)],
                   [{"secondary_y": False} for _ in range(n_metrics)]],
            vertical_spacing=0.12
        )
        
        colors = ['blue', 'red', 'green', 'purple', 'orange', 'brown']
        timepoint_order = ['TDR126', 'TDR127', 'TDR128', 'TDR118', 'TDR125', 'TDR124']
        hpf_labels = {'TDR126': '10hpf', 'TDR127': '12hpf', 'TDR128': '14hpf', 
                     'TDR118': '16hpf', 'TDR125': '19hpf', 'TDR124': '24hpf'}
        
        # Row 1: Violin plots by timepoint
        for i, metric in enumerate(metrics):
            metric_data = tp_raw_data[tp_raw_data['metric'] == metric]
            
            if not metric_data.empty:
                for j, tp in enumerate(timepoint_order):
                    if tp in metric_data['timepoint'].values:
                        tp_metric_data = metric_data[metric_data['timepoint'] == tp]
                        
                        fig.add_trace(
                            go.Violin(
                                y=tp_metric_data['similarity'],
                                x=[hpf_labels[tp]] * len(tp_metric_data),
                                name=hpf_labels[tp],
                                box_visible=True,
                                meanline_visible=True,
                                fillcolor=colors[j % len(colors)],
                                opacity=0.7,
                                showlegend=(i == 0),
                                points='outliers'  # Show outlier points
                            ),
                            row=1, col=i+1
                        )
        
        # Row 2: Violin plots by lineage interactions (if lineage data available)
        if 'lineage_pair_type' in tp_raw_data.columns:
            for i, metric in enumerate(metrics):
                metric_data = tp_raw_data[tp_raw_data['metric'] == metric]
                
                if not metric_data.empty:
                    # Get unique lineage pair types
                    lineage_types = sorted(metric_data['lineage_pair_type'].unique())
                    lineage_colors = px.colors.qualitative.Set3
                    
                    for j, lineage_type in enumerate(lineage_types):
                        lineage_data = metric_data[metric_data['lineage_pair_type'] == lineage_type]
                        
                        fig.add_trace(
                            go.Violin(
                                y=lineage_data['similarity'],
                                x=[lineage_type] * len(lineage_data),
                                name=lineage_type,
                                box_visible=True,
                                meanline_visible=True,
                                fillcolor=lineage_colors[j % len(lineage_colors)],
                                opacity=0.7,
                                showlegend=(i == 0),
                                points='outliers'
                            ),
                            row=2, col=i+1
                        )
        
        # Update layout
        for i in range(n_metrics):
            fig.update_xaxes(title_text="Developmental Stage", tickangle=45, row=1, col=i+1)
            fig.update_xaxes(title_text="Lineage Interaction Type", tickangle=45, row=2, col=i+1)
            
            fig.update_yaxes(title_text="Similarity Score", row=1, col=i+1)
            fig.update_yaxes(title_text="Similarity Score", row=2, col=i+1)
        
        fig.update_layout(
            height=800,
            width=400 * n_metrics,
            title_text="GRN Similarity Distributions: Violin Plot Analysis<br><sub>Showing full distributions and outliers</sub>",
            showlegend=True,
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(showgrid=False)
        
        if save_path:
            fig.write_html(save_path)
        
        fig.show()
        return fig

# Usage example with superset approach:
"""
# Initialize the analyzer
analyzer = GRNSimilarityAnalyzer(dict_filtered_GRNs)

# Analyze with multiple metrics
metrics = ['pearson', 'cosine', 'jaccard']

# Analyze celltype similarity within timepoints (your original analysis)
tp_results, tp_stats = analyzer.analyze_timepoint_similarity(metrics=metrics)

# Analyze timepoint similarity within celltypes (temporal dynamics)
ct_results, ct_stats = analyzer.analyze_celltype_similarity(metrics=metrics)

# Create comprehensive plots
fig = analyzer.plot_similarity_trends(tp_results, ct_results, tp_stats, ct_stats, 
                                     metrics=metrics, save_path='grn_similarity_superset.html')
fig.show()

# Display results
print("\n" + "="*60)
print("TIMEPOINT ANALYSIS RESULTS:")
print("="*60)
print(tp_results.round(3))

print("\n" + "="*60) 
print("CELLTYPE ANALYSIS RESULTS:")
print("="*60)
print(ct_results.round(3))
"""
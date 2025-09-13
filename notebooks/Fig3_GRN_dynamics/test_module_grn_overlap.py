"""
Test script for the updated GRN overlap quantification module

This script creates sample data matching your dict_filtered_GRNs structure
and tests the updated functions.
"""

import pandas as pd
import numpy as np
from module_grn_overlap_quant import (
    extract_tf_gene_pairs, 
    compute_timepoint_presence_fractions,
    compute_celltype_presence_fractions,
    analyze_grn_overlap
)

# Create sample data matching your structure
def create_sample_grn_data():
    """Create sample data matching dict_filtered_GRNs structure"""
    
    # Sample TF-target pairs with realistic gene names
    sample_pairs = [
        ('hmga1a', 'rrm2'), ('nr2f5', 'agrn'), ('msx1b', 'cdh6'), 
        ('sox5', 'sox6'), ('tbx16', 'itm2cb'), ('hoxc6b', 'hoxc3a'),
        ('meis1b', 'tenm4'), ('rarga', 'cdx4'), ('uncx', 'comp'),
        ('sox11a', 'notch1a'), ('meis2a', 'ncam1a'), ('foxp1b', 'robo1'),
        ('atoh1a', 'tenm3'), ('fli1a', 'apoeb'), ('foxd3', 'hsp90ab1'),
        ('hnf1bb', 'ednrab'), ('barx1', 'nrp2b')
    ]
    
    timepoints = ['TDR118', 'TDR120', 'TDR122']
    celltypes = ['NMPs', 'PSM', 'Neural', 'Mesoderm']
    
    dict_filtered_grns = {}
    
    for tp in timepoints:
        dict_filtered_grns[tp] = {}
        for ct in celltypes:
            # Create random subset of pairs for each timepoint/celltype
            n_pairs = np.random.randint(8, 15)  # Random number of pairs
            selected_pairs = np.random.choice(len(sample_pairs), size=n_pairs, replace=False)
            
            grn_data = []
            for i in selected_pairs:
                source, target = sample_pairs[i]
                grn_data.append({
                    'source': source,
                    'target': target,
                    'coef_mean': np.random.uniform(0.02, 0.2),
                    'coef_abs': np.random.uniform(0.02, 0.2),
                    'p': np.random.uniform(1e-20, 1e-5),
                    '-logp': np.random.uniform(5, 20)
                })
            
            dict_filtered_grns[tp][ct] = pd.DataFrame(grn_data)
    
    return dict_filtered_grns

def test_extract_tf_gene_pairs():
    """Test the updated extract_tf_gene_pairs function"""
    print("Testing extract_tf_gene_pairs function...")
    
    # Create sample GRN dataframe
    sample_grn = pd.DataFrame({
        'source': ['hmga1a', 'nr2f5', 'msx1b'],
        'target': ['rrm2', 'agrn', 'cdh6'],
        'coef_mean': [0.141, 0.133, 0.124],
        'coef_abs': [0.141, 0.133, 0.124],
        'p': [1.54e-20, 1.84e-12, 8.24e-18],
        '-logp': [19.81, 11.74, 17.08]
    })
    
    pairs = extract_tf_gene_pairs(sample_grn)
    expected_pairs = {'hmga1a_rrm2', 'nr2f5_agrn', 'msx1b_cdh6'}
    
    assert pairs == expected_pairs, f"Expected {expected_pairs}, got {pairs}"
    print("✓ extract_tf_gene_pairs test passed")
    
    # Test empty dataframe
    empty_pairs = extract_tf_gene_pairs(pd.DataFrame())
    assert empty_pairs == set(), f"Expected empty set, got {empty_pairs}"
    print("✓ Empty dataframe test passed")

def test_full_analysis():
    """Test the complete analysis workflow"""
    print("\nTesting complete analysis workflow...")
    
    # Create sample data
    dict_filtered_grns = create_sample_grn_data()
    
    print(f"Created sample data with {len(dict_filtered_grns)} timepoints")
    for tp, ct_dict in dict_filtered_grns.items():
        print(f"  {tp}: {len(ct_dict)} celltypes")
        for ct, grn_df in ct_dict.items():
            print(f"    {ct}: {len(grn_df)} TF-target pairs")
    
    # Test individual functions
    print("\nTesting timepoint presence fractions...")
    tp_presence = compute_timepoint_presence_fractions(dict_filtered_grns)
    print(f"✓ Computed presence fractions for {len(tp_presence)} timepoints")
    
    print("\nTesting celltype presence fractions...")
    ct_presence = compute_celltype_presence_fractions(dict_filtered_grns)
    print(f"✓ Computed presence fractions for {len(ct_presence)} celltypes")
    
    # Test full analysis
    print("\nTesting full analysis...")
    results = analyze_grn_overlap(dict_filtered_grns, verbose=False)
    print(f"✓ Full analysis completed successfully")
    print(f"  Results keys: {list(results.keys())}")
    
    return results

def test_filtering_functionality():
    """Test the filtering functionality with exclude_groups parameter"""
    print("\nTesting filtering functionality...")
    
    # Create sample data
    dict_filtered_grns = create_sample_grn_data()
    
    # Create exclude list - exclude some timepoint-celltype combinations
    exclude_groups = ['TDR118_Neural', 'TDR120_PSM', 'TDR122_Mesoderm']
    print(f"Excluding groups: {exclude_groups}")
    
    # Test timepoint analysis with filtering
    print("\nTesting timepoint analysis with filtering...")
    tp_presence = compute_timepoint_presence_fractions(dict_filtered_grns, exclude_groups)
    
    # Verify exclusions worked
    for tp in tp_presence.keys():
        excluded_celltypes_for_tp = [grp.split('_')[1] for grp in exclude_groups if grp.startswith(tp)]
        valid_celltypes = tp_presence[tp]['valid_celltypes']
        for excluded_ct in excluded_celltypes_for_tp:
            assert excluded_ct not in valid_celltypes, f"Expected {excluded_ct} to be excluded from {tp}"
    
    print("✓ Timepoint filtering working correctly")
    
    # Test celltype analysis with filtering  
    print("\nTesting celltype analysis with filtering...")
    ct_presence = compute_celltype_presence_fractions(dict_filtered_grns, exclude_groups)
    
    # Verify exclusions worked
    for ct in ct_presence.keys():
        excluded_timepoints_for_ct = [grp.split('_')[0] for grp in exclude_groups if grp.endswith(ct)]
        valid_timepoints = ct_presence[ct]['valid_timepoints']
        for excluded_tp in excluded_timepoints_for_ct:
            assert excluded_tp not in valid_timepoints, f"Expected {excluded_tp} to be excluded from {ct}"
    
    print("✓ Celltype filtering working correctly")
    
    # Test complete analysis with filtering
    print("\nTesting complete analysis with filtering...")
    results = analyze_grn_overlap(dict_filtered_grns, exclude_groups, verbose=False)
    print("✓ Complete analysis with filtering successful")
    
    return results

if __name__ == "__main__":
    print("Testing updated GRN overlap quantification module")
    print("=" * 50)
    
    # Test individual function
    test_extract_tf_gene_pairs()
    
    # Test full workflow
    results = test_full_analysis()
    
    # Test filtering functionality
    filtered_results = test_filtering_functionality()
    
    print("\n" + "=" * 50)
    print("All tests passed! The module is ready to use with your data structure.")
    print("\nTo use with your actual data:")
    print("from module_grn_overlap_quant import complete_grn_overlap_analysis")
    print("results = complete_grn_overlap_analysis(dict_filtered_GRNs)")
    print("\nTo exclude groups with too few cells:")
    print("exclude_list = ['TDR124_epidermis', 'TDR118_tail_bud', ...]")
    print("results = complete_grn_overlap_analysis(dict_filtered_GRNs, exclude_groups=exclude_list)")
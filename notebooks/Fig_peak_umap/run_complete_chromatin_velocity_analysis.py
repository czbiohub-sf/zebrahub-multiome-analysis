#!/usr/bin/env python3
"""
Complete Chromatin Velocity Analysis and Visualization Pipeline

This script performs the complete workflow:
1. Computes chromatin velocity using the optimized pipeline
2. Visualizes velocity vectors on peak UMAP embeddings  
3. Generates comprehensive analysis and validation plots
4. Saves all results and creates publication-ready figures

Author: Zebrahub-Multiome Analysis Pipeline
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add current directory to path for imports
sys.path.append('.')
sys.path.append('../scripts')

print("=== Complete Chromatin Velocity Analysis Pipeline ===")
print("This pipeline will:")
print("1. Compute chromatin velocity from accessibility and co-accessibility data")
print("2. Create comprehensive visualizations on peak UMAP embeddings")
print("3. Generate analysis reports and validation plots")
print("4. Save publication-ready figures")

def main():
    """Run the complete chromatin velocity analysis pipeline."""
    
    # =====================================================================
    # STEP 1: COMPUTE CHROMATIN VELOCITY (if not already done)
    # =====================================================================
    
    velocity_results_file = "chromatin_velocity_results.h5ad"
    
    if not Path(velocity_results_file).exists():
        print("\n" + "="*60)
        print("STEP 1: COMPUTING CHROMATIN VELOCITY")
        print("="*60)
        
        # Import and run the optimized chromatin velocity computation
        try:
            exec(open('run_chromatin_velocity_optimized.py').read())
            print("✓ Chromatin velocity computation completed successfully!")
        except Exception as e:
            print(f"Error during velocity computation: {e}")
            print("Please ensure run_chromatin_velocity_optimized.py is available and functional.")
            return False
    else:
        print(f"\n✓ Found existing chromatin velocity results: {velocity_results_file}")
    
    # =====================================================================
    # STEP 2: LOAD PEAK UMAP DATA AND CREATE VISUALIZATIONS  
    # =====================================================================
    
    print("\n" + "="*60)
    print("STEP 2: CREATING VELOCITY VISUALIZATIONS ON PEAK UMAP")
    print("="*60)
    
    # Check for required files
    umap_coords_file = "peak_umap_3d_annotated_v6.csv"
    
    if not Path(umap_coords_file).exists():
        print(f"Error: Peak UMAP coordinates file not found: {umap_coords_file}")
        print("Looking for alternative UMAP coordinate files...")
        
        # Search for alternative UMAP files
        umap_candidates = list(Path('.').glob('peak_umap_*annotated*.csv'))
        if umap_candidates:
            umap_coords_file = str(umap_candidates[0])
            print(f"Using alternative UMAP file: {umap_coords_file}")
        else:
            print("No UMAP coordinate files found. Cannot proceed with visualization.")
            return False
    
    # Import the visualization module
    try:
        from peak_umap_velocity_visualizer import PeakUMAPVelocityVisualizer, create_example_visualization
        print("✓ Imported velocity visualizer successfully!")
    except ImportError as e:
        print(f"Error importing visualizer: {e}")
        return False
    
    # =====================================================================
    # STEP 3: CREATE COMPREHENSIVE VISUALIZATIONS
    # =====================================================================
    
    print("\n" + "="*60)
    print("STEP 3: GENERATING COMPREHENSIVE VISUALIZATIONS")
    print("="*60)
    
    # Create output directory
    output_dir = "chromatin_velocity_analysis_results"
    Path(output_dir).mkdir(exist_ok=True)
    print(f"Results will be saved to: {output_dir}")
    
    try:
        # Run the complete visualization pipeline
        visualizer = create_example_visualization(
            umap_path=umap_coords_file,
            velocity_path=velocity_results_file,
            output_dir=output_dir
        )
        print("✓ Basic visualizations completed successfully!")
        
    except Exception as e:
        print(f"Error during visualization: {e}")
        print("Attempting manual visualization setup...")
        
        # Manual setup if automated function fails
        try:
            visualizer = PeakUMAPVelocityVisualizer(umap_coords_file, velocity_results_file)
            print("✓ Visualizer initialized successfully!")
            
            # Create individual plots
            print("Creating velocity vector plot...")
            fig1 = visualizer.plot_velocity_vectors_on_umap(
                color_by='velocity_magnitude',
                title="Chromatin Velocity Vectors on Peak UMAP", 
                save_path=f"{output_dir}/velocity_vectors_main.png"
            )
            plt.close(fig1)
            
            print("Creating velocity heatmap...")
            fig2 = visualizer.create_velocity_heatmap_on_umap(
                save_path=f"{output_dir}/velocity_heatmap.png"
            )
            plt.close(fig2)
            
            print("✓ Manual visualization completed successfully!")
            
        except Exception as e2:
            print(f"Error during manual visualization: {e2}")
            return False
    
    # =====================================================================
    # STEP 4: ADVANCED ANALYSIS AND VALIDATION
    # =====================================================================
    
    print("\n" + "="*60) 
    print("STEP 4: ADVANCED ANALYSIS AND VALIDATION")
    print("="*60)
    
    try:
        # Analyze high-velocity peaks
        print("Analyzing high-velocity peaks...")
        top_peaks, analysis_results = visualizer.analyze_high_velocity_peaks(top_n=200)
        
        # Save detailed results
        top_peaks.to_csv(f"{output_dir}/high_velocity_peaks_detailed.csv", index=False)
        
        # Create validation plots
        print("Creating validation plots...")
        
        # Velocity magnitude distribution
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Velocity magnitude histogram
        axes[0, 0].hist(visualizer.integrated_data['velocity_magnitude'], bins=50, alpha=0.7, edgecolor='black')
        axes[0, 0].set_xlabel('Velocity Magnitude')
        axes[0, 0].set_ylabel('Number of Peaks')
        axes[0, 0].set_title('Distribution of Velocity Magnitudes')
        
        # Plot 2: Velocity components scatter
        axes[0, 1].scatter(visualizer.integrated_data['velocity_x'], 
                          visualizer.integrated_data['velocity_y'], 
                          alpha=0.5, s=1)
        axes[0, 1].set_xlabel('Velocity X Component')
        axes[0, 1].set_ylabel('Velocity Y Component')
        axes[0, 1].set_title('Velocity Vector Components')
        
        # Plot 3: UMAP 1 vs Velocity
        axes[1, 0].scatter(visualizer.integrated_data['UMAP_1'], 
                          visualizer.integrated_data['velocity_magnitude'], 
                          alpha=0.5, s=1)
        axes[1, 0].set_xlabel('UMAP 1')
        axes[1, 0].set_ylabel('Velocity Magnitude')
        axes[1, 0].set_title('UMAP 1 vs Velocity Magnitude')
        
        # Plot 4: UMAP 2 vs Velocity
        axes[1, 1].scatter(visualizer.integrated_data['UMAP_2'], 
                          visualizer.integrated_data['velocity_magnitude'], 
                          alpha=0.5, s=1)
        axes[1, 1].set_xlabel('UMAP 2')
        axes[1, 1].set_ylabel('Velocity Magnitude')
        axes[1, 1].set_title('UMAP 2 vs Velocity Magnitude')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/validation_plots.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✓ Validation plots created successfully!")
        
    except Exception as e:
        print(f"Warning: Advanced analysis failed: {e}")
        print("Basic visualization completed, but advanced analysis unavailable.")
    
    # =====================================================================
    # STEP 5: GENERATE SUMMARY REPORT
    # =====================================================================
    
    print("\n" + "="*60)
    print("STEP 5: GENERATING SUMMARY REPORT")
    print("="*60)
    
    try:
        # Generate summary statistics
        summary_stats = {
            'total_peaks_analyzed': len(visualizer.integrated_data),
            'mean_velocity_magnitude': visualizer.integrated_data['velocity_magnitude'].mean(),
            'std_velocity_magnitude': visualizer.integrated_data['velocity_magnitude'].std(),
            'min_velocity_magnitude': visualizer.integrated_data['velocity_magnitude'].min(),
            'max_velocity_magnitude': visualizer.integrated_data['velocity_magnitude'].max(),
            'high_velocity_peaks_threshold': visualizer.integrated_data['velocity_magnitude'].quantile(0.9),
            'num_high_velocity_peaks': (visualizer.integrated_data['velocity_magnitude'] > 
                                       visualizer.integrated_data['velocity_magnitude'].quantile(0.9)).sum()
        }
        
        # Create summary report
        report_path = f"{output_dir}/chromatin_velocity_analysis_report.txt"
        with open(report_path, 'w') as f:
            f.write("CHROMATIN VELOCITY ANALYSIS REPORT\n")
            f.write("="*50 + "\n\n")
            f.write("SUMMARY STATISTICS\n")
            f.write("-"*20 + "\n")
            
            for key, value in summary_stats.items():
                if isinstance(value, float):
                    f.write(f"{key}: {value:.4f}\n")
                else:
                    f.write(f"{key}: {value}\n")
            
            f.write(f"\nFILES GENERATED\n")
            f.write("-"*15 + "\n")
            output_files = list(Path(output_dir).glob('*'))
            for file_path in sorted(output_files):
                f.write(f"- {file_path.name}\n")
            
            f.write(f"\nDATA INTEGRATION SUMMARY\n")
            f.write("-"*24 + "\n")
            f.write(f"Peak UMAP coordinates: {umap_coords_file}\n")
            f.write(f"Velocity results: {velocity_results_file}\n")
            f.write(f"Successfully integrated: {len(visualizer.integrated_data)} peaks\n")
            
        print(f"✓ Summary report saved to: {report_path}")
        
        # Print summary to console
        print(f"\nANALYSIS COMPLETE!")
        print(f"Total peaks analyzed: {summary_stats['total_peaks_analyzed']}")
        print(f"Mean velocity magnitude: {summary_stats['mean_velocity_magnitude']:.4f}")
        print(f"High-velocity peaks (>90th percentile): {summary_stats['num_high_velocity_peaks']}")
        print(f"Results directory: {output_dir}")
        
        return True
        
    except Exception as e:
        print(f"Warning: Report generation failed: {e}")
        print("Analysis completed but summary report unavailable.")
        return True
    
if __name__ == "__main__":
    success = main()
    
    if success:
        print("\n" + "="*60)
        print("✓ CHROMATIN VELOCITY ANALYSIS PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("\nNext steps:")
        print("1. Review the generated visualizations")
        print("2. Examine high-velocity peaks for biological relevance")
        print("3. Compare velocity patterns across developmental timepoints")
        print("4. Integrate findings with gene regulatory network analysis")
    else:
        print("\n" + "="*60)
        print("✗ CHROMATIN VELOCITY ANALYSIS PIPELINE FAILED")
        print("="*60)
        print("Please check error messages above and ensure all required files are available.")
        
    print(f"\nFor questions or issues, please refer to the documentation in:")
    print(f"- chromatin_velocity_development.py")  
    print(f"- peak_umap_velocity_visualizer.py")
    print(f"- CLAUDE.md")
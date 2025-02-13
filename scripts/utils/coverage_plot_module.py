import os
import pandas as pd
import numpy as np
import pyranges as pr
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
import seaborn as sns
import tabix
import subprocess
from typing import Dict, Optional, Union

class CoveragePlotter:
    """Class for generating genomic coverage plots from fragment files."""
    
    def __init__(self):
        """Initialize the CoveragePlotter."""
        # Set default plotting style
        sns.set_style('ticks')
        plt.rcParams['figure.dpi'] = 100
        plt.rcParams['image.cmap'] = 'Spectral_r'
        plt.rcParams["font.family"] = "Helvetica"

    def compute_coverage(self, 
                        fragments_files: Dict[str, str],
                        region: str,
                        barcodes: Dict[str, list],
                        out_prefix: str,
                        smooth: Optional[int] = None,
                        normalize: bool = False,
                        frag_type: str = 'All') -> pd.Series:
        """
        Compute coverage for a genomic region from fragment files.
        
        Args:
            fragments_files: Dict mapping sample names to fragment file paths
            region: Genomic region string (e.g. "chr1:1000-2000")
            barcodes: Dict mapping sample names to lists of cell barcodes
            out_prefix: Prefix for temporary files
            smooth: Window size for smoothing coverage
            normalize: Whether to normalize coverage
            frag_type: Type of fragments to use ('All', 'NFR', or 'NUC')
            
        Returns:
            pd.Series containing coverage values
        """
        # Write fragments to temporary BED file
        bed_file = open(out_prefix + '.bed', 'w')
        
        for sample in fragments_files:
            tb = tabix.open(fragments_files[sample])
            records = tb.querys(region)
            
            for record in records:
                if record[3] in barcodes[sample]:
                    # Filter by fragment type if specified
                    if frag_type == 'NFR' and int(record[2]) - int(record[1]) > 145:
                        continue
                    if frag_type == 'NUC' and int(record[2]) - int(record[1]) <= 145:
                        continue
                        
                    line = f"{record[0]}\t{record[1]}\t{record[2]}\n"
                    bed_file.writelines(line)
                    
        bed_file.close()

        # Create region BED file
        region_bed = open(out_prefix + '.region.bed', 'w')
        line = region.replace(':', '\t').replace('-', '\t') + '\n'
        region_bed.writelines(line)
        region_bed.close()

        # Calculate coverage
        coverage_bed = open(out_prefix + '.coverage.bed', 'w')
        args = ['bedtools', 'coverage', '-a', out_prefix + '.region.bed',
                '-b', out_prefix + '.bed', '-d']
        subprocess.call(args, stdout=coverage_bed)
        coverage_bed.close()

        # Read coverage into pandas Series
        df = pd.read_csv(out_prefix + '.coverage.bed', sep='\t', header=None)
        coverage = pd.Series(df[4].values, index=df[1] + df[3] - 1)
        coverage.attrs['chr'] = df[0][0]

        # Apply smoothing if specified
        if smooth:
            coverage = coverage.rolling(smooth).mean()
            coverage[coverage.isnull()] = coverage.iloc[smooth]

        # Apply normalization if specified  
        if normalize:
            # ... existing normalization code ...
            pass

        # Clean up temp files
        for ext in ['.bed', '.coverage.bed', '.region.bed']:
            os.unlink(out_prefix + ext)

        return coverage

    def plot_coverage(self,
                     coverage: pd.Series,
                     track_name: str = 'Coverage',
                     ax: Optional[plt.Axes] = None,
                     color: str = '#ff7f00',
                     min_coverage: float = 0,
                     ylim: Optional[tuple] = None,
                     fill: bool = True,
                     linestyle: str = '-',
                     y_font: Optional[int] = None) -> None:
        """Plot coverage track."""
        # ... existing _plot_coverage code ...
        pass

    def plot_bed(self,
                plot_peaks: pr.PyRanges,
                track_name: str = "Bed",
                ax: Optional[plt.Axes] = None,
                facecolor: str = '#ff7f00') -> None:
        """Plot BED track."""
        # ... existing _plot_bed code ...
        pass

    def plot_gene(self,
                 genes: pr.PyRanges,
                 ax: Optional[plt.Axes] = None,
                 track_name: str = 'Genes',
                 facecolor: str = '#377eb8',
                 exon_height: float = 0.9,
                 utr_height: float = 0.4) -> None:
        """Plot gene track."""
        # ... existing _plot_gene code ...
        pass

    def plot_region(self,
                   barcode_groups: pd.Series,
                   region: str,
                   fragments_files: Dict[str, str],
                   peak_groups: Optional[pd.Series] = None,
                   genes: Optional[pr.PyRanges] = None,
                   highlight_peaks: Optional[pr.PyRanges] = None,
                   min_coverage: float = 0,
                   smooth: Optional[int] = None,
                   common_scale: bool = False,
                   plot_cov_size: float = 2,
                   plot_bed_size: float = 0.75,
                   collapsed: bool = False,
                   coverage_colors: Optional[Dict[str, str]] = None,
                   fig_width: float = 15,
                   frag_type: str = 'All',
                   normalize: bool = True,
                   y_font: Optional[int] = None) -> plt.Figure:
        """
        Generate multi-track coverage plot for a genomic region.
        
        Args:
            barcode_groups: Series mapping group names to cell barcodes
            region: Genomic region string
            fragments_files: Dict mapping sample names to fragment files
            peak_groups: Optional peak annotations
            genes: Optional gene annotations
            highlight_peaks: Optional peaks to highlight
            min_coverage: Minimum coverage to display
            smooth: Window size for smoothing
            common_scale: Use common y-axis scale across tracks
            plot_cov_size: Height multiplier for coverage tracks
            plot_bed_size: Height multiplier for BED tracks
            collapsed: Plot all coverage tracks overlaid
            coverage_colors: Dict mapping groups to colors
            fig_width: Figure width
            frag_type: Fragment type filter
            normalize: Whether to normalize coverage
            y_font: Font size for y-axis labels
            
        Returns:
            matplotlib Figure object
        """
        # ... existing plot_coverage implementation ...
        pass

# # Example usage
# # Create plotter instance
# plotter = CoveragePlotter()

# # Generate coverage plot
# fig = plotter.plot_region(
#     barcode_groups=barcode_groups,
#     region="chr3:128,446,488-128,499,744",
#     fragments_files=fragment_files,
#     genes=genes,
#     highlight_peaks=highlight_peaks,
#     common_scale=True,
#     smooth=75,
#     coverage_colors=cluster_colors,
#     fig_width=8,
#     plot_cov_size=0.65,
#     plot_bed_size=0.35,
#     frag_type='NFR',
#     normalize=True,
#     y_font=8
# )

# # Save or display the plot
# fig.savefig('coverage_plot.png', dpi=600)
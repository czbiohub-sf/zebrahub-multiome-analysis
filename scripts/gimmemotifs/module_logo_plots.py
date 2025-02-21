# A module for generating sequence logo plots from the motif database
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logomaker
from gimmemotifs.motif import Motif, read_motifs
# from gimmemotifs.plot import plot_logo

class LogoPlot:
    def __init__(self, motif_file):
        """
        Initialize LogoPlot class
        
        Parameters:
        -----------
        motif_file : str
            Path to motif database file
        """
        self.motif_file = motif_file
        self.motifs = read_motifs(motif_file)

    def compute_information_content(self, pwm):
        """
        Compute Shannon information content from PWM
        """
        background = np.array([0.25, 0.25, 0.25, 0.25])  # Uniform background
        ic_matrix = pwm * np.log2(pwm / background)
        ic_matrix = np.nan_to_num(ic_matrix)  # Replace NaNs with 0
        return ic_matrix

    def generate_logo_plot(self, motif_name, output_dir=None, figsize=(12, 4)):
        """
        Generate an information content-based sequence logo plot
        """
        # Get the specific motif
        selected_motif = next(m for m in self.motifs if m.id == motif_name)
        
        # Get PWM and compute information content
        pwm = selected_motif.pwm
        ic_matrix = self.compute_information_content(pwm)
        
        # Convert to DataFrame
        df = pd.DataFrame(ic_matrix, columns=["A", "C", "G", "T"])
        
        # Create the plot
        plt.figure(figsize=figsize)
        logomaker.Logo(df)
        
        # Customize the plot
        plt.title(f"Sequence Logo for {selected_motif.id}")
        plt.ylabel("bits")
        plt.xlabel("position")
        plt.grid(False)
        
        # Save the plots if output directory is provided
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            base_path = os.path.join(output_dir, motif_name)
            plt.savefig(f"{base_path}.seq_logo.png")
            plt.savefig(f"{base_path}.seq_logo.pdf")
        
        return plt.gcf()

    def get_motif_by_name(self, motif_name):
        """
        Retrieve a specific motif by its name.
        """
        try:
            return next(m for m in self.motifs if m.id == motif_name)
        except StopIteration:
            raise ValueError(f"Motif '{motif_name}' not found in the database")

    def get_consensus_sequence(self, motif_name):
        """
        Get the consensus sequence for a specific motif.
        
        Args:
            motif_name (str): Name of the motif
            
        Returns:
            str: Consensus sequence
        """
        motif = self.get_motif_by_name(motif_name)
        return motif.to_consensus()
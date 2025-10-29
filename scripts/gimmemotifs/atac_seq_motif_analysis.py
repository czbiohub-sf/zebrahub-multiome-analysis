# module for converting peaks to fasta and finding motifs using gimmemotifs
# reference: https://gimmemotifs.readthedocs.io/en/latest/
# Most utility functions are fromCellOracle github repository (Kamimoto et al., 2023)
# link: https://github.com/morris-lab/CellOracle/blob/master/celloracle/motif_analysis/process_bed_file.py

import os
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from genomepy import Genome
from gimmemotifs.fasta import Fasta
from pybedtools import BedTool

class ATACSeqMotifAnalysis:
    def __init__(self, ref_genome, genomes_dir):
        """
        Initialize the class with reference genome information.

        Args:
            ref_genome (str): Name of the reference genome (e.g., "hg19", "mm10", "zebrafish").
            genomes_dir (str): Directory where genome data is stored.
        """
        self.ref_genome = ref_genome
        self.genomes_dir = genomes_dir
        self.genome_data = Genome(ref_genome, genomes_dir=genomes_dir)

    @staticmethod
    def decompose_chrstr(peak_str):
        """
        Split peak string into chromosome, start, and end.

        Args:
            peak_str (str): Peak string in the format "chromosome-start-end".

        Returns:
            tuple: (chromosome, start, end)
        """
        chromosome, start, end = peak_str.split("-")
        return chromosome, int(start), int(end)

    @staticmethod
    def list_peakstr_to_df(peaks):
        """
        Convert a list of peak strings into a DataFrame.

        Args:
            peaks (list): List of peak strings (e.g., ["1-100-200", "2-300-400"]).

        Returns:
            pd.DataFrame: DataFrame with columns ["chr", "start", "end"].
        """
        decomposed = [ATACSeqMotifAnalysis.decompose_chrstr(peak) for peak in peaks]
        df = pd.DataFrame(decomposed, columns=["chr", "start", "end"])
        return df

    def check_peak_format(self, peaks_df):
        """
        Validate peak format and filter invalid peaks based on genome information.

        Args:
            peaks_df (pd.DataFrame): DataFrame with columns ["chr", "start", "end"].

        Returns:
            pd.DataFrame: Filtered DataFrame with valid peaks.
        """
        n_peaks_before = peaks_df.shape[0]
        all_chr_list = list(self.genome_data.keys())

        # Check chromosome names and peak lengths
        lengths = np.abs(peaks_df["end"] - peaks_df["start"])
        n_threshold = 5
        valid_peaks = peaks_df[(lengths >= n_threshold) & peaks_df["chr"].isin(all_chr_list)]

        # Check if peaks exceed chromosome lengths
        for idx, row in peaks_df.iterrows():
            # chr_length = self.genome_data[row["chr"]].length
            chr_length = len(self.genome_data[row["chr"]])  # Fixed here
            if row["end"] > chr_length:
                valid_peaks = valid_peaks.drop(idx)

        # Print summary
        n_invalid_length = len(lengths[lengths < n_threshold])
        n_invalid_chr = n_peaks_before - peaks_df["chr"].isin(all_chr_list).sum()
        n_invalid_end = n_peaks_before - valid_peaks.shape[0]
        print(f"Peaks before filtering: {n_peaks_before}")
        print(f"Invalid chromosome names: {n_invalid_chr}")
        print(f"Invalid lengths (< {n_threshold} bp): {n_invalid_length}")
        print(f"Peaks exceeding chromosome lengths: {n_invalid_end}")
        print(f"Peaks after filtering: {valid_peaks.shape[0]}")

        return valid_peaks

    def peak_to_fasta(self, peaks):
        """
        Convert peak coordinates to a FASTA object.

        Args:
            peaks (list or pd.Series): List of peak strings (e.g., "1-100-200").

        Returns:
            Fasta: GimmeMotifs Fasta object containing DNA sequences for peaks.
        """
        def peak_to_seq(peak_str):
            chromosome, start, end = self.decompose_chrstr(peak_str)
            seq = self.genome_data[chromosome][start:end].seq
            name = f"{chromosome}-{start}-{end}"
            return name, seq

        fasta = Fasta()
        for peak in peaks:
            name, seq = peak_to_seq(peak)
            fasta.add(name, seq)

        return fasta

    @staticmethod
    def remove_zero_seq(fasta_object):
        """
        Remove sequences with zero length from a FASTA object.

        Args:
            fasta_object (Fasta): GimmeMotifs Fasta object.

        Returns:
            Fasta: Filtered Fasta object.
        """
        fasta_filtered = Fasta()
        for name, seq in zip(fasta_object.ids, fasta_object.seqs):
            if seq:
                fasta_filtered.add(name, seq)
        return fasta_filtered

    @staticmethod
    def read_bed(bed_path):
        """
        Read a BED file and return it as a DataFrame.

        Args:
            bed_path (str): Path to the BED file.

        Returns:
            pd.DataFrame: DataFrame representation of the BED file.
        """
        df = BedTool(bed_path).to_dataframe().dropna(axis=0)
        df["seqname"] = df["chrom"] + "-" + df["start"].astype(str) + "-" + df["end"].astype(str)
        return df

# Example usage
if __name__ == "__main__":
    genomes_dir = "/path/to/genomes"
    ref_genome = "zebrafish"

    # Initialize the class
    atac_analysis = ATACSeqMotifAnalysis(ref_genome, genomes_dir)

    # Example peaks
    peaks = ["1-100-200", "2-300-400"]

    # Convert to DataFrame
    peaks_df = atac_analysis.list_peakstr_to_df(peaks)

    # Validate peaks
    valid_peaks = atac_analysis.check_peak_format(peaks_df)

    # Convert peaks to FASTA
    fasta = atac_analysis.peak_to_fasta(valid_peaks["chr"] + "-" + valid_peaks["start"].astype(str) + "-" + valid_peaks["end"].astype(str))

    # Remove zero-length sequences
    filtered_fasta = atac_analysis.remove_zero_seq(fasta)

    # Save FASTA file
   # Step 5: Save FASTA file
    output_fasta_path = ""
    with open(output_fasta_path, "w") as f:
        for name, seq in zip(filtered_fasta.ids, filtered_fasta.seqs):
            f.write(f">{name}\n{seq}\n")
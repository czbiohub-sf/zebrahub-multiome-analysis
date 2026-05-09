"""Small reusable helpers for looking up gene structure from a GTF.

Used by plot_peaks_locus_view.py and gene_locus_explore.py — kept
separate so any new gene-locus utility can import without duplicating
GTF parsing logic.
"""

import gzip
import re
from collections import defaultdict
from typing import Optional, Tuple, List

DEFAULT_GTF = ("/hpc/reference/sequencing_alignment/alignment_references/"
                "zebrafish_genome_GRCz11/genes/genes.gtf.gz")


def get_gene_struct(gtf_path: str, gene_name: str
                     ) -> Optional[Tuple[str, int, int, str, List[Tuple[int, int]]]]:
    """Return (chrom, gene_start, gene_end, strand, exons) for the longest
    transcript of `gene_name`. exons = list of (start, end). None if not found.
    """
    opener = gzip.open if gtf_path.endswith(".gz") else open
    transcripts = defaultdict(list)
    tx_meta = {}
    needle = f'gene_name "{gene_name}"'
    with opener(gtf_path, "rt") as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 9 or needle not in parts[8]:
                continue
            ftype = parts[2]
            chrom, start, end, strand = parts[0], int(parts[3]), int(parts[4]), parts[6]
            m = re.search(r'transcript_id "([^"]+)"', parts[8])
            tx = m.group(1) if m else None
            if not tx:
                continue
            if ftype == "transcript":
                tx_meta[tx] = (chrom, start, end, strand)
            elif ftype == "exon":
                transcripts[tx].append((start, end))

    if not tx_meta:
        return None
    longest_tx = max(tx_meta, key=lambda t: tx_meta[t][2] - tx_meta[t][1])
    chrom, gs, ge, strand = tx_meta[longest_tx]
    exons = sorted(transcripts.get(longest_tx, []))
    return chrom, gs, ge, strand, exons


def get_gene_tss(gtf_path: str, gene_name: str) -> Optional[Tuple[str, int]]:
    """Return ('chr<N>', tss_pos) for the most upstream TSS of `gene_name`.
    For + strand genes that's the smallest start across all transcripts;
    for - strand it's the largest end."""
    opener = gzip.open if gtf_path.endswith(".gz") else open
    needle = f'gene_name "{gene_name}"'
    starts, ends = [], []
    chrom, strand = None, None
    with opener(gtf_path, "rt") as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 9 or needle not in parts[8] or parts[2] != "transcript":
                continue
            chrom = parts[0]
            strand = parts[6]
            starts.append(int(parts[3]))
            ends.append(int(parts[4]))

    if not starts:
        return None
    chrom_str = f"chr{chrom}" if not chrom.startswith("chr") else chrom
    tss = min(starts) if strand == "+" else max(ends)
    return chrom_str, tss

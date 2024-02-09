# Compute the GC content of peaks for SEACells
# SEACells.genescore.prepare_multiome_anndata()
# source: https://github.com/dpeerlab/SEACells/issues/12

# inputs:
# 1) adata_atac: anndata for the scATAC-seq (cells-by-peaks)
# output:
def compute_GC_contents_scATAC(adata_atac, genome_name, provider):
    import numpy as np
    import anndata as ad

    import genomepy
    from Bio.SeqUtils import GC

    #adata_atac = ad.read_h5ad('../write/filtered_data_atac.h5ad')

    # download genome from NCBI
    genome_name = "GRCz11"
    provider = "Ensembl"
    genomepy.install_genome(name=genome_name, provider=provider, genome_dir = "/hpc/scratch/group.data.science/yangjoon.kim/data")
    genome = genomepy.Genome(name=genome_name, genome_dir="/hpc/scratch/group.data.science/yangjoon.kim/data")
    #genomepy.install_genome(name='GRCh38', provider='NCBI', genomes_dir = ''../data') # took about 9 min
    #genome = genomepy.Genome(name = 'GRCh38', genomes_dir = '../data')

    GC_content_list = []

    for i, region in enumerate(adata_atac.var_names):
        chromosome, start, end = region.split('-')
        chromosome = chromosome[3:]

        # get sequence
        sequence = str(genome.get_seq(name = chromosome, start = int(start), end = int(end)))

        # calculate GC content
        GC_content = GC(str(sequence))
        GC_content_list.append(GC_content)

    # GC content ranges from 0% - 100%, should be 0 to 1
    adata_atac.var['GC'] = GC_content_list
    adata_atac.var.GC = adata_atac.var.GC/100

    return adata_atac

    # # To find the right genome name and provider
    # for provided_genome in genomepy.search('GRCh38', provider=None):
    #    print(provided_genome)
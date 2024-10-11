#load conda environment vel38zebra
#prepare counts layer of adata
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl

import scvelo as scv
import cellrank as cr
import scipy.sparse

from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
from numpy.linalg import solve, norm
from numpy.random import rand

import sys

adata_path = sys.argv[1]
save_path = sys.argv[2]


adata_fresh = sc.read_h5ad(adata_path)

adata_fresh.var_names_make_unique()

adata = adata_fresh.copy()


scv.pp.filter_genes(adata, min_shared_counts=20)
scv.pp.normalize_per_cell(adata)
scv.pp.filter_genes_dispersion(adata, n_top_genes=2000)
scv.pp.log1p(adata)


scv.pp.filter_and_normalize(adata, min_shared_counts=20, n_top_genes=2000)
scv.pp.moments(adata, n_pcs=30, n_neighbors=30)
sc.pp.neighbors(adata, n_neighbors=30, n_pcs=30)

scv.tl.velocity(adata, mode='deterministic')

scv.tl.velocity_graph(adata)

adata.write(save_path)
"""Reproduce exploratory UMAP layout comparisons without changing the main atlas.

Run after continuous_atlas_benchmark.py. All initializations come from measured
profiles; no latent labels or synthetic positions are provided to UMAP.
"""
import argparse
from pathlib import Path

import anndata as ad
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc

from continuous_atlas_benchmark import HERE, PALETTE


def plot_layouts(a,layouts,output):
    fig=plt.figure(figsize=(15,5),layout='constrained')
    family=a.obs.true_family.to_numpy().astype(int)
    for panel,(label,xyz) in enumerate(layouts):
        ax=fig.add_subplot(1,len(layouts),panel+1,projection='3d')
        for f in [-1,0,1,2,3]:
            idx=family==f
            ax.scatter(*xyz[idx].T,c=PALETTE[f+1],s=9 if f==-1 else 3,alpha=0.8)
        ax.set(title=label,xticks=[],yticks=[],zticks=[],xlabel='UMAP 1',ylabel='UMAP 2',zlabel='UMAP 3')
        ax.view_init(elev=20,azim=35)
    fig.savefig(output,dpi=160,bbox_inches='tight')
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output-dir',type=Path,default=HERE/'outputs'/'synthetic_20ct_continuous_v2')
    args=parser.parse_args()
    a=ad.read_h5ad(args.output_dir/'synthetic_continuous.h5ad')
    layouts=[]
    for k in [30,60,100]:
        b=a.copy()
        sc.pp.neighbors(b,n_neighbors=k,use_rep='X_pca',random_state=42)
        sc.tl.umap(b,n_components=3,min_dist=0.4,random_state=42,
                   init_pos='spectral',gamma=1.,maxiter=500)
        print(f'Finished spectral initialization, k={k}',flush=True)
        layouts.append((f'Spectral initialization; k={k}\nRepulsion 1; 500 iterations',b.obsm['X_umap'].copy()))
    plot_layouts(a,layouts,args.output_dir/'03_neighbor_sensitivity.png')
    initialization=[('Spectral; repulsion 1\n500 iterations',layouts[0][1])]
    for repulsion in [1.,0.5]:
        b=a.copy()
        b.obsm['X_pca_init3d']=b.obsm['X_pca'][:,:3].copy()
        sc.tl.umap(b,n_components=3,min_dist=0.4,random_state=42,
                   init_pos='X_pca_init3d',gamma=repulsion,maxiter=1000)
        print(f'Finished PCA initialization, repulsion={repulsion}',flush=True)
        initialization.append((f'PCA initialization; repulsion {repulsion:g}\n1000 iterations',b.obsm['X_umap'].copy()))
    plot_layouts(a,initialization,args.output_dir/'04_initialization_sensitivity.png')
    np.savez_compressed(args.output_dir/'diagnostic_layout_coordinates.npz',
        **{f'spectral_k{k}':row[1] for k,row in zip([30,60,100],layouts)},
        pca_repulsion_1=initialization[1][1],pca_repulsion_05=initialization[2][1])


if __name__=='__main__':
    main()

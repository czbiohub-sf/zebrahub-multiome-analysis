#!/usr/bin/env python3
"""Scientific figures for all-peak frozen-landmark projections."""
from pathlib import Path
import h5py,numpy as np
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ct_peak_atlas import OUT,dump

def main():
 plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':190})
 with h5py.File(OUT/'full_ct_peak_atlas.h5','r') as h:
  y2=h['umap2d'][:];y3=h['umap3d'][:];mean=h['annotations/mean_cpm'][:];tau=h['annotations/ct_tau'][:];stage=h['annotations/stage_of_max_mean_cpm'][:];clusters=h['leiden_coarse'].asstr()[:]
 codes=np.searchsorted(np.unique(clusters),clusters);order=np.random.default_rng(20261021).permutation(len(mean));cols=[(np.log1p(mean),'Mean accessibility: log1p(CPM)','viridis'),(tau,'Celltype specificity τ','magma'),(codes,'Original coarse peak clusters','turbo')]
 fig,axes=plt.subplots(1,3,figsize=(16,5.5))
 for ax,(c,label,cmap) in zip(axes,cols):
  sc=ax.scatter(y2[order,0],y2[order,1],c=c[order],cmap=cmap,s=.16,alpha=.55,linewidths=0,rasterized=True);ax.set_title(label);ax.set_xticks([]);ax.set_yticks([])
  if cmap!='turbo':fig.colorbar(sc,ax=ax,shrink=.65)
 fig.suptitle('All 640,830 peaks in the celltype × stage log1p(CPM) atlas\nFrozen PCA/UMAP fitted on 20,000 training peaks; every sampled coordinate preserved exactly.',fontsize=14);fig.tight_layout(rect=(0,0,1,.9));fig.savefig(OUT/'04_full_peak_atlas_2d.png');fig.savefig(OUT/'04_full_peak_atlas_2d.pdf');plt.close(fig)
 take=order[:60000];fig=plt.figure(figsize=(16,5.5))
 for i,(c,label,cmap) in enumerate([(np.log1p(mean),'Mean accessibility: log1p(CPM)','viridis'),(tau,'Celltype specificity τ','magma'),(stage,'Stage of maximum mean CPM','viridis')],1):
  ax=fig.add_subplot(1,3,i,projection='3d');ax.view_init(elev=55,azim=90);ax.set_box_aspect(np.ptp(y3[take],axis=0));sc=ax.scatter(y3[take,0],y3[take,1],y3[take,2],c=c[take],cmap=cmap,s=.5,alpha=.5,linewidths=0,rasterized=True);ax.set_title(label);ax.set_xticks([]);ax.set_yticks([]);ax.set_zticks([]);fig.colorbar(sc,ax=ax,shrink=.55)
 fig.suptitle('Full-peak 3D landmark projection • display: fixed random 60,000-peak subset\nCoordinates exist for all 640,830 peaks. No root or temporal direction imposed.',fontsize=14);fig.tight_layout(rect=(0,0,1,.9));fig.savefig(OUT/'05_full_peak_atlas_3d.png');fig.savefig(OUT/'05_full_peak_atlas_3d.pdf');plt.close(fig)
 dump(OUT/'full_plot_sampling.json',{'2d_display_peaks':len(mean),'3d_display_peaks':len(take),'3d_rng_seed':20261021,'all_full_coordinates_saved':True})
if __name__=='__main__':main()

#load environment vel38zebra 
#Leah made this

import numpy as np
import loompy
import os
import scvelo as scv
import pandas as pd

destdir="/hpc/projects/data.science/sarah.ancheta/ZF_atlas/rna_velocity/velocity_subset_take2/"
savedir="/hpc/projects/data.science/sarah.ancheta/ZF_atlas/rna_velocity/subset_adatas/full_adatas_subset/"

whitelist_barcodes=pd.read_csv(destdir + 'whitelist_barcodes.csv', index_col=0)
fishfolders = []
for folder in os.listdir(destdir):
    if folder.startswith('TDR'):
        fishfolders.append(folder)

    #this iterates through a list for any objects that meet criteria. 
fraction=0.25
for iteration in range(1,6):
    frit=str(fraction)+'-'+str(iteration)+'.loom'
    file1 = [i for i in os.listdir(destdir+fishfolders[0]) if i.split("-",maxsplit=1)[1]==frit][0]
    file2 = [i for i in os.listdir(destdir+fishfolders[1]) if i.split("-",maxsplit=1)[1]==frit][0]
    file3 = [i for i in os.listdir(destdir+fishfolders[2]) if i.split("-",maxsplit=1)[1]==frit][0]
    file4 = [i for i in os.listdir(destdir+fishfolders[3]) if i.split("-",maxsplit=1)[1]==frit][0]
    print(file1,file2,file3)
    files=[destdir+fishfolders[0]+'/'+file1,destdir+fishfolders[1]+'/'+file2,destdir+fishfolders[2]+'/'+file3,destdir+fishfolders[3]+'/'+file4]
    newfile=destdir+"allfish_"+frit
    loompy.combine(files,newfile,key="Accession")
    adata  = scv.read_loom(newfile)

    adata.obs['new_index'] = adata.obs.index.copy()
    adata.obs['fish'] = [i.split("_")[0] for i in adata.obs['new_index']]
    adata.obs['cellbarcode'] = [i.split(":")[1].split("x")[0] for i in adata.obs['new_index']]
    adata.obs['my_ID'] = adata.obs['fish'].astype(str) + "_" + adata.obs['cellbarcode'].astype(str) + "-1"
    adata.obs.set_index('my_ID', inplace=True)
    adata=adata[adata.obs.index.isin(whitelist_barcodes.index.tolist())].copy()
    adata.obs['clusters'] = whitelist_barcodes.clusters.copy()
    adata.write_h5ad(savedir + 'allfish_' + str(fraction) + '-' + str(iteration)+ '.h5ad')
    #break

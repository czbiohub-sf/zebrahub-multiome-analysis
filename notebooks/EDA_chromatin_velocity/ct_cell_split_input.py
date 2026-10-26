"""Independent cell-half pseudobulks for conditional sampling validation.

Cells are partitioned within celltype, stage and dataset. The same peak retains
its genomic identity; no absent combination is replaced by zero accessibility.
These are sampling splits, not biological replicates. Original input files are
opened read-only; exact pooled-count reconstruction is audited for every peak.
"""
from pathlib import Path
import json,sys,time
import h5py,numpy as np,pandas as pd
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from real_input_provenance import read_column
OUT=HERE/'outputs/ct_flow_noise_v1';SAMPLE=HERE/'outputs/state_trajectory_atlas_v1/sample_data.h5'
BASE=HERE.parents[2]/'data';RAW=BASE/'processed_data/01_Signac_processed/integrated_RNA_ATAC_counts_peaks_integrated_raw_counts_master_filtered.h5ad';PB=BASE/'annotated_data/objects_v2/peaks_by_ct_tp_master.h5ad'
STAGES={'0somites':10,'5somites':12,'10somites':14,'15somites':16,'20somites':19,'30somites':24};SEED=20261021

def signature(path):
 s=path.stat();return dict(path=str(path),bytes=s.st_size,mtime_ns=s.st_mtime_ns)

def main():
 OUT.mkdir(parents=True,exist_ok=True);before={str(p):signature(p) for p in [RAW,PB,SAMPLE]}
 with h5py.File(SAMPLE) as s:
  ids=s['peak_ids'].asstr()[:];source=s['source_indices'][:];ct=s['celltypes'].asstr()[:];stages=s['stages'][:];valid=s['condition_valid'][:];train=s['train_mask'][:];test=s['test_mask'][:]
 n=len(ids);C=len(ct);T=len(stages);ctmap={x:i for i,x in enumerate(ct)};tmap={x:i for i,x in enumerate(stages)}
 with h5py.File(RAW,'r') as f,h5py.File(PB,'r') as pb:
  rawids=read_column(f['var'],'_index');lookup=pd.Index(rawids);indices=lookup.get_indexer(ids);assert (indices>=0).all() and len(np.unique(indices))==n
  names=read_column(f['obs'],'_index');celltypes=read_column(f['obs'],'annotation_ML_coarse');timepoints=read_column(f['obs'],'dev_stage');datasets=read_column(f['obs'],'dataset');exposure=np.asarray(read_column(f['obs'],'nCount_peaks_integrated'),float)
  groups=np.asarray([ctmap[c]*T+tmap[STAGES[t]] for c,t in zip(celltypes,timepoints)],int);rng=np.random.default_rng(SEED);half=np.full(len(names),-1,np.int8)
  for group in np.unique(groups):
   groupidx=np.flatnonzero(groups==group);na=nb=0
   for dataset in sorted(set(datasets[groupidx])):
    members=groupidx[datasets[groupidx]==dataset];members=rng.permutation(members);cut=len(members)//2
    if len(members)%2 and na<=nb:cut+=1
    half[members[:cut]]=0;half[members[cut:]]=1;na+=cut;nb+=len(members)-cut
   assert abs(na-nb)<=1
  assert (half>=0).all()
  combo=groups*2+half;cov=np.bincount(combo,weights=exposure,minlength=C*T*2).reshape(C,T,2);nc=np.bincount(combo,minlength=C*T*2).reshape(C,T,2)
  pct=read_column(pb['var'],'annotation_ML_coarse');pst=read_column(pb['var'],'dev_stage');pcov=read_column(pb['var'],'total_coverage');pnc=read_column(pb['var'],'n_cells');columns=np.full((C,T),-1,int)
  for j,(c,t) in enumerate(zip(pct,pst)):
   ci=ctmap[c];ti=tmap[STAGES[t]];columns[ci,ti]=j;assert cov[ci,ti].sum()==pcov[j],(c,t,cov[ci,ti].sum(),pcov[j]);assert nc[ci,ti].sum()==pnc[j]
  counts=np.zeros((n,C,T,2),np.int64);x=f['layers/counts'];indptr=x['indptr'][:];order=np.argsort(indices)
  start=time.time()
  for k,i in enumerate(order):
   a,b=indptr[indices[i]:indices[i]+2];rows=x['indices'][a:b];vals=x['data'][a:b]
   counts[i]=np.bincount(combo[rows],weights=vals,minlength=C*T*2).reshape(C,T,2).astype(np.int64)
   if (k+1)%5000==0:print(f'Aggregated {k+1:,}/{n:,} peak columns ({time.time()-start:.1f}s)',flush=True)
  porder=np.argsort(source);inv=np.argsort(porder);reference=pb['layers/sum'][source[porder]][inv]
  pooled=counts.sum(axis=-1);colvalues=np.empty((n,len(pct)),np.int64)
  for c,t in zip(*np.where(columns>=0)):colvalues[:,columns[c,t]]=pooled[:,c,t]
  assert np.array_equal(colvalues,reference),'Cell-half sums differ from existing pseudobulk counts'
  cpm=np.divide(counts*1e6,cov[None],out=np.full(counts.shape,np.nan),where=cov[None]>0)
  tmp=OUT/'cell_split_input.partial.h5'
  with h5py.File(tmp,'w') as out:
   for key,array in [('peak_ids',ids),('celltypes',ct)]:out.create_dataset(key,data=np.asarray(array,object),dtype=h5py.string_dtype())
   for key,array in [('source_indices',source),('raw_peak_indices',indices),('stages',stages),('train_mask',train),('test_mask',test),('condition_valid',valid),('column_index',columns)]:out.create_dataset(key,data=array)
   for label,h in [('a',0),('b',1)]:
    out.create_dataset('raw_counts_'+label,data=counts[:,:,:,h].transpose(0,2,1),compression='gzip')
    out.create_dataset('observed_cpm_'+label,data=cpm[:,:,:,h].transpose(0,2,1).astype('f4'),compression='gzip')
    out.create_dataset('coverage_'+label,data=cov[:,:,h]);out.create_dataset('n_cells_'+label,data=nc[:,:,h])
   out.create_dataset('cell_half',data=half,compression='gzip');out.create_dataset('cell_ids',data=np.asarray(names,object),dtype=h5py.string_dtype(),compression='gzip')
   out.attrs['axis_order']='CPM/counts peak,stage,celltype; coverage/cell-counts celltype,stage';out.attrs['interpretation']='Independent cells within each CT×stage×dataset; conditional sampling validation, not biological replication';out.attrs['seed']=SEED
  tmp.replace(OUT/'cell_split_input.h5')
  pd.DataFrame(dict(cell_id=names,celltype=celltypes,stage_hpf=[STAGES[t] for t in timepoints],dataset=datasets,half=half)).to_csv(OUT/'cell_split_assignment.csv.gz',index=False)
  samples=pd.crosstab(pd.Series([STAGES[t] for t in timepoints],name='hpf'),pd.Series(datasets,name='dataset'));samples.to_csv(OUT/'dataset_stage_counts.csv')
  report=dict(status='passed',n_cells=len(names),n_peaks=n,seed=SEED,pooled_counts_match_source_exactly=True,all_190_source_coverages_match=True,no_cell_overlap=True,split_rule='Within CT×stage, shuffled separately by dataset and balanced between halves',minimum_half_cells_primary=int(nc[valid.all(axis=1)].min()),sampling_not_biological_replication=True,dataset_stage_table=samples.to_dict(),source_signatures=before)
 assert before=={str(p):signature(p) for p in [RAW,PB,SAMPLE]},'Source file modified'
 (OUT/'cell_split_validation.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k not in ['source_signatures','dataset_stage_table']}),flush=True)
if __name__=='__main__':main()

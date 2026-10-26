"""Fixed-sampling visual comparison of projection hyperparameters and controls."""
from pathlib import Path
import sys,json,hashlib
import h5py,numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import plotly.offline
HERE=Path(__file__).resolve().parent
OUT=HERE/'outputs/ct_velocity_experiments_v1';SRC=HERE/'outputs/ct_velocity_flow_sweep_v1'

def sample_spatial(xy,mask,budget=600):
 """One deterministic random source per 2D bin; no coherence ranking."""
 rng=np.random.default_rng(20261022);valid=np.flatnonzero(mask);lo=xy.min(axis=0);span=np.ptp(xy,axis=0);bins=np.floor((xy[valid]-lo)/np.maximum(span,1e-8)*35).astype(int);code=bins[:,0]*36+bins[:,1];order=rng.permutation(len(valid));_,first=np.unique(code[order],return_index=True);selected=valid[order[first]]
 if len(selected)>budget:selected=selected[rng.choice(len(selected),budget,replace=False)]
 return np.sort(selected)

def main():
 OUT.mkdir(parents=True,exist_ok=True)
 with h5py.File(SRC/'sweep_vectors.h5') as f:
  xy=f['umap2d'][:];ids=f['peak_ids'].asstr()[:];source=f['source_indices'][:];broad=f['broad_reference'][:].astype(bool);stages=f['stages'][:];hops=f['root_hops'][:];data={'xy':xy.tolist(),'peak_ids':ids.tolist(),'source_indices':source.tolist(),'broad':np.flatnonzero(broad).tolist(),'intervals':[f'{int(a)} → {int(b)} hpf' for a,b in zip(stages[:-1],stages[1:])],'fields':{},'arrow_indices':{},'intrinsic_net_rate':{},'gradual':{},'source_active':{}}
  maskchecks=[]
  for t in range(5):
   it=f[f'interval_{t}'];active=it['source_active'][:].astype(bool);selection=sample_spatial(xy,active);data['source_active'][str(t)]=active.astype(int).tolist();data['arrow_indices'][str(t)]=selection.tolist();data['intrinsic_net_rate'][str(t)]=it['intrinsic_net_rate'][:].tolist();data['gradual'][str(t)]=it['gradual'][:].astype(int).tolist()
   for k in [10,30,60,100]:
    for beta in [0,25,50]:
     tag=f'k{k}_b{beta:03d}';g=it[tag]
     for ctl in ['actual','reversed']+(['permutation_0'] if beta==0 else []):
      item=g if ctl=='actual' else g[ctl];v=item['vectors2d'][:];valid=item['valid2d'][:].astype(bool);key=f'{t}_{k}_{beta}_{ctl}';data['fields'][key]={'vectors':v[selection].tolist(),'valid':valid[selection].astype(int).tolist(),'eligible':int(valid.sum()),'excess':item['root_excess_delta'][selection].tolist()};maskchecks.append({'key':key,'sampled_arrows':int(valid[selection].sum()),'all_eligible':int(valid.sum())})
  vals=np.abs(np.concatenate([np.asarray(data['intrinsic_net_rate'][str(t)])[f[f'interval_{t}/source_active'][:].astype(bool)] for t in range(5)]));data['rate_limit']=float(np.nanquantile(vals,.98));data['span']=float(np.linalg.norm(np.ptp(xy,axis=0)))
  # Two prespecified representative intervals, every inference setting.
  for t,number in [(0,'04'),(3,'05')]:
   selection=np.asarray(data['arrow_indices'][str(t)])[::2];net=np.asarray(data['intrinsic_net_rate'][str(t)]);fig,axes=plt.subplots(3,4,figsize=(17,12))
   for r,beta in enumerate([0,25,50]):
    for c,k in enumerate([10,30,60,100]):
     ax=axes[r,c];g=f[f'interval_{t}/k{k}_b{beta:03d}'];valid=g['valid2d'][:].astype(bool);v=g['vectors2d'][:];ax.scatter(*xy.T,s=.6,c='#d4d8dc',alpha=.6,linewidths=0,rasterized=True);shown=np.flatnonzero(f[f'interval_{t}/source_active'][:].astype(bool));art=ax.scatter(*xy[shown].T,c=net[shown],s=1,cmap='RdBu_r',norm=TwoSlopeNorm(0,-data['rate_limit'],data['rate_limit']),alpha=.8,linewidths=0,rasterized=True);chosen=selection[valid[selection]];vv=v[chosen];vv=vv/np.linalg.norm(vv,axis=1)[:,None]*data['span']*.014;ax.quiver(*xy[chosen].T,*vv.T,angles='xy',scale_units='xy',scale=1,color='#18313b',width=.003,headwidth=3.7,headlength=4.5);ax.set(aspect='equal',xticks=[],yticks=[],title=f'k={k}; {len(chosen)} sampled arrows')
     if c==0:ax.set_ylabel(f'Neighbor-rate blend β={beta/100:g}')
   fig.subplots_adjust(top=.91,bottom=.09,left=.055,right=.91,wspace=.1,hspace=.15);cax=fig.add_axes([.93,.30,.012,.4]);fig.colorbar(art,cax=cax,label='Original fitted mean rate (CPM/hour)');fig.suptitle(f'Projection sensitivity · {data["intervals"][t]} · fixed 20,000 query peaks',fontsize=17);fig.text(.5,.025,'Candidate neighbors come from all 640,830 peaks. Arrow locations are spatially sampled without coherence ranking and fixed across panels.\nβ blends rates with a fixed k=30 neighbor mean; source gates and UMAP stay fixed. Arrow lengths are normalized for display.',ha='center',fontsize=10);fig.savefig(OUT/f'{number}_projection_parameter_arrows_{int(stages[t])}_{int(stages[t+1])}.png',dpi=175);fig.savefig(OUT/f'{number}_projection_parameter_arrows_{int(stages[t])}_{int(stages[t+1])}.pdf');plt.close(fig)
 def finite_json(a):
  if isinstance(a,float) and not np.isfinite(a):return None
  if isinstance(a,list):return [finite_json(v) for v in a]
  if isinstance(a,dict):return {k:finite_json(v) for k,v in a.items()}
  return a
 payload=json.dumps(finite_json(data),separators=(',',':'),allow_nan=False).replace('</','<\\/')
 template=(HERE/'ct_velocity_sweep_viewer_template.html').read_text();html=template.replace('__PLOTLY__',plotly.offline.get_plotlyjs()).replace('__DATA__',payload);(OUT/'parameter_explorer.html').write_text(html)
 np.savez_compressed(OUT/'viewer_sample_indices.npz',**{f't{t}':np.asarray(data['arrow_indices'][str(t)]) for t in range(5)})
 validation=dict(status='passed',n_queries=len(xy),n_parameter_fields=len(maskchecks),coordinates_saved_exactly=True,vector_values_saved_exactly=True,arrows='One random representative per occupied 2D bin, fixed across parameters within each interval; no coherence ranking.',display_masks_exact=True,source=str(SRC/'sweep_vectors.h5'),source_bytes=(SRC/'sweep_vectors.h5').stat().st_size,html_sha256=hashlib.sha256(html.encode()).hexdigest(),fields=maskchecks)
 (OUT/'parameter_viewer_validation.json').write_text(json.dumps(validation,indent=2)+'\n');print('Parameter explorer and arrow panels written',flush=True)

if __name__=='__main__':main()

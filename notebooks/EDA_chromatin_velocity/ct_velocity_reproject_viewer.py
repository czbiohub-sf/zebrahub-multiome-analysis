#!/usr/bin/env python3
"""Offline viewer: retrospective chromatin-velocity directions on the fixed CT atlas.

All numeric payloads preserve source floating-point precision. Display arrow normalization is
performed only in JavaScript and does not alter the saved projected vectors.
"""
from pathlib import Path
import base64, hashlib, json
import h5py
import numpy as np
import plotly.offline

HERE=Path(__file__).resolve().parent
OUT=HERE/'outputs/ct_velocity_reprojection_v1'
ATLAS=HERE/'outputs/ct_peak_atlas_v1'
METHODS={'updated':('updated_projection.h5','Updated celltype-space neighbors'),
         'reused':('reused_projection.h5','Previous transition weights · coordinate-only control')}

def pack(a):
    a=np.asarray(a)
    dtype='<f8' if a.dtype.kind=='f' and a.dtype.itemsize==8 else '<f4'
    a=np.asarray(a,dtype=dtype)
    if np.isinf(a).any():
        raise ValueError('Infinite values in viewer payload')
    return {'shape':list(a.shape),'dtype':dtype,'data':base64.b64encode(a.tobytes()).decode()}

def unpack(p):
    return np.frombuffer(base64.b64decode(p['data']),dtype=p['dtype']).reshape(p['shape'])

def spatial_sample(coords, eligible, maximum=600, seed=20261022):
    """One random eligible peak per spatial bin, then random fill; no direction optimization."""
    inds=np.flatnonzero(eligible)
    if len(inds)<=maximum:
        return inds
    rng=np.random.default_rng(seed)
    d=coords.shape[1]
    bins=26 if d==2 else 9
    lo=coords.min(axis=0);span=np.maximum(np.ptp(coords,axis=0),1e-8)
    grid=np.minimum(bins-1,((coords[inds]-lo)/span*bins).astype(int))
    code=np.ravel_multi_index(grid.T, (bins,)*d)
    perm=rng.permutation(len(inds))
    _,first=np.unique(code[perm],return_index=True)
    chosen=perm[first]
    if len(chosen)>maximum:
        chosen=rng.choice(chosen,maximum,replace=False)
    elif len(chosen)<maximum:
        available=np.setdiff1d(np.arange(len(inds)),chosen,assume_unique=True)
        chosen=np.r_[chosen,rng.choice(available,maximum-len(chosen),replace=False)]
    return inds[rng.permutation(chosen)]

def build():
    OUT.mkdir(parents=True,exist_ok=True)
    inputs=[OUT/v[0] for v in METHODS.values()]+[ATLAS/'full_ct_peak_atlas.h5',ATLAS/'aligned_features.h5',ATLAS/'viewer_sample_indices.npy']
    def signatures():
        return {str(p):{'bytes':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns} for p in inputs}
    before=signatures()
    with h5py.File(ATLAS/'aligned_features.h5') as old:
        source=old['source_indices'][:]
    previous=np.load(ATLAS/'viewer_sample_indices.npy')
    selected=np.sort(source[previous])
    with h5py.File(ATLAS/'full_ct_peak_atlas.h5') as h:
        ids=h['peak_ids'].asstr()[:];genes=h['nearest_gene'].asstr()[:]
        coords={str(d):h[f'umap{d}d'][:] for d in (2,3)}
        mean=h['annotations/mean_cpm'][:]
        tau=h['annotations/ct_tau'][:]
    data={'title':'Chromatin velocity on the celltype × stage peak atlas',
          'n_full':len(ids),'peak_ids':ids[selected].tolist(),'genes':genes[selected].tolist(),
          'source_indices':selected.tolist(),'times':[10,12,14,16,19,24],
          'intervals':['10 → 12','12 → 14','14 → 16','16 → 19','19 → 24'],
          'coordinates':{d:pack(v[selected]) for d,v in coords.items()},
          'mean_cpm':pack(mean[selected]),'tau':pack(tau[selected]),
          'display_base':{d:float(np.max(np.ptp(v,axis=0))*.012) for d,v in coords.items()},
          'methods':{}}
    checks=[]; reference_rates=[]; reference_rms=[]
    def verify(name,payload,expected):
        actual=unpack(payload)
        ok=np.array_equal(actual,np.asarray(expected,dtype=payload['dtype']),equal_nan=True)
        if not ok:raise AssertionError(name)
        checks.append({'check':name,'passed':bool(ok),'shape':list(actual.shape)})
    for d,v in coords.items():verify(f'{d}D background coordinates',data['coordinates'][d],v[selected])
    for mi,(method,(filename,label)) in enumerate(METHODS.items()):
        with h5py.File(OUT/filename) as h:
            np.testing.assert_array_equal(h['peak_ids'].asstr()[:],ids)
            data['methods'][method]={'label':label,'intervals':[]}
            for d in coords:
                np.testing.assert_array_equal(h[f'umap{d}d'][:],coords[d])
            for it in range(5):
                g=h[f'projection/{it}'];net=g['net_rate'][:];rms=g['velocity_rms'][:]
                active=g['active'][:].astype(bool);changing=g['changing'][:].astype(bool)
                if method=='updated':
                    reference_rates.append(net[changing & np.isfinite(net)])
                    reference_rms.append(rms[changing & np.isfinite(rms)])
                item={'net_rate':pack(net[selected]),'velocity_rms':pack(rms[selected]),
                      'active':pack(active[selected]),'changing':pack(changing[selected]),
                      'n_active_full':int(active.sum()),'arrows':{}}
                verify(f'{method}/{it}/net_rate',item['net_rate'],net[selected])
                for d,xy in coords.items():
                    key=f'umap{d}d';vectors=g[f'display_vectors/{key}'][:]
                    valid=g[f'projection_valid/{key}'][:].astype(bool)
                    coh=g[f'coherence/{key}'][:]
                    eligible=active & valid & np.isfinite(vectors).all(axis=1) & (np.linalg.norm(vectors,axis=1)>0)
                    ai=spatial_sample(xy,eligible,600,20261022+it*10+int(d))
                    ar={'source_indices':ai.tolist(),'peak_ids':ids[ai].tolist(),'genes':genes[ai].tolist(),
                        'origins':pack(xy[ai]),'vectors':pack(vectors[ai]),'net_rate':pack(net[ai]),
                        'coherence':pack(coh[ai]),'n_eligible_full':int(eligible.sum())}
                    item['arrows'][d]=ar
                    item[f'coherence_{d}']=pack(coh[selected])
                    item[f'valid_{d}']=pack(valid[selected])
                    verify(f'{method}/{it}/{d}D arrow origins',ar['origins'],xy[ai])
                    verify(f'{method}/{it}/{d}D arrow vectors',ar['vectors'],vectors[ai])
                    if len(ai):
                        assert eligible[ai].all()
                        assert len(np.unique(ai))==len(ai)
                data['methods'][method]['intervals'].append(item)
    # Both methods reuse the same intrinsic biological rates; the projection alone differs.
    for it in range(5):
        a=unpack(data['methods']['updated']['intervals'][it]['net_rate'])
        b=unpack(data['methods']['reused']['intervals'][it]['net_rate'])
        np.testing.assert_array_equal(a,b)
    rates=np.concatenate(reference_rates)
    data['rate_color_limit']=max(float(np.quantile(np.abs(rates),.98)),1e-5)
    data['rms_color_limit']=max(float(np.quantile(np.concatenate(reference_rms),.98)),1e-8)
    data['mean_color_limit']=max(float(np.quantile(np.log1p(mean[selected]),.99)),1e-8)
    assert before==signatures(), 'Input changed during viewer build'
    payload=json.dumps(data,separators=(',',':'),allow_nan=False).replace('</','<\\/')
    template=(HERE/'ct_velocity_reproject_viewer_template.html').read_text()
    html=template.replace('__PLOTLY__',plotly.offline.get_plotlyjs()).replace('__DATA__',payload)
    output=OUT/'ct_velocity_explorer.html';output.write_text(html)
    np.save(OUT/'viewer_source_indices.npy',selected)
    manifest={'path':str(output),'bytes':output.stat().st_size,'sha256':hashlib.sha256(output.read_bytes()).hexdigest(),
              'n_full_peaks':len(ids),'n_background_peaks':len(selected),'max_arrows_per_view':600,
              'selection':'Background: same 8,038 peaks as previous explorer, reordered by full source index. Arrows: spatially balanced random subset of eligible full-atlas peaks, independent of arrow direction; nested display-density subsets.',
              'numeric_encoding':'Exact source float32/float64 values, base64; display normalization happens only during rendering.',
              'arrow_scale':'Default equal-length directions: max(full-coordinate axis span) × 0.012 × display multiplier. Optional coherence-weighted vectors use the raw projected vector and the same scale. Neither indicates biological speed.',
              'rate_color_limit_98th_percentile':data['rate_color_limit'],'rms_color_limit_98th_percentile':data['rms_color_limit'],'rate_color_reference':'All supported-changing peaks across five intervals, identical to static figures.','source_files':[v[0] for v in METHODS.values()],
              'input_signatures':before,'source_files_unchanged':True,'checks':checks,'passed':True}
    (OUT/'viewer_payload_validation.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({k:v for k,v in manifest.items() if k!='checks'},indent=2))

if __name__=='__main__':build()

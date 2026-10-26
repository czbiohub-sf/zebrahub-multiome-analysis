"""Audit serialized viewer data independently against the saved H5 analysis files."""
from pathlib import Path
import argparse,base64,gzip,hashlib,json
import h5py
import numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=Path(__file__).resolve().parent/'outputs/state_trajectory_atlas_v1');args=p.parse_args();out=args.out
    path=out/'state_trajectory_explorer.html';text=path.read_text();start=text.rindex('const DATA=')+len('const DATA=');payload,_=json.JSONDecoder().raw_decode(text[start:])
    ids=np.load(out/'viewer_sample_indices.npy');arrays={}
    for key,item in payload['arrays'].items():
        dtype=np.uint8 if item['dtype']=='uint8' else np.float32
        arrays[key]=np.frombuffer(gzip.decompress(base64.b64decode(item['gzip_base64'])),dtype=dtype).reshape(item['shape'])
    checked=[]
    def same(key,expected):
        expected=np.asarray(expected,dtype=arrays[key].dtype)
        if not np.array_equal(arrays[key],expected,equal_nan=True):raise AssertionError('Saved data mismatch: '+key)
        checked.append({'array':key,'shape':list(expected.shape),'verified_values':int(expected.size)})
    with h5py.File(out/'sample_data.h5') as f:
        peaks=[x.decode() if isinstance(x,bytes) else str(x) for x in f['peak_ids'][:][ids]]
        assert peaks==[p['peak'] for p in payload['peaks']]
        assert np.array_equal(ids,[p['sample_index'] for p in payload['peaks']])
        for exported,source in [('observed','observed_cpm'),('fitted','fitted_cpm'),('support','rate_supported'),('fitvalid','fit_valid'),('lof','lack_of_fit'),('existing2d','umap2d'),('existing3d','umap3d')]:same(exported,f[source][:][ids])
        same('condition',f['condition_valid'][:])
    with h5py.File(out/'state_pca.h5') as f:
        for panel,meta in payload['panels'].items():
            assert np.array_equal(meta['stages'],f[panel]['stage_indices'][:]);assert np.array_equal(meta['features'],f[panel]['celltype_indices'][:])
            for transform in ['log1p','linear']:
                for source in ['observed','fitted']:same(f'{panel}_{transform}_{source}_pca',f[panel][transform][f'{source}_pca'][:][ids,:,:3])
    if (out/'state_umap.h5').exists():
        with h5py.File(out/'state_umap.h5') as f:
            for panel,meta in payload['panels'].items():
                if not meta['umap']:continue
                for source in ['observed','fitted']:
                    for dim in [2,3]:same(f'{panel}_log1p_{source}_umap{dim}d',f[panel][f'{source}_umap{dim}d'][:][ids])
    assert len(checked)==len(arrays),'Unverified serialized arrays'
    result={'passed':True,'html_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'verified_peak_ids':len(ids),'verified_arrays':checked,'floating_point_serialization':'Viewer arrays equal the saved H5 values cast to float32; missing entries remain NaN.','background_count':len(payload['background']),'panels':payload['panels']}
    folder=out/'browser_validation';folder.mkdir(exist_ok=True);(folder/'state_viewer_export_validation.json').write_text(json.dumps(result,indent=2));print(json.dumps({'passed':True,'arrays':len(checked),'peak_ids':len(ids),'verified_values':sum(a['verified_values'] for a in checked)},indent=2))

if __name__=='__main__':main()

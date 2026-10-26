"""Read-only input audit; never loads the 640K x 640K graph as a dense array."""
import json
from pathlib import Path
import sys
import h5py
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts' / 'chrom_velo'))
from temporal_velocity import SOMITES_TO_HPF, temporal_rates

BASE = ROOT.parent / 'data'
INPUT = BASE / 'annotated_data/objects_v2/peaks_by_ct_tp_master_anno.h5ad'
COACCESS = BASE / 'processed_data/13_peak_umap_analysis/cicero_integrated_ATAC_v2/02_integrated_ATAC_v2_cicero_connections_peaks_integrated_peaks.csv'
OUTPUT = Path(__file__).parent / 'outputs' / 'real_input_audit'


def read_column(group, name):
    g = group[name]
    if isinstance(g, h5py.Dataset):
        return g.asstr()[:] if g.dtype.kind in 'OSU' else g[:]
    categories = g['categories'].asstr()[:] if g['categories'].dtype.kind in 'OSU' else g['categories'][:]
    codes = g['codes'][:]
    if (codes < 0).any():
        raise ValueError(f'Missing metadata values in {name}')
    return categories[codes]


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    report = {'state_path': str(INPUT), 'coaccessibility_path': str(COACCESS)}
    with h5py.File(INPUT) as f:
        peaks = pd.Index(read_column(f['obs'], f['obs'].attrs['_index']))
        conditions = pd.DataFrame({
            'celltype': read_column(f['var'], 'annotation_ML_coarse'),
            'stage': read_column(f['var'], 'dev_stage'),
            'n_cells': read_column(f['var'], 'n_cells'),
        }, index=read_column(f['var'], f['var'].attrs['_index']))
        conditions['time_hpf'] = conditions.stage.map(SOMITES_TO_HPF)
        conditions.to_csv(OUTPUT / 'conditions.csv')
        pairs = set(zip(conditions.celltype, conditions.time_hpf))
        report['shape'] = list(f['X'].shape)
        report['unique_peak_index'] = bool(peaks.is_unique)
        report['missing_conditions'] = [(c, t) for c in sorted(conditions.celltype.unique()) for t in SOMITES_TO_HPF.values() if (c, t) not in pairs]
        report['low_n_conditions'] = conditions[conditions.n_cells < 20].reset_index().to_dict('records')
        report['embeddings'] = {k: list(f['obsm'][k].shape) for k in f['obsm']}
        report['neighbor_graphs'] = {k: list(map(int, f['obsp'][k].attrs['shape'])) for k in f['obsp']}
        n = len(peaks)
        sample_idx = np.linspace(0, n-1, 512, dtype=int)
        sample_x = f['X'][sample_idx]
        norm = f['layers/normalized'][sample_idx]
        log = f['layers/log_norm'][sample_idx]
        report['normalization_sample_512_peaks'] = dict(
            x_equals_normalized=bool(np.allclose(sample_x, norm)),
            x_equals_log_norm=bool(np.allclose(sample_x, log)),
            log_norm_equals_log1p_normalized=bool(np.allclose(log, np.log1p(norm))),
            finite_X=bool(np.isfinite(sample_x).all()))
        report['pca_metadata_available'] = list(f['varm'])
        if 'PCs' in f['varm']:
            pcs = f['varm/PCs'][:]
            mean, std = read_column(f['var'], 'mean'), read_column(f['var'], 'std')
            stored = f['obsm/X_pca'][sample_idx]
            report['pca_reconstruction_sample'] = {}
            for label, values in [('normalized', norm), ('log_norm', log)]:
                predicted = ((values - mean)/std) @ pcs
                relative_rmse = np.sqrt(np.mean((predicted-stored)**2)) / np.sqrt(np.mean(stored**2))
                report['pca_reconstruction_sample'][label+'_relative_rmse_using_saved_scaling'] = float(relative_rmse)
        for minimum in [None, 20]:
            _, mask, _ = temporal_rates(sample_x[:1], conditions, min_cells=minimum)
            report[f'valid_forward_columns_min_cells_{minimum}'] = int(mask.sum())
        # Scan values in bounded blocks to distinguish zeros from NaNs in present columns.
        nonfinite, zeros = 0, 0
        for start in range(0, n, 8192):
            x = f['X'][start:start+8192]
            nonfinite += int((~np.isfinite(x)).sum())
            zeros += int((x == 0).sum())
        report['X_nonfinite_entries'] = nonfinite
        report['X_zero_entries'] = zeros
    print('Scanned state; auditing Cicero endpoint alignment.', flush=True)
    coverage = np.zeros(len(peaks), dtype=bool)
    counts = dict(rows=0, cis_rows=0, trans_rows=0, rows_both_endpoints_in_atlas=0, nonfinite_scores=0)
    thresholds = {str(t): 0 for t in [0.25, 0.5, 0.8]}
    coverage_by_threshold = {t: np.zeros(len(peaks), dtype=bool) for t in thresholds}
    for chunk in pd.read_csv(COACCESS, chunksize=200_000):
        left = peaks.get_indexer(chunk.Peak1)
        right = peaks.get_indexer(chunk.Peak2)
        matched = (left >= 0) & (right >= 0)
        coverage[left[left >= 0]] = True
        coverage[right[right >= 0]] = True
        cis = chunk.Peak1.str.rsplit('-', n=2).str[0] == chunk.Peak2.str.rsplit('-', n=2).str[0]
        counts['rows'] += len(chunk)
        counts['cis_rows'] += int(cis.sum())
        counts['trans_rows'] += int((~cis).sum())
        counts['rows_both_endpoints_in_atlas'] += int(matched.sum())
        counts['nonfinite_scores'] += int((~np.isfinite(chunk.coaccess)).sum())
        for t in thresholds:
            selected = matched & (chunk.coaccess.to_numpy() > float(t))
            thresholds[t] += int(selected.sum())
            coverage_by_threshold[t][left[selected]] = True
            coverage_by_threshold[t][right[selected]] = True
    report['coaccessibility'] = dict(**counts, peaks_with_any_endpoint=int(coverage.sum()),
                                    atlas_peaks_without_any_edge=int((~coverage).sum()),
                                    matched_rows_above_threshold=thresholds,
                                    peaks_connected_above_threshold={t: int(m.sum()) for t, m in coverage_by_threshold.items()})
    report['alignment_policy'] = 'Atlas row order is canonical; map Cicero endpoints by peak ID. Retain isolates with a self edge; never drop most of the atlas merely because edges are absent.'
    (OUTPUT / 'audit.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()

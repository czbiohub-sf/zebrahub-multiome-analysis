"""Time-resolved accessibility rates on a fixed peak atlas.

These are retrospective changes, not RNA splicing kinetics or peak-to-peak
physical motion. Co-accessibility regularizes a measured temporal signal;
it does not supply a direction of time. See the toy notebook for a counterexample
to interpreting profile-shift arrows as early-to-late program succession.
"""

import numpy as np
import pandas as pd
from scipy import sparse

STAGES_HPF = np.array([10., 12., 14., 16., 19., 24.])
SOMITES_TO_HPF = dict(zip(
    ['0somites', '5somites', '10somites', '15somites', '20somites', '30somites'],
    STAGES_HPF,
))


def temporal_rates(x, conditions, stages=STAGES_HPF, divide_by_dt=True,
                   form='forward', min_cells=None):
    """Return same-shaped rates, validity mask, and per-column interval lengths.

    `conditions` has explicit celltype and time_hpf columns in the order of X.
    Only adjacent *scheduled* stages are differenced. Missing/low-cell-count
    endpoints invalidate an interval; an absent stage is never bridged or zero
    imputed. Terminal zeros are storage placeholders, excluded by the mask.
    `slope` fits a per-celltype OLS summary in hpf (not a time-resolved field).
    """
    x = np.asarray(x, dtype=float)
    stages = np.asarray(stages, dtype=float)
    if x.ndim != 2 or x.shape[1] != len(conditions):
        raise ValueError('X columns must match the condition rows')
    if len(stages) < 2 or not np.all(np.isfinite(stages)) or np.any(np.diff(stages) <= 0):
        raise ValueError('Stages must be finite and strictly increasing')
    if form not in ('forward', 'slope'):
        raise ValueError('form must be forward or slope')
    if form == 'slope' and not divide_by_dt:
        raise ValueError('A slope is a rate per hpf; divide_by_dt must be True')
    ct = conditions['celltype'].astype(str).to_numpy()
    time = conditions['time_hpf'].to_numpy(dtype=float)
    if conditions['celltype'].isna().any() or not np.isin(time, stages).all():
        raise ValueError('Each condition needs a celltype and a scheduled time')
    keys = list(zip(ct, time))
    if len(set(keys)) != len(keys):
        raise ValueError('Duplicate celltype/time conditions; aggregate replicates explicitly')
    reliable = np.ones(len(time), dtype=bool)
    if min_cells is not None:
        reliable = conditions['n_cells'].to_numpy(dtype=float) >= min_cells
    v = np.zeros_like(x)
    valid = np.zeros(x.shape, dtype=bool)
    dt = np.zeros(len(time))
    lookup = dict(zip(keys, range(len(keys))))
    if form == 'forward':
        for j, (c, t) in enumerate(keys):
            pos = np.flatnonzero(stages == t)[0]
            if pos == len(stages) - 1:
                continue
            next_t = stages[pos + 1]
            k = lookup.get((c, next_t))
            if k is None or not (reliable[j] and reliable[k]):
                continue
            dt[j] = next_t - t
            valid[:, j] = np.isfinite(x[:, j]) & np.isfinite(x[:, k])
            v[valid[:, j], j] = (x[valid[:, j], k] - x[valid[:, j], j]) / (
                dt[j] if divide_by_dt else 1.)
    else:
        for c in np.unique(ct):
            cols = np.flatnonzero((ct == c) & reliable)
            for i in range(len(x)):
                good = cols[np.isfinite(x[i, cols])]
                if len(good) < 2:
                    continue
                t = time[good] - time[good].mean()
                v[i, good] = t @ x[i, good] / (t @ t)
                valid[i, good] = True
    return v, valid, dt


def cis_weights(graph, chromosomes, threshold=0.35):
    """Positive, symmetric cis graph with one self edge, row-normalized.

    The graph MUST already have the same ordered peak index as X. Reject
    trans edges instead of silently assuming that an input graph is cis-only.
    Isolated peaks retain their own rates. Never densifies the peak graph.
    """
    if not sparse.issparse(graph):
        raise TypeError('Co-accessibility must be a sparse matrix')
    n = len(chromosomes)
    if graph.shape != (n, n) or not 0 <= threshold <= 1:
        raise ValueError('Graph shape or threshold is invalid')
    g = graph.tocoo(copy=True)
    chrom = np.asarray(chromosomes)
    if not np.isfinite(g.data).all():
        raise ValueError('Nonfinite co-accessibility scores')
    if np.any(chrom[g.row] != chrom[g.col]):
        raise ValueError('Co-accessibility contains trans-chromosome edges')
    keep = (g.data > threshold) & (g.data > 0)
    g = sparse.csr_matrix((g.data[keep], (g.row[keep], g.col[keep])), shape=g.shape)
    g = g.maximum(g.T)
    g.setdiag(1.)
    g.eliminate_zeros()
    return sparse.diags(1. / np.asarray(g.sum(axis=1)).ravel()) @ g


def couple_rates(v, valid, weights, alpha=0.6):
    """Mask-aware regulatory first moment and intrinsic/coupled mixture.

    Unlike expression-kNN first moments, weights come from cis co-accessibility.
    An invalid focal derivative remains invalid even if neighbors have data.
    """
    if not 0 <= alpha <= 1:
        raise ValueError('alpha must lie in [0, 1]')
    if weights.shape != (len(v), len(v)) or valid.shape != v.shape:
        raise ValueError('Rate, mask, and graph shapes do not match')
    if not np.isfinite(v[valid]).all():
        raise ValueError('Valid derivatives must be finite')
    denominator = weights @ valid.astype(float)
    smoothed = np.divide(weights @ np.where(valid, v, 0.), denominator,
                         out=np.zeros_like(v), where=denominator > 0)
    final = np.where(valid, alpha * v + (1. - alpha) * smoothed, 0.)
    return final, np.where(valid, smoothed, 0.)


def rate_summary(v, valid):
    """Equal-condition mean rate and RMS rate; unsupported rows stay NaN.

    RMS avoids making magnitudes bigger simply because more intervals exist.
    Positive and negative components are retained so pulses cannot disappear
    behind a near-zero signed mean. This is not a duration-weighted net change.
    """
    n = valid.sum(axis=1)
    def mean(values):
        return np.divide(np.where(valid, values, 0.).sum(axis=1), n,
                         out=np.full(len(v), np.nan), where=n > 0)
    return pd.DataFrame(dict(
        signed_rate=mean(v), magnitude_rms=np.sqrt(mean(v * v)),
        opening_rate=mean(np.maximum(v, 0.)),
        closing_rate=mean(np.minimum(v, 0.)), n_valid=n,
    ))


def project_scvelo(adata, valid, bases=('umap',), vkey='velocity'):
    """Use scVelo only for projection, preserving the supplied atlas/graph.

    scVelo centers each velocity across features (cosine *correlation*), so a
    common opening component may be lost. Keep signed rates outside projection.
    Undefined terminal and missing-interval features are excluded, not just
    filled with zeros. scVelo lacks per-edge feature masks: this conservative
    adapter uses features valid for every peak and reports the retained count.
    """
    import scvelo as scv

    if 'neighbors' not in adata.uns or 'distances' not in adata.obsp:
        raise ValueError('Supply the existing high-dimensional neighbor graph')
    selected = valid.all(axis=0)
    if selected.sum() < 2:
        raise ValueError('Need at least two globally valid features for scVelo projection')
    if not np.isfinite(np.asarray(adata.X)[:, selected]).all():
        raise ValueError('Selected projection states contain nonfinite values')
    for basis in bases:
        if f'X_{basis}' not in adata.obsm:
            raise ValueError(f'Missing supplied embedding X_{basis}')
    # scVelo cannot concatenate an entirely empty cosine result. A globally
    # zero (or feature-constant) field has no centered directional component.
    values = np.asarray(adata.layers[vkey])[:, selected]
    if not np.isfinite(values).all():
        raise ValueError('Selected projection velocities contain nonfinite values')
    if not np.any(np.ptp(values, axis=1) > 1e-12):
        adata.var[f'{vkey}_projection_feature'] = selected
        adata.uns[f'{vkey}_graph'] = sparse.csr_matrix((len(adata), len(adata)))
        adata.uns[f'{vkey}_graph_neg'] = sparse.csr_matrix((len(adata), len(adata)))
        adata.obs[f'{vkey}_self_transition'] = 1.
        for basis in bases:
            adata.obsm[f'{vkey}_{basis}'] = np.zeros_like(adata.obsm[f'X_{basis}'])
        adata.uns[f'{vkey}_interpretation'] = {
            'meaning': 'No centered directional component; inspect signed rates separately',
            'n_features': int(selected.sum()), 'scvelo_version': scv.__version__,
        }
        return adata
    adata.layers['state'] = adata.X.copy()
    adata.var[f'{vkey}_projection_feature'] = selected
    scv.tl.velocity_graph(
        adata, vkey=vkey, xkey='state', gene_subset=adata.var_names[selected],
        sqrt_transform=False, approx=False, n_recurse_neighbors=1, n_jobs=1,
    )
    for basis in bases:
        if f'X_{basis}' not in adata.obsm:
            raise ValueError(f'Missing supplied embedding X_{basis}')
        scv.tl.velocity_embedding(
            adata, vkey=vkey, basis=basis, autoscale=False, retain_scale=False,
            use_negative_cosines=False, all_comps=True,
        )
    adata.uns[f'{vkey}_interpretation'] = {
        'meaning': 'Retrospective profile-shift similarity; not peak motion or developmental fate',
        'projection': 'scVelo centered cosine correlation; positive transitions only',
        'n_features': int(selected.sum()),
        'scvelo_version': scv.__version__,
    }
    return adata

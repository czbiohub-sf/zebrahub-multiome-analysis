"""Controlled synthetic peak x (celltype, stage) accessibility benchmark.

All celltypes, genomic coordinates, regulatory modules, and noise parameters are
synthetic. Scores lie in (0, 1); this is not a simulator of fragments or single
cells. Independent beta sampling replicates have expectation equal to the saved
latent accessibility. Cell counts/depth control precision, not mean accessibility.
"""
from dataclasses import asdict, dataclass

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.special import expit

from temporal_velocity import STAGES_HPF, temporal_rates

PROGRAMS = (
    'opening', 'closing', 'transient_opening', 'transient_closing',
    'late_opening', 'opposite_celltypes', 'stable_specific', 'stable_broad',
)
CELLTYPES = tuple(f'CT{i+1:02d}' for i in range(20))
MISSING_STRESS = (('CT20', 10.), ('CT20', 12.), ('CT08', 14.), ('CT12', 24.))


@dataclass(frozen=True)
class SyntheticConfig:
    seed: int = 42
    n_modules: int = 80
    peaks_per_module: int = 30
    n_replicates: int = 3
    # Four related families of five celltypes. These are similarity groups,
    # not an assumed lineage tree or a biological ancestor/descendant claim.
    related_families: bool = True
    noise_precision: float = 1.5
    graph_retention: float = 0.75
    graph_false_edge_fraction: float = 0.15
    graph_isolated_fraction: float = 0.1

    def validate(self):
        if self.n_modules < len(PROGRAMS) or self.n_modules % len(PROGRAMS):
            raise ValueError('n_modules must be a positive multiple of eight')
        if self.peaks_per_module < 4 or self.n_replicates < 1:
            raise ValueError('Need >=4 peaks/module and >=1 sampling replicate')
        if self.noise_precision <= 0:
            raise ValueError('noise_precision must be positive')
        for name in ['graph_retention', 'graph_isolated_fraction']:
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f'{name} must lie in [0,1]')
        if not 0 <= self.graph_false_edge_fraction < 1:
            raise ValueError('graph_false_edge_fraction must lie in [0,1)')


def latent_parameters(config):
    """Separate random streams keep the ground truth fixed when noise changes."""
    config.validate()
    rng = np.random.default_rng(np.random.SeedSequence([config.seed, 0]))
    per_module = config.peaks_per_module
    n = config.n_modules * per_module
    rows = []
    loadings = np.zeros((n, len(CELLTYPES)))
    lags = np.zeros_like(loadings)
    baseline = np.zeros_like(loadings)
    for module in range(config.n_modules):
        # Alternate types across chromosomes to avoid chromosome == program labels.
        program = PROGRAMS[module % len(PROGRAMS)]
        primary = (module * 7 + module // len(PROGRAMS)) % len(CELLTYPES)
        partner = (primary + 5) % len(CELLTYPES)
        family = primary // 5
        scope = ['single', 'family', 'broad'][(module // len(PROGRAMS)) % 3]
        if not config.related_families and scope == 'family':
            scope = 'single'
        weights = np.zeros(len(CELLTYPES))
        if scope == 'single':
            weights[primary] = 1.
        elif scope == 'family':
            weights[family*5:family*5+5] = rng.uniform(0.45, 0.95, 5)
            weights[primary] = 1.
        else:
            weights[:] = rng.uniform(0.25, 0.9, len(CELLTYPES))
            weights[primary] = 1.
        if program == 'opposite_celltypes':
            scope = 'opposite_pair'
            weights[:] = 0.
            weights[[primary, partner]] = 1.
        if program == 'stable_specific':
            scope = 'single'
            weights[:] = 0.
            weights[primary] = 1.
        if program == 'stable_broad':
            scope = 'broad'
            weights[:] = 1.
        center = rng.uniform(12.5, 21.5)
        if program in ('transient_opening', 'transient_closing'):
            center = rng.uniform(14., 19.)
        if program == 'late_opening':
            center = rng.uniform(20.5, 22.)
        width = rng.uniform(1.2, 2.6)
        module_lags = rng.normal(0., 0.6, len(CELLTYPES))
        module_lags[primary] = 0.
        if program == 'opposite_celltypes':
            module_lags[partner] = 0.  # Exact cancellation in the paired truth.
        chrom = f'chr{module % 5 + 1}'
        block = (module // 5 + 1) * 100_000
        base_ct = rng.uniform(0.04, 0.15, len(CELLTYPES))
        for k in range(per_module):
            i = module*per_module + k
            # Within-module variation creates a continuum, not duplicated rows.
            amplitude = rng.uniform(0.40, 0.68)
            peak_weights = weights * rng.uniform(0.85, 1.0)
            loadings[i] = peak_weights
            lags[i] = module_lags
            baseline[i] = base_ct + rng.uniform(-0.015, 0.015, len(CELLTYPES))
            rows.append(dict(
                peak=f'{chrom}-{block+600*k}-{block+600*k+300}',
                chrom=chrom, start=block+600*k, end=block+600*k+300,
                true_module=f'M{module:03d}', true_program=program, true_scope=scope,
                primary_celltype=CELLTYPES[primary], partner_celltype=CELLTYPES[partner],
                true_center_hpf=center+rng.uniform(-0.7, 0.7),
                true_width_hpf=width*rng.uniform(0.85, 1.15), true_amplitude=amplitude,
            ))
    obs = pd.DataFrame(rows).set_index('peak')
    return obs, loadings, lags, baseline


def evaluate_truth(obs, loadings, lags, baseline, times):
    """Continuous mean and analytic instantaneous derivative, both peak x CT x t."""
    t = np.asarray(times, dtype=float)[None, None, :]
    center = obs.true_center_hpf.to_numpy()[:, None, None] + lags[:, :, None]
    width = obs.true_width_hpf.to_numpy()[:, None, None]
    z = (t - center) / width
    sigmoid = expit(z)
    pulse = np.exp(-0.5*z*z)
    f = sigmoid.copy()
    df = sigmoid*(1-sigmoid)/width
    kinds = obs.true_program.to_numpy()
    closing = kinds == 'closing'
    f[closing] = 1-sigmoid[closing]
    df[closing] *= -1
    for kind, invert in [('transient_opening', False), ('transient_closing', True)]:
        idx = kinds == kind
        f[idx] = 1-pulse[idx] if invert else pulse[idx]
        df[idx] = (1 if invert else -1)*z[idx]*pulse[idx]/width[idx]
    for i in np.flatnonzero(kinds == 'opposite_celltypes'):
        c = CELLTYPES.index(obs.partner_celltype.iloc[i])
        f[i, c] = 1-sigmoid[i, c]
        df[i, c] *= -1
    stable = np.isin(kinds, ['stable_specific', 'stable_broad'])
    f[stable] = 0.8
    df[stable] = 0.
    amp = obs.true_amplitude.to_numpy()[:, None, None] * loadings[:, :, None]
    return baseline[:, :, None] + amp*f, amp*df


def regulatory_graph(obs, config):
    """Planted cis edges, incomplete coverage, and overlapping-score wrong edges.

    Modules define candidate relationships independently of any estimated velocity.
    Wrong edges link different modules on the same chromosome. Their scores overlap
    true-edge scores, so a threshold cannot magically recover the true graph.
    """
    rng = np.random.default_rng(np.random.SeedSequence([config.seed, 1]))
    n = len(obs)
    pairs = {}
    for _, group in obs.groupby('true_module', sort=False):
        idx = obs.index.get_indexer(group.index)
        for i in idx:
            for j in rng.choice(idx[idx != i], min(5, len(idx)-1), replace=False):
                pairs[tuple(sorted((int(i), int(j))))] = rng.uniform(0.3, 0.95)
    truth_pairs = pairs.copy()
    isolated = rng.random(n) < config.graph_isolated_fraction
    pairs = {key: score for key, score in pairs.items()
             if rng.random() < config.graph_retention and not isolated[list(key)].any()}
    n_wrong = int(len(pairs)*config.graph_false_edge_fraction/(1-config.graph_false_edge_fraction))
    chrom = obs.chrom.to_numpy()
    modules = obs.true_module.to_numpy()
    wrong = 0
    for _ in range(max(100, n_wrong*30)):
        if wrong >= n_wrong:
            break
        i = int(rng.integers(n))
        candidates = np.flatnonzero((chrom == chrom[i]) & (modules != modules[i]) & ~isolated)
        if isolated[i] or not len(candidates):
            continue
        j = int(rng.choice(candidates))
        key = tuple(sorted((i, j)))
        if key not in pairs:
            pairs[key] = rng.uniform(0.2, 0.9)
            wrong += 1
    def matrix(edges):
        if not edges:
            return sparse.csr_matrix((n, n), dtype='float32')
        indices = np.array(list(edges))
        score = np.array(list(edges.values()), dtype='float32')
        return sparse.csr_matrix((np.r_[score, score],
            (np.r_[indices[:, 0], indices[:, 1]], np.r_[indices[:, 1], indices[:, 0]])), shape=(n, n))
    return matrix(pairs), matrix(truth_pairs), dict(
        observed_undirected_edges=len(pairs), planted_candidate_edges=len(truth_pairs),
        wrong_undirected_edges=wrong, prescribed_isolated_peaks=int(isolated.sum()),
        score_ranges_overlap=True,
    )


def make_dataset(config=SyntheticConfig(), scenario='noisy'):
    """Return an aligned AnnData; clean/noisy share 120 columns, stress has 116.

    X is the mean of independent simulated sampling replicates. Replicate layers
    are continuous beta draws, not single-cell matrices or biological replicates.
    The beta precision scales with simulated cell count and depth, leaving E[X]
    equal to the latent score. Noisy and stress scenarios share latent programs.
    """
    if scenario not in ('clean', 'noisy', 'stress'):
        raise ValueError('scenario must be clean, noisy, or stress')
    obs, loading, lags, baseline = latent_parameters(config)
    truth, instantaneous = evaluate_truth(obs, loading, lags, baseline, STAGES_HPF)
    truth = truth.reshape(len(obs), -1)
    instantaneous = instantaneous.reshape(len(obs), -1)
    var = pd.DataFrame([(c, t) for c in CELLTYPES for t in STAGES_HPF],
                       columns=['celltype', 'time_hpf'])
    # Explicit 5-member groups (avoid inferring groups from integer codes elsewhere).
    var['family'] = [f'Family{CELLTYPES.index(c)//5+1}' for c in var.celltype]
    var.index = [f'{c}_{t:g}hpf' for c, t in var[['celltype', 'time_hpf']].to_numpy()]
    rng = np.random.default_rng(np.random.SeedSequence([config.seed, 2]))
    counts = np.clip(rng.lognormal(np.log(65), 0.65, (config.n_replicates, len(var))), 20, 400).astype(int)
    depth = rng.lognormal(-0.5*0.3**2, 0.3, counts.shape)
    if scenario == 'stress':
        for c, t in [('CT04', 24.), ('CT13', 12.)]:
            j = np.flatnonzero((var.celltype == c) & (var.time_hpf == t))[0]
            counts[:, j] = 3
    var['n_cells'] = counts.sum(0)
    var['mean_depth_factor'] = depth.mean(0)
    observed = []
    for r in range(config.n_replicates):
        var[f'n_cells_rep{r+1}'] = counts[r]
        var[f'depth_rep{r+1}'] = depth[r]
        precision = counts[r]*depth[r]*config.noise_precision*(0.5 if scenario == 'stress' else 1.)
        values = truth.copy() if scenario == 'clean' else rng.beta(truth*precision, (1-truth)*precision)
        observed.append(values.astype('float32'))
    measured = truth.astype('float32') if scenario == 'clean' else np.mean(observed, axis=0).astype('float32')
    a = ad.AnnData(measured, obs=obs, var=var)
    a.layers['truth_state'] = truth.astype('float32')
    a.layers['truth_instantaneous_rate'] = instantaneous.astype('float32')
    for r, values in enumerate(observed):
        a.layers[f'sampling_rep{r+1}'] = values
    a.obsm['true_celltype_loadings'] = loading.astype('float32')
    a.obsm['true_celltype_lags_hpf'] = lags.astype('float32')
    a.obsm['true_celltype_baseline'] = baseline.astype('float32')
    graph, truth_graph, graph_report = regulatory_graph(obs, config)
    a.obsp['coaccessibility'] = graph
    a.obsp['true_cis_candidates'] = truth_graph
    a.uns['synthetic'] = dict(**asdict(config), scenario=scenario,
        celltypes=list(CELLTYPES), stages_hpf=STAGES_HPF.tolist(), graph=graph_report,
        model='Continuous bounded accessibility; independent beta sampling replicates',
        missing_conditions=[])
    if scenario == 'stress':
        keep = np.array([(c, t) not in MISSING_STRESS for c, t in zip(var.celltype, var.time_hpf)])
        a = a[:, keep].copy()
        a.uns['synthetic']['missing_conditions'] = [f'{c}_{t:g}hpf' for c, t in MISSING_STRESS]
    v, mask, dt = temporal_rates(a.layers['truth_state'], a.var)
    a.layers['truth_velocity'] = v.astype('float32')
    a.layers['truth_velocity_valid'] = mask
    a.var['interval_hpf'] = dt
    return a

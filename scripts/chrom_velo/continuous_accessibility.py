"""A connected family of peak accessibility profiles, with a constitutive core.

Each row is a fixed genomic peak, not a cell or an ancestral state. A synthetic
specialization coordinate indexes different peaks; developmental time remains a
separate axis inside each peak's 120-dimensional accessibility profile.
"""
from dataclasses import asdict, dataclass

import anndata as ad
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.neighbors import NearestNeighbors
from scipy import sparse

from synthetic_accessibility import CELLTYPES
from temporal_velocity import STAGES_HPF, temporal_rates


@dataclass(frozen=True)
class ContinuousConfig:
    seed: int = 42
    n_constitutive: int = 200
    n_per_family_trunk: int = 200
    n_per_celltype_branch: int = 80
    n_replicates: int = 3
    noise_precision: float = 1.5
    root_accessibility: float = 0.65

    def validate(self):
        if min(self.n_constitutive, self.n_per_family_trunk, self.n_per_celltype_branch) < 4:
            raise ValueError('Each population needs at least four peaks')
        if self.n_replicates < 1 or self.noise_precision <= 0:
            raise ValueError('Invalid replicate count or noise precision')
        if not 0.58 <= self.root_accessibility <= 0.72:
            raise ValueError('root_accessibility must lie in [0.58,0.72] for the fixed contrasts')


def normalized_gate(times, center, width):
    """Monotone sigmoid, exactly 0 at 10 hpf and 1 at 24 hpf, and its derivative."""
    t = np.asarray(times, dtype=float)
    lo, hi = expit((10-center)/width), expit((24-center)/width)
    sig = expit((t-center)/width)
    return (sig-lo)/(hi-lo), sig*(1-sig)/(width*(hi-lo))


def hierarchy_parameters(config):
    config.validate()
    rng = np.random.default_rng(np.random.SeedSequence([config.seed, 10]))
    rows = []
    def append(role, family, celltype, coordinate):
        i = len(rows)
        if role == 'constitutive_core':
            wf, wc, specialization = 0., 0., 0.
        elif role == 'family_bridge':
            wf, wc, specialization = coordinate, 0., coordinate
        else:
            wf, wc, specialization = 1-coordinate, coordinate, 1+coordinate
        # Timing perturbations taper away at junctions, preserving shared limits.
        jitter = rng.normal(0,0.18)*np.sin(np.pi*coordinate)
        rows.append(dict(
            peak=f'chr{i%5+1}-{1000+i*500}-{1300+i*500}', chrom=f'chr{i%5+1}',
            true_program=role, true_scope='broad' if role=='constitutive_core' else 'graded_specificity',
            true_family=family, primary_celltype=CELLTYPES[max(0,celltype)],
            partner_celltype=CELLTYPES[(max(0,celltype)+5)%20],
            true_celltype_index=celltype, true_specialization=specialization,
            true_family_weight=wf, true_celltype_weight=wc,
            true_baseline=config.root_accessibility+rng.uniform(-0.005,0.005),
            true_amplitude_scale=rng.uniform(0.985,1.015),
            true_family_center_hpf=15.5+jitter, true_celltype_center_hpf=18.5+jitter,
            true_family_width_hpf=2.0, true_celltype_width_hpf=1.7,
            model='continuous_hierarchy',
        ))
    for _ in range(config.n_constitutive):
        append('constitutive_core',-1,-1,0.)
    # Stratified continuous sampling avoids unobserved gaps between junctions.
    for family in range(4):
        position=(np.arange(config.n_per_family_trunk)+rng.uniform(0.05,0.95,config.n_per_family_trunk))/config.n_per_family_trunk
        for s in position:
            append('family_bridge',family,family*5,float(s))
    for celltype in range(20):
        position=(np.arange(config.n_per_celltype_branch)+rng.uniform(0.05,0.95,config.n_per_celltype_branch))/config.n_per_celltype_branch
        for u in position:
            append('celltype_branch',celltype//5,celltype,float(u))
    return pd.DataFrame(rows).set_index('peak')


def evaluate_hierarchy(obs, times):
    """Noise-free peak x celltype x time profiles and exact instantaneous rates.

    R -> F -> C is a family of *different peak profiles*. The family gate changes
    earlier than the celltype gate. Intermediate peaks can transiently open in
    related celltypes before closing, while the target celltype keeps opening.
    """
    times=np.asarray(times,dtype=float)
    n=len(obs)
    state=np.empty((n,20,len(times)))
    rate=np.empty_like(state)
    for i,(_,p) in enumerate(obs.iterrows()):
        g,dg=normalized_gate(times,p.true_family_center_hpf,p.true_family_width_hpf)
        h,dh=normalized_gate(times,p.true_celltype_center_hpf,p.true_celltype_width_hpf)
        family_delta=np.where(np.arange(20)//5==p.true_family,0.18,-0.55)
        cell_delta=np.where(np.arange(20)==p.true_celltype_index,0.23,-0.55)
        a=p.true_family_weight*family_delta[:,None]
        b=p.true_celltype_weight*cell_delta[:,None]
        state[i]=p.true_baseline+p.true_amplitude_scale*(a*g+b*h)
        rate[i]=p.true_amplitude_scale*(a*dg+b*dh)
    return state,rate


def local_cis_graph(truth, chromosomes, seed):
    """Synthetic regulatory-neighborhood surrogate, separate from atlas kNN.

    Neighbor candidates use noiseless profiles within each chromosome. This is a
    favorable simulation assumption, not inferred Cicero and not a causal graph.
    Drop 25% of candidate edges and isolate 10% of peaks to retain coverage controls.
    """
    rng=np.random.default_rng(np.random.SeedSequence([seed,12]))
    rows,cols,scores=[],[],[]
    isolated=rng.random(len(truth))<0.1
    for chrom in np.unique(chromosomes):
        idx=np.flatnonzero(chromosomes==chrom)
        k=min(7,len(idx))
        neighbors=NearestNeighbors(n_neighbors=k,n_jobs=1).fit(truth[idx]).kneighbors(truth[idx],return_distance=False)
        for i,neighbors_i in enumerate(neighbors):
            for j in neighbors_i:
                src,dst=int(idx[i]),int(idx[j])
                if src==dst or isolated[src] or isolated[dst] or rng.random()<0.25:
                    continue
                rows.append(src);cols.append(dst);scores.append(rng.uniform(0.3,0.9))
    graph=sparse.csr_matrix((scores,(rows,cols)),shape=(len(truth),len(truth)),dtype='float32')
    return graph.maximum(graph.T)


def make_continuous(config=ContinuousConfig()):
    obs=hierarchy_parameters(config)
    state,rate=evaluate_hierarchy(obs,STAGES_HPF)
    truth=state.reshape(len(obs),120)
    var=pd.DataFrame([(c,t) for c in CELLTYPES for t in STAGES_HPF],columns=['celltype','time_hpf'])
    var.index=[f'{c}_{t:g}hpf' for c,t in var.to_numpy()]
    var['family']=[f'Family{CELLTYPES.index(c)//5+1}' for c in var.celltype]
    rng=np.random.default_rng(np.random.SeedSequence([config.seed,11]))
    counts=np.clip(rng.lognormal(np.log(65),0.65,(config.n_replicates,120)),20,400).astype(int)
    depth=rng.lognormal(-0.5*0.3**2,0.3,counts.shape)
    var['n_cells']=counts.sum(0)
    observed=[]
    for r in range(config.n_replicates):
        precision=counts[r]*depth[r]*config.noise_precision
        observed.append(rng.beta(truth*precision,(1-truth)*precision).astype('float32'))
        var[f'n_cells_rep{r+1}']=counts[r]
        var[f'depth_rep{r+1}']=depth[r]
    a=ad.AnnData(np.mean(observed,axis=0).astype('float32'),obs=obs,var=var)
    a.layers['truth_state']=truth.astype('float32')
    a.layers['truth_instantaneous_rate']=rate.reshape(len(obs),120).astype('float32')
    for r,x in enumerate(observed):
        a.layers[f'sampling_rep{r+1}']=x
    v,mask,dt=temporal_rates(a.layers['truth_state'],var)
    a.layers['truth_velocity']=v.astype('float32')
    a.layers['truth_velocity_valid']=mask
    a.var['interval_hpf']=dt
    a.obsp['coaccessibility']=local_cis_graph(truth,obs.chrom.to_numpy(),config.seed)
    a.uns['synthetic']=dict(**asdict(config),scenario='continuous_hierarchy',
        celltypes=list(CELLTYPES),stages_hpf=STAGES_HPF.tolist(),
        root_meaning='Constitutive broad accessibility reference; not a temporal ancestor',
        specialization_meaning='Peak-profile coordinate independent of developmental time',
        graph_model='Incomplete cis-restricted neighbor graph from noiseless profiles',
        model='root + family_weight * family_contrast(t) + celltype_weight * celltype_contrast(t)')
    return a

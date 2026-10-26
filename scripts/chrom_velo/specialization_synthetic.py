"""Synthetic peak atlas with independent temporal and specificity controls.

The inherited connected hierarchy describes a family of *different fixed peaks*.
Control peaks deliberately violate a universal root-to-tip temporal narrative:
constitutive and stable-specific peaks do not change, uniform opening need not
increase specificity, and transient programs reverse. Coordinates and links are
never planted in the visualization. The optional cis graph is a favorable
noiseless-profile surrogate, not measured Cicero co-accessibility.
"""
from dataclasses import asdict

import anndata as ad
import numpy as np
import pandas as pd

from continuous_accessibility import (
    ContinuousConfig,
    evaluate_hierarchy,
    hierarchy_parameters,
    local_cis_graph,
    normalized_gate,
)
from synthetic_accessibility import CELLTYPES
from temporal_velocity import STAGES_HPF, temporal_rates


CONTROL_KINDS = (
    "stable_celltype_specific",
    "uniform_opening",
    "lineage_opening",
    "transient_lineage",
    "low_signal",
)


def control_parameters(seed, n_per_control_family, start_index):
    """Make genuine genomic identifiers and explicit, interpretable truth labels.

    Uniform-opening and low-signal peaks are generated in four sampling batches,
    but do not belong to a biological family; their true family/celltype is -1.
    """
    if n_per_control_family < 1:
        raise ValueError("n_per_control_family must be at least one")
    rng = np.random.default_rng(np.random.SeedSequence([seed, 101]))
    rows = []
    for kind in CONTROL_KINDS:
        for family in range(4):
            for j in range(n_per_control_family):
                i = start_index + len(rows)
                is_targeted = kind in (
                    "stable_celltype_specific", "lineage_opening", "transient_lineage"
                )
                target = family * 5 + j % 5 if is_targeted else -1
                base = rng.uniform(0.04, 0.06)
                amplitude = rng.uniform(0.65, 0.78)
                if kind == "uniform_opening":
                    base, amplitude = rng.uniform(0.10, 0.14), rng.uniform(0.55, 0.65)
                elif kind == "low_signal":
                    base, amplitude = rng.uniform(0.003, 0.009), 0.0
                rows.append(dict(
                    peak=f"chr{i % 5 + 1}-{1000 + i * 500}-{1300 + i * 500}",
                    chrom=f"chr{i % 5 + 1}",
                    true_program=kind,
                    true_scope="celltype_specific" if is_targeted else "broad",
                    true_family=family if is_targeted else -1,
                    primary_celltype=CELLTYPES[target] if is_targeted else "all_celltypes",
                    partner_celltype="none",
                    true_celltype_index=target,
                    true_specialization=np.nan,
                    true_control_baseline=base,
                    true_control_amplitude=amplitude,
                    true_control_center_hpf=(18.0 if kind == "lineage_opening" else 16.0)
                    + rng.uniform(-0.3, 0.3),
                    true_control_width_hpf=rng.uniform(1.5, 2.1),
                    model="specialization_control",
                    truth_model="specialization_control",
                    control_kind=kind,
                    is_hierarchy=False,
                    truth_module=f"{kind}_{CELLTYPES[target]}" if is_targeted else kind,
                ))
    return pd.DataFrame(rows).set_index("peak")


def evaluate_specialization(obs, times):
    """Return exact accessibility and derivative as peak x celltype x time arrays.

    Valid only for the simulated developmental interval, 10 through 24 hpf.
    Nothing is clipped: invalid parameter choices fail at construction instead of
    silently flattening the state while retaining an incompatible derivative.
    Transient target-celltype accessibility follows sin(pi * (t-10)/14)^2,
    returning exactly to its uniform background at the final observed stage.
    """
    times = np.asarray(times, dtype=float)
    if times.ndim != 1 or not np.all(np.isfinite(times)):
        raise ValueError("times must be a finite one-dimensional array")
    if np.any((times < 10) | (times > 24)):
        raise ValueError("Synthetic truth is defined only on [10, 24] hpf")
    state = np.empty((len(obs), len(CELLTYPES), len(times)), dtype=float)
    rate = np.empty_like(state)
    hierarchy = obs.truth_model.to_numpy() == "continuous_hierarchy"
    if hierarchy.any():
        state[hierarchy], rate[hierarchy] = evaluate_hierarchy(obs.loc[hierarchy], times)
    for i in np.flatnonzero(~hierarchy):
        p = obs.iloc[i]
        kind = p.control_kind
        targeted = np.arange(len(CELLTYPES)) == int(p.true_celltype_index)
        if kind in ("stable_celltype_specific", "low_signal"):
            gate, dgate = np.ones(len(times)), np.zeros(len(times))
        elif kind in ("uniform_opening", "lineage_opening"):
            gate, dgate = normalized_gate(
                times, p.true_control_center_hpf, p.true_control_width_hpf
            )
        elif kind == "transient_lineage":
            phase = np.pi * (times - 10.0) / 14.0
            gate = np.sin(phase) ** 2
            dgate = np.pi / 14.0 * np.sin(2.0 * phase)
        else:
            raise ValueError(f"Unknown control kind: {kind}")
        loading = np.ones(len(CELLTYPES)) if kind == "uniform_opening" else targeted
        state[i] = p.true_control_baseline + p.true_control_amplitude * loading[:, None] * gate
        rate[i] = p.true_control_amplitude * loading[:, None] * dgate
    if not np.all(np.isfinite(state)) or not np.all((state > 0) & (state < 1)):
        raise ValueError("Synthetic accessibility must remain strictly within (0, 1)")
    return state, rate


def make_specialization_dataset(
    seed=42,
    n_constitutive=200,
    n_per_family_trunk=200,
    n_per_celltype_branch=80,
    n_per_control_family=40,
    n_replicates=6,
    noise_precision=1.5,
    root_accessibility=0.65,
    include_cis_graph=True,
):
    """Build a reproducible 3,400 peaks x 120 conditions benchmark by default.

    Counts/depth describe simulated pseudobulk sampling precision, not a sampled
    cells-by-peaks matrix. Replicate noise streams are independent, and the mean
    of all replicate layers is X. Seed 42/43/44 may be used for train/validation/
    held-out datasets; no embedding, fitted settings, or data are shared by them.
    """
    if n_replicates < 3:
        raise ValueError("At least three independent measurement replicates are required")
    config = ContinuousConfig(
        seed=seed,
        n_constitutive=n_constitutive,
        n_per_family_trunk=n_per_family_trunk,
        n_per_celltype_branch=n_per_celltype_branch,
        n_replicates=n_replicates,
        noise_precision=noise_precision,
        root_accessibility=root_accessibility,
    )
    hierarchy_obs = hierarchy_parameters(config)
    hierarchy_obs["truth_model"] = "continuous_hierarchy"
    hierarchy_obs["is_hierarchy"] = True
    hierarchy_obs["control_kind"] = "none"
    hierarchy_obs["truth_module"] = [
        "constitutive_core" if p.true_program == "constitutive_core"
        else f"Family{int(p.true_family) + 1}_bridge" if p.true_program == "family_bridge"
        else f"{p.primary_celltype}_branch"
        for _, p in hierarchy_obs.iterrows()
    ]
    # The primary-celltype placeholder in the old generator is not a target for
    # constitutive peaks. Make that distinction explicit in the new benchmark.
    hierarchy_obs.loc[hierarchy_obs.true_program == "constitutive_core", "primary_celltype"] = "all_celltypes"
    controls = control_parameters(seed, n_per_control_family, len(hierarchy_obs))
    obs = pd.concat([hierarchy_obs, controls], axis=0, sort=False)
    state, instantaneous_rate = evaluate_specialization(obs, STAGES_HPF)
    truth = state.reshape(len(obs), -1)
    var = pd.DataFrame(
        [(c, t) for c in CELLTYPES for t in STAGES_HPF],
        columns=["celltype", "time_hpf"],
    )
    var.index = [f"{c}_{t:g}hpf" for c, t in var.to_numpy()]
    var["family"] = [f"Family{CELLTYPES.index(c) // 5 + 1}" for c in var.celltype]
    observations, counts_all = [], []
    for r in range(n_replicates):
        # Separate metadata and beta draws; adding a new metadata field cannot
        # consume or correlate the replicate's accessibility noise stream.
        meta_rng = np.random.default_rng(np.random.SeedSequence([seed, 102, r]))
        noise_rng = np.random.default_rng(np.random.SeedSequence([seed, 103, r]))
        counts = np.clip(meta_rng.lognormal(np.log(65), 0.65, 120), 20, 400).astype(int)
        depth = meta_rng.lognormal(-0.5 * 0.3 ** 2, 0.3, 120)
        precision = counts * depth * noise_precision
        observations.append(noise_rng.beta(truth * precision, (1 - truth) * precision).astype("float32"))
        counts_all.append(counts)
        var[f"n_cells_rep{r + 1}"] = counts
        var[f"depth_rep{r + 1}"] = depth
    var["n_cells"] = np.sum(counts_all, axis=0)
    a = ad.AnnData(np.mean(observations, axis=0).astype("float32"), obs=obs, var=var)
    a.layers["truth_state"] = truth.astype("float32")
    a.layers["truth_instantaneous_rate"] = instantaneous_rate.reshape(len(obs), -1).astype("float32")
    for r, values in enumerate(observations):
        a.layers[f"sampling_rep{r + 1}"] = values
    velocity, valid, dt = temporal_rates(a.layers["truth_state"], var)
    a.layers["truth_velocity"] = velocity.astype("float32")
    a.layers["truth_velocity_valid"] = valid
    a.var["interval_hpf"] = dt
    if include_cis_graph:
        a.obsp["coaccessibility"] = local_cis_graph(truth, obs.chrom.to_numpy(), seed)
    a.uns["synthetic"] = dict(
        **asdict(config),
        n_per_control_family=n_per_control_family,
        scenario="continuous_specialization_with_temporal_controls",
        n_hierarchy=int(obs.is_hierarchy.sum()),
        n_controls=int((~obs.is_hierarchy).sum()),
        celltypes=list(CELLTYPES),
        stages_hpf=STAGES_HPF.tolist(),
        controls=list(CONTROL_KINDS),
        root_meaning="Constitutive broad-accessibility reference, not a temporal ancestor",
        specialization_meaning="Known profile coordinate only for hierarchy peaks; NaN for controls",
        graph_model="Incomplete cis-restricted neighbors from noiseless profiles, if requested; favorable surrogate, not inferred Cicero",
        observation_model="Mean of independent continuous beta samples; condition-specific precision = n_cells * depth * noise_precision",
        time_domain_hpf=[10.0, 24.0],
        interpretation="Each row is one fixed genomic peak; controls need not follow a root-outward temporal progression",
    )
    return a

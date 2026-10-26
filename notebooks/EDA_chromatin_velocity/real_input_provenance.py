"""Read-only provenance and bounded-memory audit of the real peak atlas.

Only the requested report directory is written. Coordinates and ordered peak IDs
are compared in full; large accessibility layers are checked in deterministic
contiguous sample blocks. Sparse graphs are described and compared by sampled
rows without loading their full matrices. No source H5AD is modified.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT.parent / "data"
DEFAULT_OUTPUT = Path(__file__).parent / "outputs" / "real_specialization_v1"
STAGE_TO_HPF = {"0somites": 10, "5somites": 12, "10somites": 14,
                "15somites": 16, "20somites": 19, "30somites": 24}
STAGES = list(STAGE_TO_HPF.values())
PATHS = {
    "celltype_state": BASE / "annotated_data/objects_v2/peaks_by_ct_tp_master.h5ad",
    "legacy_celltype_alias": BASE / "annotated_data/objects_v2/peaks_by_ct_tp_master_anno.h5ad",
    "atlas_master": BASE / "annotated_data/objects_v2/peaks_by_pb_annotated_master.h5ad",
    "raw_cells": BASE / "processed_data/01_Signac_processed/integrated_RNA_ATAC_counts_peaks_integrated_raw_counts_master_filtered.h5ad",
    "coaccessibility": BASE / "processed_data/13_peak_umap_analysis/cicero_integrated_ATAC_v2/02_integrated_ATAC_v2_cicero_connections_peaks_integrated_peaks.csv",
}


def plain(value):
    if isinstance(value, bytes):
        return value.decode()
    if isinstance(value, np.ndarray):
        return [plain(v) for v in value.tolist()]
    if isinstance(value, np.generic):
        return plain(value.item())
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def read_column(group, name):
    value = group[name]
    if isinstance(value, h5py.Dataset):
        return value.asstr()[:] if value.dtype.kind in "OSU" else value[:]
    cats = value["categories"]
    categories = cats.asstr()[:] if cats.dtype.kind in "OSU" else cats[:]
    codes = value["codes"][:]
    result = categories[np.maximum(codes, 0)]
    if (codes < 0).any():
        result = result.astype(object)
        result[codes < 0] = None
    return result


def file_info(path):
    info = dict(path=str(path), exists=path.exists())
    if path.exists():
        stat = path.stat()
        info.update(size_bytes=stat.st_size,
                    modified_utc=datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat())
    return info


def array_info(node):
    info = {"attrs": plain(dict(node.attrs))}
    if isinstance(node, h5py.Dataset):
        info.update(shape=list(node.shape), dtype=str(node.dtype),
                    chunks=list(node.chunks) if node.chunks else None,
                    compression=node.compression)
    else:
        info.update(shape=plain(node.attrs.get("shape")),
                    nnz=int(len(node["data"])) if "data" in node else None,
                    dtype=str(node["data"].dtype) if "data" in node else None)
    return info


def structure(f):
    return dict(
        X=array_info(f["X"]),
        layers={k: array_info(v) for k, v in f["layers"].items()},
        obs_columns=list(f["obs"]), var_columns=list(f["var"]),
        obsm={k: array_info(v) for k, v in f["obsm"].items()},
        obsp={k: array_info(v) for k, v in f["obsp"].items()},
        uns_keys=list(f["uns"]),
    )


def small_uns(f, key):
    node = f["uns"].get(key)
    if node is None:
        return None
    if isinstance(node, h5py.Dataset):
        return plain(node[()])
    result = {}
    for name, child in node.items():
        if isinstance(child, h5py.Dataset) and child.shape == ():
            result[name] = plain(child[()])
        elif name == "params":
            result[name] = {k: plain(v[()]) for k, v in child.items()
                            if isinstance(v, h5py.Dataset) and v.shape == ()}
    return result


def compare_coordinates(left, right, keys, block=32768):
    report = {}
    for key in keys:
        a, b = left["obsm"][key], right["obsm"][key]
        equal, maxdiff, finite = a.shape == b.shape, 0.0, True
        if equal:
            for start in range(0, len(a), block):
                aa, bb = a[start:start + block], b[start:start + block]
                equal &= np.array_equal(aa, bb)
                finite &= np.isfinite(aa).all() and np.isfinite(bb).all()
                maxdiff = max(maxdiff, float(np.max(np.abs(aa - bb))))
        report[key] = dict(shape=list(a.shape), exact_equal=bool(equal),
                           max_abs_difference=maxdiff, finite=bool(finite), scope="all rows")
    return report


def sampled_graph_comparison(left, right, n_rows=256):
    report = {}
    for key in ["connectivities", "distances"]:
        a, b = left["obsp"][key], right["obsp"][key]
        pointers_a, pointers_b = a["indptr"][:], b["indptr"][:]
        rows = np.linspace(0, len(pointers_a) - 2, n_rows, dtype=int)
        jac, exact = [], 0
        for i in rows:
            ia = a["indices"][pointers_a[i]:pointers_a[i + 1]]
            ib = b["indices"][pointers_b[i]:pointers_b[i + 1]]
            va = a["data"][pointers_a[i]:pointers_a[i + 1]]
            vb = b["data"][pointers_b[i]:pointers_b[i + 1]]
            sa, sb = set(ia.tolist()), set(ib.tolist())
            jac.append(len(sa & sb) / len(sa | sb) if sa | sb else 1.0)
            exact += int(np.array_equal(ia, ib) and np.array_equal(va, vb))
        report[key] = dict(ct_nnz=int(len(a["data"])), master_nnz=int(len(b["data"])),
                           sampled_rows=len(rows), exact_matching_sample_rows=exact,
                           mean_sample_neighbor_jaccard=float(np.mean(jac)),
                           conclusion="Use the master graph with the published master coordinates")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = dict(audited_utc=datetime.now(timezone.utc).isoformat(),
                  files={k: file_info(v) for k, v in PATHS.items()},
                  audit_scope="Read-only source access; full peak-ID and coordinate checks; bounded samples of large state layers and sparse graph rows")
    with h5py.File(PATHS["celltype_state"], "r") as ct, \
            h5py.File(PATHS["atlas_master"], "r") as master, \
            h5py.File(PATHS["legacy_celltype_alias"], "r") as alias:
        for label, f in [("celltype_state", ct), ("atlas_master", master), ("legacy_celltype_alias", alias)]:
            report["files"][label]["h5_structure"] = structure(f)
            report["files"][label]["normalization_and_graph_metadata"] = {
                k: small_uns(f, k) for k in ["common_scale_factor", "neighbors", "pca", "umap"]}
        index_key = ct["obs"].attrs["_index"]
        ids = read_column(ct["obs"], index_key)
        master_ids = read_column(master["obs"], master["obs"].attrs["_index"])
        alias_ids = read_column(alias["obs"], alias["obs"].attrs["_index"])
        report["peak_alignment"] = dict(n_peaks=len(ids), unique=bool(pd.Index(ids).is_unique),
            ct_master_ordered_ids_exact_equal=bool(np.array_equal(ids, master_ids)),
            ct_alias_ordered_ids_exact_equal=bool(np.array_equal(ids, alias_ids)),
            first_ids=ids[:3].tolist(), last_ids=ids[-3:].tolist())
        if not report["peak_alignment"]["ct_master_ordered_ids_exact_equal"]:
            raise ValueError("Master and celltype object peak order differs; explicit alignment required")
        del master_ids, alias_ids
        keys = ["X_umap", "X_umap_2D", "X_umap_3D"]
        report["ct_vs_master_coordinates"] = compare_coordinates(ct, master, keys)
        report["ct_vs_alias_coordinates"] = compare_coordinates(ct, alias, keys)
        sample_starts = np.linspace(0, len(ids) - 256, 16, dtype=int)
        pca_equal = all(np.array_equal(ct["obsm/X_pca"][i:i + 256], master["obsm/X_pca"][i:i + 256])
                        for i in sample_starts)
        report["pca_comparison"] = dict(ct_master_sample_equal=bool(pca_equal), sampled_rows=4096,
            interpretation="Copied from Leiden-pseudobulk atlas; not reconstructed from the CT-by-stage matrix")
        report["ct_vs_master_graphs"] = sampled_graph_comparison(ct, master)
        var_key = ct["var"].attrs["_index"]
        columns = ["annotation_ML_coarse", "dev_stage", "n_cells", "total_coverage", "scale_factor", "mean_depth"]
        conditions = pd.DataFrame({k: read_column(ct["var"], k) for k in columns},
                                  index=read_column(ct["var"], var_key))
        conditions.index.name = "condition"
        conditions["celltype"] = conditions.annotation_ML_coarse
        conditions["time_hpf"] = conditions.dev_stage.map(STAGE_TO_HPF)
        conditions["usable_min20"] = conditions.n_cells >= 20
        conditions["usable_min50"] = conditions.n_cells >= 50
        celltypes = sorted(conditions.celltype.unique())
        pairs = set(zip(conditions.celltype, conditions.time_hpf))
        support = {}
        for cutoff in [0, 20, 50]:
            mask = np.zeros((len(celltypes), 6), dtype=bool)
            for _, row in conditions.iterrows():
                mask[celltypes.index(row.celltype), STAGES.index(row.time_hpf)] = row.n_cells >= cutoff
            support[str(cutoff)] = dict(stage_celltypes=mask.sum(0).tolist(),
                paired_interval_celltypes=(mask[:, :-1] & mask[:, 1:]).sum(0).tolist(),
                complete_celltypes=[c for c, valid in zip(celltypes, mask) if valid.all()],
                excluded_all_stages=[c for c, valid in zip(celltypes, mask) if not valid.any()])
        report["conditions"] = dict(n_columns=len(conditions), n_celltypes=len(celltypes),
            celltypes=celltypes, stages_hpf=STAGES, stage_to_hpf=STAGE_TO_HPF,
            unique_names=bool(conditions.index.is_unique), unique_celltype_time_pairs=len(pairs) == len(conditions),
            missing_combinations=[dict(celltype=c, time_hpf=t) for c in celltypes for t in STAGES if (c, t) not in pairs],
            low_n_conditions=conditions[conditions.n_cells < 20].reset_index().to_dict("records"),
            support_by_min_cells=support,
            parser="Use var.annotation_ML_coarse and var.dev_stage directly. If parsing names, rsplit('_', 1); celltype names contain underscores.")
        common = float(ct["uns/common_scale_factor"][()])
        samples, all_finite, all_nonnegative = [], True, True
        x_equals_norm, log_relation, raw_integer, norm_relation = True, True, True, True
        max_raw_scale_error = 0.0
        for start in sample_starts:
            x = ct["X"][start:start + 256]
            norm = ct["layers/normalized"][start:start + 256]
            raw = ct["layers/sum"][start:start + 256]
            log = ct["layers/log_norm"][start:start + 256]
            samples.append(x)
            all_finite &= np.isfinite(x).all()
            all_nonnegative &= (x >= 0).all()
            x_equals_norm &= np.array_equal(x, norm)
            log_relation &= np.array_equal(log, np.log1p(norm))
            raw_integer &= np.array_equal(raw, np.floor(raw))
            scaled = raw * conditions.scale_factor.to_numpy()[None]
            norm_relation &= np.allclose(norm, scaled, rtol=1e-12, atol=1e-12)
            max_raw_scale_error = max(max_raw_scale_error, float(abs(norm - scaled).max()))
        sample_x = np.concatenate(samples)
        report["normalization"] = dict(sampled_rows=len(sample_x), sample_blocks=len(sample_starts), rows_per_block=256,
            x_equals_normalized=bool(x_equals_norm), log_equals_log1p_normalized=bool(log_relation),
            normalized_equals_sum_times_scale_factor=bool(norm_relation), max_raw_scale_error=max_raw_scale_error,
            raw_sum_sample_is_integer=bool(raw_integer), x_sample_finite=bool(all_finite),
            x_sample_nonnegative=bool(all_nonnegative), common_scale_factor=common,
            scale_factor_times_total_coverage_constant=bool(np.allclose(conditions.scale_factor * conditions.total_coverage, common)),
            median_present_group_coverage=float(conditions.total_coverage.median()),
            median_with_two_missing_groups_as_zero_for_scale_only=float(np.median(np.r_[conditions.total_coverage, 0, 0])),
            formula="normalized = raw_pseudobulk_sum * common_scale_factor / group_total_coverage",
            cpm_formula="CPM = normalized * 1e6 / common_scale_factor",
            x_sample_quantiles=dict(zip(["min", "p10", "p50", "p90", "p99", "max"], np.quantile(sample_x, [0, .1, .5, .9, .99, 1]).tolist())),
            mean_peak_cpm_sample_quantiles=dict(zip(["min", "p01", "p05", "p10", "p50", "p90", "p99", "max"],
                np.quantile(sample_x.mean(1) * 1e6 / common, [0, .01, .05, .1, .5, .9, .99, 1]).tolist())),
            units="Library-depth-normalized counts (or CPM after the stated common rescaling), not accessibility probabilities",
            missing_zero_semantics="The two absent conditions have no column. Zeros in present columns are observed zero summed counts. Do not zero-fill absent/low-cell conditions for specificity or temporal rates.")
        report["atlas_annotations"] = {}
        for name in ["leiden_coarse", "leiden_unified", "celltype", "timepoint", "lineage"]:
            values = read_column(master["obs"], name)
            unique = pd.unique(values)
            report["atlas_annotations"][name] = dict(n_unique=len(unique), labels=plain(unique) if len(unique) <= 40 else None)
        del sample_x, samples
    print("State and atlas audit complete; checking per-cell metadata.", flush=True)
    with h5py.File(PATHS["raw_cells"], "r") as raw:
        report["files"]["raw_cells"]["h5_structure"] = structure(raw)
        meta = pd.DataFrame({name: read_column(raw["obs"], name)
                             for name in ["dataset", "dev_stage", "orig.ident", "annotation_ML_coarse"]})
        library_counts = meta.groupby(["dataset", "dev_stage"], observed=True).size().reset_index(name="n_cells")
        pooled_counts = meta.groupby(["annotation_ML_coarse", "dev_stage"], observed=True).size()
        mismatches = []
        for condition, row in conditions.iterrows():
            count = int(pooled_counts.get((row.celltype, row.dev_stage), 0))
            if count != int(row.n_cells):
                mismatches.append(dict(condition=condition, pseudobulk_n_cells=int(row.n_cells), raw_n_cells=count))
        raw_peaks = pd.Index(read_column(raw["var"], raw["var"].attrs["_index"]))
        report["raw_peak_alignment"] = dict(raw_n_peaks=len(raw_peaks), atlas_n_peaks=len(ids),
            atlas_peaks_missing_from_raw=int((raw_peaks.get_indexer(ids) < 0).sum()),
            raw_peaks_excluded_from_atlas=raw_peaks.difference(pd.Index(ids)).tolist())
        report["replicate_evidence"] = dict(n_cells=len(meta), library_stage_counts=library_counts.to_dict("records"),
            libraries_per_stage=meta.groupby("dev_stage").dataset.nunique().to_dict(),
            orig_ident_counts=meta["orig.ident"].value_counts().to_dict(),
            pooled_cell_count_mismatches=mismatches,
            pseudobulk_replicate_layers=[],
            interpretation="Seven dataset/library labels: two at 16 hpf and one at each other stage. Metadata alone does not establish biological independence. The 190-column object pools these labels and has no replicate-specific layers.",
            implication="Cell resampling can estimate within-library sampling sensitivity. It cannot reproduce six independent biological replicates or separate developmental time from stage-specific library effects.")
    prior_path = Path(__file__).parent / "outputs/real_input_audit/audit.json"
    if prior_path.exists():
        prior = json.loads(prior_path.read_text())
        report["prior_full_scan"] = dict(source=file_info(prior_path), scope="Existing prior audit; full state and Cicero scans not rerun here",
            X_nonfinite_entries=prior["X_nonfinite_entries"], X_zero_entries=prior["X_zero_entries"],
            coaccessibility=prior["coaccessibility"],
            coaccessibility_fraction_peaks_connected_above_025=prior["coaccessibility"]["peaks_connected_above_threshold"]["0.25"] / len(ids))
    report["recommended_primary"] = dict(
        state_path=str(PATHS["celltype_state"]), state_layer="normalized", convert_to_cpm=True,
        atlas_path=str(PATHS["atlas_master"]), graph_key="connectivities", pca_key="X_pca",
        embedding_keys=["X_umap_2D", "X_umap_3D"],
        primary_celltypes=support["20"]["complete_celltypes"], n_primary_celltypes=len(support["20"]["complete_celltypes"]),
        retain_all_190_condition_columns=True,
        support_policy="Primary specificity uses the fixed 21 celltypes with >=20 cells at all six stages; report paired-support and >=50-cell alternatives separately. Mask unsupported condition-level rates.",
        uncertainty_policy="Do not reuse simulated replicate SEs or call sensitivity filters biological confidence intervals.",
        coaccessibility_policy="Unused in the primary field; any later cis ablation must retain isolates and disclose extremely limited thresholded coverage.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    conditions.to_csv(args.output_dir / "input_conditions.csv")
    path = args.output_dir / "input_provenance.json"
    path.write_text(json.dumps(plain(report), indent=2, allow_nan=False) + "\n")
    print(json.dumps(plain({k: report[k] for k in ["peak_alignment", "ct_vs_master_graphs", "recommended_primary", "replicate_evidence"]}), indent=2))
    print(f"Saved {path.resolve()}", flush=True)


if __name__ == "__main__":
    main()

"""Offline display of real peak-accessibility fields on an existing atlas.

The viewer never fits a model, computes neighbors, or changes UMAP coordinates.
``select_display_peaks`` only selects rows for browser display; all inference
must precede display sampling. Large numeric arrays are embedded as float32 or
uint8 buffers, retaining NaN masks without duplicating wide JSON matrices.

Public APIs: ``select_display_peaks(...) -> (indices, report)`` and
``export_real_viewer(payload, output_path) -> Path``.
"""
from __future__ import annotations

import base64
import json
from pathlib import Path

import numpy as np
from plotly.offline import get_plotlyjs


def select_display_peaks(umap2d, umap3d=None, strata=None, include=None,
                         max_points=18_000, seed=42):
    """Select deterministic display rows with spatial coverage and random fill.

    Forced marker rows take precedence. Spatial bins and annotation strata keep
    rare regions visible, so the resulting display is not an abundance estimate.
    Sorted returned indices can be used directly for HDF5 row reads.
    """
    xy = np.asarray(umap2d)
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError("umap2d must be N x 2")
    n = len(xy)
    xyz = None if umap3d is None else np.asarray(umap3d)
    if xyz is not None and xyz.shape != (n, 3):
        raise ValueError("umap3d must be N x 3")
    if int(max_points) < 1:
        raise ValueError("max_points must be positive")
    valid = np.isfinite(xy).all(axis=1)
    if xyz is not None:
        valid &= np.isfinite(xyz).all(axis=1)
    pool = np.flatnonzero(valid)
    budget = min(int(max_points), len(pool))
    forced = np.unique(np.asarray([] if include is None else include, dtype=np.int64))
    if ((forced < 0) | (forced >= n)).any():
        raise ValueError("include contains an out-of-range global row index")
    forced = forced[valid[forced]]
    if len(forced) > budget:
        raise ValueError("Forced rows exceed the display budget")
    rng = np.random.default_rng(seed)
    priority = rng.random(n)
    selected = np.zeros(n, dtype=bool)
    selected[forced] = True
    allocations = {"forced": int(len(forced))}

    def add_representatives(codes, allowance, name):
        available = pool[~selected[pool]]
        if not len(available) or allowance <= 0:
            allocations[name] = 0
            return
        order = np.lexsort((priority[available], codes[available]))
        ordered = available[order]
        take = np.r_[True, codes[ordered][1:] != codes[ordered][:-1]]
        reps = ordered[take]
        remaining = budget - int(selected.sum())
        limit = min(allowance, remaining)
        if len(reps) > limit:
            reps = reps[np.argsort(priority[reps])[:limit]]
        selected[reps] = True
        allocations[name] = int(len(reps))

    def bin_codes(coords, bins):
        points = coords[pool]
        lo, hi = points.min(axis=0), points.max(axis=0)
        scaled = (coords - lo) / np.maximum(hi - lo, 1e-12)
        scaled[~valid] = 0
        b = np.clip(np.floor(scaled * bins), 0, bins - 1).astype(np.int64)
        return sum(b[:, d] * bins ** d for d in range(coords.shape[1]))

    if len(pool):
        add_representatives(bin_codes(xy, 64), max(1, budget // 4), "umap2d_bins")
        if xyz is not None:
            add_representatives(bin_codes(xyz, 32), max(1, budget * 3 // 10), "umap3d_bins")
        if strata is not None:
            labels = np.asarray(strata).astype(str)
            if labels.shape != (n,):
                raise ValueError("strata must contain one label per original peak")
            _, codes = np.unique(labels, return_inverse=True)
            add_representatives(codes, max(1, budget // 10), "annotation_strata")
        available = pool[~selected[pool]]
        remaining = budget - int(selected.sum())
        fill = available[np.argsort(priority[available])[:remaining]]
        selected[fill] = True
        allocations["uniform_random_fill"] = int(len(fill))
    result = np.flatnonzero(selected)
    return result, {
        "total_peaks": n, "finite_embedding_peaks": int(valid.sum()),
        "display_peaks": len(result), "max_points": int(max_points),
        "seed": int(seed), "allocations": allocations,
        "policy": "Forced markers, spatial-bin and annotation coverage, then uniform random fill; display selection only, not abundance preserving.",
    }


_ARRAY_KEYS = {
    "umap2d", "umap3d", "accessibility", "opening_rate", "condition_n_cells",
    "condition_valid", "specificity", "specificity_rate", "vectors2d", "vectors3d",
    "support", "coherence", "arrow_mask", "stability", "support2d", "support3d",
    "coherence2d", "coherence3d", "arrow_mask2d", "arrow_mask3d", "stability2d", "stability3d",
}


def _encode(value, key=None):
    if key in _ARRAY_KEYS and isinstance(value, (list, tuple, np.ndarray)) and np.asarray(value).ndim > 0:
        array = np.asarray(value)
        dtype = "u1" if array.dtype.kind == "b" else "f4"
        array = np.ascontiguousarray(array, dtype=np.dtype("<" + dtype))
        return {"__array__": dtype, "shape": list(array.shape),
                "data": base64.b64encode(array.tobytes()).decode("ascii")}
    if isinstance(value, dict):
        return {str(k): _encode(v, str(k)) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_encode(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def _validate(payload):
    n, c, t = len(payload["peaks"]), len(payload["celltypes"]), len(payload["stages"])
    if n < 1 or c < 1 or t < 2 or np.any(np.diff(payload["stages"]) <= 0):
        raise ValueError("Need peaks, celltypes and strictly ordered stages")
    for key, shape in [("umap2d", (n, 2)), ("accessibility", (n, c * t))]:
        a = np.asarray(payload[key], dtype=float)
        if a.shape != shape or np.isinf(a).any():
            raise ValueError(f"{key} must have shape {shape} without infinities")
    if not np.isfinite(payload["umap2d"]).all():
        raise ValueError("The display subset must have finite UMAP coordinates")
    dims = [2]
    if payload.get("umap3d") is not None:
        if np.asarray(payload["umap3d"]).shape != (n, 3) or not np.isfinite(payload["umap3d"]).all():
            raise ValueError("umap3d must be finite N x 3")
        dims.append(3)
    for key in ["condition_n_cells", "condition_valid"]:
        if key in payload and np.asarray(payload[key]).shape != (c, t):
            raise ValueError(f"{key} must have shape {(c, t)}")
    if "opening_rate" in payload and np.asarray(payload["opening_rate"]).shape not in ((n, c * (t - 1)), (n, c * t)):
        raise ValueError("opening_rate must have celltype-major stage or interval columns")
    if not payload.get("methods"):
        raise ValueError("At least one specificity method is required")
    for name, method in payload["methods"].items():
        for key, shape in [("specificity", (n, t)), ("specificity_rate", (n, t - 1))]:
            if np.asarray(method[key]).shape != shape:
                raise ValueError(f"{name}.{key} must have shape {shape}")
        if not method.get("fields"):
            raise ValueError(f"{name} has no arrow fields")
        if len({f["id"] for f in method["fields"]}) != len(method["fields"]):
            raise ValueError(f"{name} has duplicate field IDs")
        for field in method["fields"]:
            for dim in dims:
                if np.asarray(field[f"vectors{dim}d"]).shape not in ((n, dim), (n, t - 1, dim)):
                    raise ValueError(f"Invalid {name}.{field['id']}.vectors{dim}d shape")
            for key in _ARRAY_KEYS - {"vectors2d", "vectors3d"}:
                if key in field and np.asarray(field[key]).shape not in ((n,), (n, t - 1)):
                    raise ValueError(f"Invalid {name}.{field['id']}.{key} shape")
        for key, limit in method.get("color_limits", {}).items():
            if not np.isfinite(limit) or limit <= 0:
                raise ValueError(f"{name}.color_limits.{key} must be positive and finite")
    for key, limit in payload.get("color_limits", {}).items():
        if not np.isfinite(limit) or limit <= 0:
            raise ValueError(f"color_limits.{key} must be positive and finite")


def export_real_viewer(payload: dict, output_path: Path) -> Path:
    """Write an offline HTML for an aligned display subset of real atlas peaks."""
    _validate(payload)
    encoded = json.dumps(_encode(payload), allow_nan=False, separators=(",", ":"))
    encoded = encoded.replace("<", "\\u003c").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
    html = _HTML.replace("__PLOTLY__", get_plotlyjs()).replace("__DATA__", encoded).replace("__APP__", _JS)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    return out


_HTML = r'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Chromatin accessibility on the peak atlas</title><style>
:root{color-scheme:light;--ink:#19324a;--muted:#53677a;--line:#d8e1e9;--blue:#155eaa}*{box-sizing:border-box}body{margin:0;background:#eef3f7;color:var(--ink);font:14px system-ui,sans-serif}main{max-width:1750px;margin:auto;padding:24px}h1{font-size:27px;margin:0 0 10px}h2{font-size:16px;margin:0 0 8px}p{line-height:1.5;margin:8px 0}.muted{color:var(--muted)}.badge{display:inline-block;padding:4px 8px;background:#e2edf7;border-radius:5px;font-size:12px;font-weight:650;margin-right:5px}.controls{display:flex;gap:12px;flex-wrap:wrap;background:white;border:1px solid var(--line);border-radius:9px;padding:15px;margin:16px 0 12px;align-items:end}label{display:flex;flex-direction:column;gap:5px;font-size:12px;font-weight:600}select,input,button{font:inherit}select,input[type=search]{max-width:370px;padding:7px;border:1px solid #bccad7;border-radius:4px;background:white;color:var(--ink)}input[type=range]{width:115px;accent-color:var(--blue)}.field-control{flex:1;min-width:230px}.field-control select{max-width:100%;width:100%}.check{display:flex;flex-direction:row;align-items:center;gap:5px;font-weight:400}.toolbar{display:flex;gap:17px;align-items:center;flex-wrap:wrap;padding:5px 2px 12px}.toolbar label{min-width:125px}.toolbar .check{min-width:0}.display-note{font-size:12px;max-width:400px;color:var(--muted);margin-left:auto}.grid{display:grid;grid-template-columns:1.5fr 1fr;gap:15px}.panel{background:white;border:1px solid var(--line);border-radius:9px;padding:12px;min-width:0}#atlas{height:690px}#curves{height:320px}#specificity-curve{height:280px}#field-note{background:#e7eff7;border-left:3px solid #3979ad;padding:10px 13px;font-size:13px;line-height:1.45}#method-note{font-size:12px;color:var(--muted)}#color-note,#selected-note,#render-status,#search-status{font-size:12px;color:var(--muted)}#selected-note{padding:0 10px;min-height:35px}button{border:1px solid #b9c8d6;background:white;border-radius:4px;padding:6px 10px;color:var(--ink);cursor:pointer}button:hover{background:#edf4fb}#error{display:none;color:#8b1b1b;background:#fee9e9;padding:12px;white-space:pre-wrap}.search-row{display:flex;gap:7px;flex-wrap:wrap;align-items:center;margin:9px 0}.search-row input{flex:1;min-width:140px}.search-row select{width:100%;max-width:100%}#legend-items{display:flex;gap:5px 12px;flex-wrap:wrap;max-height:125px;overflow:auto;font-size:11px;padding-top:7px}.legend-dot{display:inline-block;width:9px;height:9px;margin-right:4px;border-radius:50%}details{font-size:12px;color:var(--muted);margin-top:8px}footer{font-size:12px;color:var(--muted);padding:13px 0}#interpretation-notice{background:#fff3df;border:1px solid #e8bf7d;border-left:4px solid #b7771f;border-radius:5px;padding:11px 13px;margin:12px 0;color:#714311;font-size:13px;line-height:1.5}.hidden{display:none!important}@media(max-width:1100px){.grid{grid-template-columns:1fr}#atlas{height:560px}.display-note{margin-left:0}}@media(max-width:650px){main{padding:12px}h1{font-size:23px}#atlas{height:450px}.controls{gap:10px}}
</style></head><body><main>
<h1 id="title">Chromatin accessibility on the peak atlas</h1>
<p><strong>Every dot is one genomic peak.</strong> Coordinates come from the existing peak UMAP; accessibility changes are measured across celltypes and developmental stages.</p>
<p><span class="badge" id="dimensions"></span><span class="badge">Measured pseudobulk accessibility</span></p>
<div id="interpretation-notice" class="hidden" role="note"></div><p id="method-note"></p><div id="field-note"></div><div id="error"></div>
<div class="controls"><label>Specificity measure<select id="method"></select></label><label class="field-control">Arrow interpretation<select id="field"></select></label><label>View<select id="dimension"><option value="2">2D peak UMAP</option><option value="3">3D peak UMAP</option></select></label><label>Color shows<select id="metric"><option value="annotation:celltype">Annotated celltype</option><option value="annotation:leiden_coarse">Coarse peak cluster</option><option value="annotation:lineage">Annotated lineage</option><option value="broad_reference">Broad reference candidates</option><option value="specificity">Specificity at interval start</option><option value="specificity_rate">Specialization rate</option><option value="opening_rate">Opening / closing rate</option><option value="accessibility">Accessibility at interval start</option></select></label><label>Celltype<select id="celltype"></select></label><label>Developmental interval<select id="interval"></select></label></div>
<div class="toolbar"><label>Maximum arrows <span id="density-label"></span><input id="density" type="range" min="20" max="300" step="20" value="140"></label><label>Display length × <span id="length-label"></span><input id="length" type="range" min="0.25" max="2" step="0.05" value="1"></label><label>Minimum coherence <span id="coherence-label"></span><input id="coherence" type="range" min="0" max="1" step="0.05" value="0"></label><label class="check"><input id="show-arrows" type="checkbox" checked> Show arrows</label><button id="reset">Reset view</button><button id="png">Save atlas PNG</button><span class="display-note">Arrows show local directions with normalized display lengths. Their length is not a physical speed. 3D uses a smaller background sample for rotation; all stored profiles remain searchable.</span></div>
<div class="grid"><section class="panel"><div id="atlas"></div><p id="color-note"></p><p id="render-status"></p><details id="legend"><summary>Annotation legend</summary><div id="legend-items"></div></details><p class="muted" style="font-size:12px">Click a displayed peak to inspect its profile. In 3D, drag to rotate and use the toolbar to zoom.</p></section><section class="panel"><h2 id="selected-title">Selected peak</h2><div class="search-row"><input id="search" type="search" placeholder="Gene symbol or peak ID" aria-label="Search shown peaks"><button id="search-button">Find in display</button><select id="search-results" class="hidden" aria-label="Search results"></select></div><p id="search-status">Search covers displayed peaks and their nearest/linked gene annotations.</p><div id="curves"></div><p id="selected-note"></p><label class="check"><input id="all-celltypes" type="checkbox"> Show supported trajectories in other celltypes</label><div id="specificity-curve"></div></section></div>
<details><summary>Data and interpretation</summary><p id="panel-note"></p><p id="notes"></p></details><footer>Broad reference candidates describe accessibility patterns; they are not asserted ancestors. Neighbor-path orientation is an analytical mapping. Temporal changes in a fixed genomic peak do not mean that it becomes another peak.</footer>
</main><script>__PLOTLY__</script><script>let PACKED=__DATA__;</script><script>__APP__</script></body></html>'''

_JS = r'''
'use strict';
function decode(value){
 if(value&&typeof value==='object'&&value.__array__){
   const text=atob(value.data),bytes=new Uint8Array(text.length);for(let i=0;i<text.length;i++)bytes[i]=text.charCodeAt(i);
   const flat=value.__array__==='u1'?bytes:new Float32Array(bytes.buffer),shape=value.shape;
   const stride=shape.map((n,a)=>shape.slice(a+1).reduce((x,y)=>x*y,1));
   function rows(axis,offset){if(axis===shape.length-1)return flat.subarray(offset,offset+shape[axis]);return Array.from({length:shape[axis]},(_,j)=>rows(axis+1,offset+j*stride[axis]));}
   return rows(0,0);
 }
 if(Array.isArray(value))return value.map(decode);
 if(value&&typeof value==='object'){for(const key of Object.keys(value))value[key]=decode(value[key]);}
 return value;
}
const DATA=decode(PACKED);PACKED=null;
const $=id=>document.getElementById(id),finite=Number.isFinite;
const N=DATA.peaks.length,C=DATA.celltypes.length,T=DATA.stages.length;
const DIVERGE=[[0,'#2166ac'],[.5,'#d8dee4'],[1,'#b2182b']],SEQUENTIAL=[[0,'#eff3f7'],[.25,'#b8d9e8'],[.6,'#4c9dbb'],[1,'#174a78']];
const config={responsive:true,displaylogo:false,toImageButtonOptions:{format:'png',scale:2}};
let selected=0,atlasInitialized=false,viewReset=0,renderCounter=0;
const annotations={},samplingCache=new Map();let background3D=null;
function number(x,n=3){return finite(x)?x.toFixed(n):'unavailable';}
function label(value){return value===null||value===undefined||value===''?'unassigned':String(value);}
function peakId(i){return DATA.peaks[i].peak||DATA.peaks[i].peak_id||('peak '+i);}
function peakLabel(i){const p=DATA.peaks[i],gene=p.linked_gene||p.nearest_gene;return peakId(i)+(gene?' · '+gene:'');}
function scalarAt(values,i,t,fallback=0){if(values===undefined||values===null)return fallback;const v=values[i];return Array.isArray(v)||ArrayBuffer.isView(v)?v[t]:v;}
function vectorAt(values,i,t){const v=values[i];return Array.isArray(v[0])||ArrayBuffer.isView(v[0])?v[t]:v;}
function option(select,value,text){const o=document.createElement('option');o.value=value;o.textContent=text;select.appendChild(o);}
function state(){const method=DATA.methods[$('method').value];return {method,methodId:$('method').value,field:method.fields.find(f=>f.id===$('field').value),dim:Number($('dimension').value),ct:Number($('celltype').value),t:Number($('interval').value),metric:$('metric').value,density:Number($('density').value),length:Number($('length').value),coherence:Number($('coherence').value)};}
function validCondition(c,t){return !DATA.condition_valid||Boolean(DATA.condition_valid[c][t]);}
function openingRate(i,c,t){if(!validCondition(c,t)||!validCondition(c,t+1))return NaN;if(DATA.opening_rate){const row=DATA.opening_rate[i];return row[c*(row.length===C*T?T:T-1)+t];}const row=DATA.accessibility[i],a=row[c*T+t],b=row[c*T+t+1];return finite(a)&&finite(b)?(b-a)/(DATA.stages[t+1]-DATA.stages[t]):NaN;}
function metricValues(s){
 if(s.metric.startsWith('annotation:'))return DATA.peaks.map(p=>label(p[s.metric.slice(11)]));
 if(s.metric==='broad_reference')return DATA.peaks.map(p=>p.broad_reference?1:0);
 if(s.metric==='opening_rate')return DATA.peaks.map((p,i)=>openingRate(i,s.ct,s.t));
 if(s.metric==='accessibility')return DATA.accessibility.map(row=>validCondition(s.ct,s.t)?row[s.ct*T+s.t]:NaN);
 if(s.metric==='specificity')return s.method.specificity.map(row=>row[s.field.id==='structural'?T-1:s.t]);
 return s.method[s.metric].map(row=>row[s.t]);
}
function percentile(values,q=.97){const a=Array.from(values).filter(finite).map(Math.abs).sort((a,b)=>a-b);return a.length?Math.max(1e-8,a[Math.floor(q*(a.length-1))]):1;}
function metricLimit(s,values){const fixed=s.method.color_limits?.[s.metric]??DATA.color_limits?.[s.metric];return finite(fixed)&&fixed>0?fixed:(s.metric==='specificity'?1:percentile(values));}
function annotationColor(key,value){return annotations[key]?.colors[value]||'#9ca8b4';}
function colorSpec(s,values){
 if(s.metric.startsWith('annotation:')){const key=s.metric.slice(11);return {color:values.map(v=>annotationColor(key,v)),showscale:false};}
 if(s.metric==='broad_reference')return {color:values.map(v=>v?'#d5a32d':'#b5c0cb'),showscale:false};
 const rate=s.metric.endsWith('_rate'),limit=metricLimit(s,values);
 return {color:values,colorscale:rate?DIVERGE:SEQUENTIAL,cmin:rate?-limit:0,cmax:limit,showscale:true,colorbar:{title:{text:s.metric==='opening_rate'?'CPM / hpf':s.metric==='accessibility'?'CPM':rate?'score / hpf':'specificity'},thickness:13,len:.62,x:1.01}};
}
function colorNote(s,values){
 const a=DATA.stages[s.t],b=DATA.stages[s.t+1];
 if(s.metric.startsWith('annotation:'))return 'Colors use the supplied master-atlas annotations. Hover for labels. Display sampling is designed for spatial coverage, so point counts do not estimate peak abundance.';
 if(s.metric==='broad_reference')return 'Gold marks supplied broad-accessibility reference candidates. This is a descriptive category, not a forced graph root or a claim of developmental ancestry.';
 let text=s.metric==='specificity_rate'?`${s.method.label||s.methodId} specificity change, ${a}–${b} hpf: red = increasing specificity, blue = decreasing specificity. Specialization can arise through opening, closing, or both.`:s.metric==='opening_rate'?`${DATA.celltypes[s.ct]}, ${a}–${b} hpf: red = opening, blue = closing. Both endpoint conditions must pass the supplied cell-count filter.`:s.metric==='specificity'?(s.field.id==='structural'?`${DATA.stages[T-1]} hpf reference specificity (${s.method.label||s.methodId}), matching the structural field's terminal-stage orientation. The reference colors stay fixed when the selected interval changes.`:`${s.method.label||s.methodId} celltype specificity at interval start (${a} hpf), using the same specified celltype panel at every stage.`):`Measured accessibility in ${DATA.celltypes[s.ct]} at ${a} hpf (${DATA.accessibility_units||'CPM'}).`;
 const fixed=s.method.color_limits?.[s.metric]??DATA.color_limits?.[s.metric];
 if(s.metric.endsWith('_rate')&&finite(fixed))text+=` Colors share a fixed ±${fixed.toPrecision(3)} scale across intervals; values beyond the limit saturate.`;
 return text+' Gray points denote unsupported or unavailable values.';
}
function spatialSample(indices,coords,count,weights){
 if(indices.length<=count)return indices;
 let next=0;for(let j=1;j<indices.length;j++)if(weights[indices[j]]>weights[indices[next]])next=j;
 const distances=new Float64Array(indices.length);distances.fill(Infinity);const used=new Uint8Array(indices.length),chosen=[];
 for(let k=0;k<count;k++){chosen.push(indices[next]);used[next]=1;const p=coords[indices[next]];let best=-1,bestDistance=-1;for(let j=0;j<indices.length;j++){if(used[j])continue;let distance=0;for(let d=0;d<p.length;d++){const delta=coords[indices[j]][d]-p[d];distance+=delta*delta;}if(distance<distances[j])distances[j]=distance;if(distances[j]>bestDistance){bestDistance=distances[j];best=j;}}if(best<0)break;next=best;}
 return chosen;
}
function arrowEligibility(s,coords){
 const key=[s.methodId,s.field.id,s.dim,s.t,s.density,s.coherence].join(':');if(samplingCache.has(key))return samplingCache.get(key);
 const vectors=s.field[`vectors${s.dim}d`],indices=[],weights=new Float64Array(N);
 for(let i=0;i<N;i++){const v=vectorAt(vectors,i,s.t),support=scalarAt(s.field[`support${s.dim}d`]??s.field.support,i,s.t,1),coherence=scalarAt(s.field[`coherence${s.dim}d`]??s.field.coherence,i,s.t,1),mask=scalarAt(s.field[`arrow_mask${s.dim}d`]??s.field.arrow_mask,i,s.t,true);if(!mask||!finite(support)||support<=0||!v.every(finite)||v.every(x=>Math.abs(x)<1e-12)||!finite(coherence)||coherence<s.coherence)continue;indices.push(i);weights[i]=support*Math.max(0,coherence);}
 const value={eligible:indices.length,indices:spatialSample(indices,coords,s.density,weights)};samplingCache.set(key,value);if(samplingCache.size>120)samplingCache.delete(samplingCache.keys().next().value);return value;
}
function drawingIndices(s,arrows){
 if(s.dim!==3)return Array.from({length:N},(_,i)=>i);
 if(background3D===null){
   const limit=Math.max(100,Math.min(N,DATA.max_3d_points||6000));
   if(N<=limit)background3D=Array.from({length:N},(_,i)=>i);
   else{
     const coords=DATA.umap3d,lo=[Infinity,Infinity,Infinity],hi=[-Infinity,-Infinity,-Infinity];
     for(const p of coords)for(let d=0;d<3;d++){lo[d]=Math.min(lo[d],p[d]);hi[d]=Math.max(hi[d],p[d]);}
     const priority=i=>Math.imul((DATA.peaks[i].global_index??i)+1,2654435761)>>>0;
     const bins=new Map();for(let i=0;i<N;i++){let key=0;for(let d=0;d<3;d++){const b=Math.max(0,Math.min(23,Math.floor((coords[i][d]-lo[d])/Math.max(hi[d]-lo[d],1e-9)*24)));key+=b*24**d;}const old=bins.get(key);if(old===undefined||priority(i)<priority(old))bins.set(key,i);}
     const reps=[...bins.values()].sort((a,b)=>priority(a)-priority(b)).slice(0,Math.floor(limit*.65));const chosen=new Set(reps);const rest=Array.from({length:N},(_,i)=>i).filter(i=>!chosen.has(i)).sort((a,b)=>priority(a)-priority(b));for(const i of rest){if(chosen.size>=limit)break;chosen.add(i);}background3D=[...chosen];
   }
 }
 const chosen=new Set(background3D);chosen.add(selected);for(const i of arrows)chosen.add(i);for(let i=0;i<N;i++)if(DATA.peaks[i].display_forced)chosen.add(i);for(const i of DATA.forced_peak_indices||[])if(i>=0&&i<N)chosen.add(i);return [...chosen].sort((a,b)=>a-b);
}
function arrowTraces(s,coords,indices){
 const xs=[],ys=[],zs=[],cx=[],cy=[],cz=[],cu=[],cv=[],cw=[];
 for(const i of indices){const p=coords[i],v=Array.from(vectorAt(s.field[`vectors${s.dim}d`],i,s.t),x=>x*s.length),q=Array.from(p,(x,d)=>x+v[d]);if(!Math.hypot(...v))continue;xs.push(p[0],q[0],null);ys.push(p[1],q[1],null);if(s.dim===3){zs.push(p[2],q[2],null);cx.push(q[0]);cy.push(q[1]);cz.push(q[2]);cu.push(v[0]*.24);cv.push(v[1]*.24);cw.push(v[2]*.24);}else{const nx=-v[1],ny=v[0];xs.push(q[0]-.25*v[0]+.13*nx,q[0],q[0]-.25*v[0]-.13*nx,null);ys.push(q[1]-.25*v[1]+.13*ny,q[1],q[1]-.25*v[1]-.13*ny,null);}}
 const line={type:s.dim===3?'scatter3d':'scattergl',mode:'lines',x:xs,y:ys,line:{color:'#233c51',width:s.dim===3?3:1.5},hoverinfo:'skip',showlegend:false,name:'Supplied directions'};
 if(s.dim===2)return [line];line.z=zs;return [line,{type:'cone',x:cx,y:cy,z:cz,u:cu,v:cv,w:cw,anchor:'tip',sizemode:'raw',sizeref:1,colorscale:[[0,'#233c51'],[1,'#233c51']],showscale:false,hoverinfo:'skip',showlegend:false,lighting:{ambient:1,diffuse:0}}];
}
function atlasLayout(s){
 const layout={margin:s.dim===2?{l:58,r:75,t:12,b:48}:{l:15,r:48,t:12,b:25},paper_bgcolor:'white',plot_bgcolor:'white',font:{family:'system-ui, sans-serif',color:'#19324a'},showlegend:false,uirevision:`${s.dim}-${viewReset}`,hovermode:'closest'};
 if(DATA.specificity_estimator){const mode=DATA.specificity_estimator==='depth_matched'?'Exploratory depth-matched specificity; technical thinnings are not biological replicates.':'Exploratory: raw specificity is depth-confounded; arrows are not developmental flow.';layout.title={text:`${s.method.label||s.methodId} · ${s.field.label||s.field.id}${s.field.id==='temporal'?` · ${DATA.stages[s.t]}–${DATA.stages[s.t+1]} hpf`:''}<br><sup>${mode}</sup>`,x:.04,xanchor:'left',font:{size:13}};layout.margin.t=66;}
 if(s.dim===3)layout.scene={xaxis:{title:'UMAP 1',showgrid:false,zeroline:false},yaxis:{title:'UMAP 2',showgrid:false,zeroline:false},zaxis:{title:'UMAP 3',showgrid:false,zeroline:false},aspectmode:'data',camera:{eye:{x:1.25,y:1.25,z:.9}}};else{layout.xaxis={title:'UMAP 1',showgrid:false,zeroline:false};layout.yaxis={title:'UMAP 2',showgrid:false,zeroline:false,scaleanchor:'x',scaleratio:1};}return layout;
}
function showLegend(s){
 const container=$('legend-items');container.replaceChildren();const key=s.metric.startsWith('annotation:')?s.metric.slice(11):null;$('legend').classList.toggle('hidden',!key);if(!key)return;
 for(const value of annotations[key].labels){const item=document.createElement('span'),dot=document.createElement('span');dot.className='legend-dot';dot.style.backgroundColor=annotationColor(key,value);item.appendChild(dot);item.appendChild(document.createTextNode(value));container.appendChild(item);}
}
async function renderAtlas(){
 const revision=++renderCounter,s=state(),coords=DATA[`umap${s.dim}d`],values=metricValues(s),numeric=!s.metric.startsWith('annotation:');
 $('metric').querySelector('option[value="specificity"]').textContent=s.field.id==='structural'?`${DATA.stages[T-1]} hpf reference specificity`:'Specificity at interval start';
 $('density-label').textContent=s.density;$('length-label').textContent=s.length.toFixed(2);$('coherence-label').textContent=s.coherence.toFixed(2);$('field-note').textContent=s.field.description||s.field.label;$('method-note').textContent=[s.method.description,Array.isArray(s.method.notes)?s.method.notes.join(' '):s.method.notes].filter(Boolean).join(' ');$('color-note').textContent=colorNote(s,values);
 const arrows=arrowEligibility(s,coords),drawn=drawingIndices(s,$('show-arrows').checked?arrows.indices:[]),missing=numeric?drawn.filter(i=>!finite(values[i])):[];
 const custom=DATA.peaks.map((p,i)=>[i,peakId(i),label(p.nearest_gene),label(p.linked_gene),label(p.celltype),label(p.leiden_coarse),numeric?(finite(values[i])?values[i]:null):values[i]]);
 const colors=colorSpec(s,values);colors.color=drawn.map(i=>colors.color[i]);
 const trace={type:s.dim===3?'scatter3d':'scattergl',mode:'markers',x:drawn.map(i=>coords[i][0]),y:drawn.map(i=>coords[i][1]),customdata:drawn.map(i=>custom[i]),marker:{size:drawn.map(i=>numeric&&!finite(values[i])?0:(s.dim===3?2.7:3.1)),opacity:s.dim===3?1:.75,...colors},hovertemplate:'%{customdata[1]}<br>Nearest gene: %{customdata[2]}<br>Linked gene: %{customdata[3]}<br>Celltype: %{customdata[4]} · cluster: %{customdata[5]}<br>Color value: %{customdata[6]}<extra></extra>',name:'Displayed peaks',showlegend:false};if(s.dim===3)trace.z=drawn.map(i=>coords[i][2]);const traces=[trace];
 if(missing.length){const gray={type:trace.type,mode:'markers',x:missing.map(i=>coords[i][0]),y:missing.map(i=>coords[i][1]),customdata:missing.map(i=>custom[i]),marker:{size:s.dim===3?2.7:3.1,color:'#aeb8c2',opacity:s.dim===3?1:.65},hovertemplate:'%{customdata[1]}<br>Nearest gene: %{customdata[2]}<br>Unavailable / unsupported value<extra></extra>',showlegend:false,name:'Unavailable'};if(s.dim===3)gray.z=missing.map(i=>coords[i][2]);traces.push(gray);}
 if($('show-arrows').checked&&arrows.indices.length)traces.push(...arrowTraces(s,coords,arrows.indices));
 const chosen={type:trace.type,mode:'markers',x:[coords[selected][0]],y:[coords[selected][1]],marker:{size:s.dim===3?6:11,color:'#172b43',symbol:'circle-open',line:{color:'#172b43',width:2}},hoverinfo:'skip',showlegend:false};if(s.dim===3)chosen.z=[coords[selected][2]];traces.push(chosen);
 await Plotly.react('atlas',traces,atlasLayout(s),config);if(revision!==renderCounter)return;
 $('render-status').textContent=`${drawn.length.toLocaleString()} drawn / ${N.toLocaleString()} selectable profiles / ${Number(DATA.total_peaks||N).toLocaleString()} atlas peaks · ${arrows.eligible.toLocaleString()} supported directions in the profile subset · ${$('show-arrows').checked?arrows.indices.length:0} arrows shown. Model computed on ${Number(DATA.analysis_peak_count||DATA.total_peaks||N).toLocaleString()} peaks.`;
 showLegend(s);if(!atlasInitialized){$('atlas').on('plotly_click',e=>{const point=e.points.find(p=>p.customdata);if(point){selected=Number(point.customdata[0]);renderAll();}});atlasInitialized=true;}
}
function curveLayout(title,ytitle,yrange){return {title:{text:title,font:{size:14},x:.04,y:.99,yanchor:'top'},margin:{l:58,r:12,t:68,b:42},xaxis:{title:'Developmental time (hpf)',tickvals:DATA.stages,range:[DATA.stages[0]-.4,DATA.stages[T-1]+.4]},yaxis:{title:ytitle,...(yrange?{range:yrange}:{rangemode:'tozero'})},legend:{orientation:'h',x:0,y:1.03,yanchor:'bottom',font:{size:10}},font:{family:'system-ui, sans-serif',color:'#19324a'},paper_bgcolor:'white',plot_bgcolor:'white',hovermode:'x unified',uirevision:'curves'};}
function renderCurves(){
 const s=state(),p=DATA.peaks[selected],stages=DATA.stages,celltype=DATA.celltypes[s.ct],row=DATA.accessibility[selected],traces=[];
 if($('all-celltypes').checked)for(let c=0;c<C;c++)if(c!==s.ct)traces.push({type:'scatter',mode:'lines',x:stages,y:stages.map((t,k)=>validCondition(c,k)&&finite(row[c*T+k])?row[c*T+k]:null),line:{color:'#c2cdd7',width:1},opacity:.5,showlegend:false,hoverinfo:'skip',connectgaps:false});
 traces.push({type:'scatter',mode:'lines+markers',x:stages,y:stages.map((t,k)=>validCondition(s.ct,k)&&finite(row[s.ct*T+k])?row[s.ct*T+k]:null),line:{color:'#1c567f',width:2},marker:{size:6},name:'Supported pseudobulk',connectgaps:false});
 const low=stages.map((t,k)=>!validCondition(s.ct,k)&&finite(row[s.ct*T+k])?row[s.ct*T+k]:null);if(low.some(x=>x!==null))traces.push({type:'scatter',mode:'markers',x:stages,y:low,marker:{size:7,color:'#a3aeb9',symbol:'x'},name:'Below cell-count filter'});
 const band=[{type:'rect',xref:'x',yref:'paper',x0:stages[s.t],x1:stages[s.t+1],y0:0,y1:1,fillcolor:'#dce8f3',opacity:.3,line:{width:0},layer:'below'}];const layout=curveLayout(`${celltype}: accessibility`,DATA.accessibility_units||'CPM');layout.shapes=band;Plotly.react('curves',traces,layout,config);
 const spec={type:'scatter',mode:'lines+markers',x:stages,y:Array.from(s.method.specificity[selected],v=>finite(v)?v:null),line:{color:'#79569a',width:2},marker:{size:6},name:s.method.label||s.methodId,connectgaps:false};const sl=curveLayout('Specificity across the fixed celltype panel','Specificity',[-.025,1.025]);sl.shapes=band;Plotly.react('specificity-curve',[spec],sl,config);
 $('selected-title').textContent=peakLabel(selected);
 const counts=DATA.condition_n_cells?`${number(DATA.condition_n_cells[s.ct][s.t],0)} → ${number(DATA.condition_n_cells[s.ct][s.t+1],0)} cells`:'cell counts unavailable';
 const support=scalarAt(s.field[`support${s.dim}d`]??s.field.support,selected,s.t,NaN),coherence=scalarAt(s.field[`coherence${s.dim}d`]??s.field.coherence,selected,s.t,NaN),stability=scalarAt(s.field[`stability${s.dim}d`]??s.field.stability,selected,s.t,NaN);
 $('selected-note').textContent=`${peakId(selected)} · global row ${p.global_index??'unavailable'} · nearest gene ${label(p.nearest_gene)} · linked gene ${label(p.linked_gene)}. ${stages[s.t]}–${stages[s.t+1]} hpf: specificity rate ${number(s.method.specificity_rate[selected][s.t],4)} /hpf; ${celltype} opening rate ${number(openingRate(selected,s.ct,s.t),3)} CPM/hpf (${counts}). Direction support ${number(support,0)} neighbors; coherence ${number(coherence,2)}.`+(finite(stability)?` Parameter-choice stability ${number(stability,2)}; this is sensitivity to analysis settings, not replicate uncertainty.`:'');
}
async function renderAll(){try{await renderAtlas();renderCurves();$('error').style.display='none';}catch(error){$('error').textContent='Viewer error: '+error.message;$('error').style.display='block';console.error(error);}}
function updateFields(){const previous=$('field').value;$('field').replaceChildren();const m=DATA.methods[$('method').value];m.fields.forEach(f=>option($('field'),f.id,f.label||f.id));$('field').value=m.fields.some(f=>f.id===previous)?previous:(DATA.default_field||m.fields[0].id);}
function searchPeaks(){const query=$('search').value.trim().toLowerCase(),select=$('search-results');select.replaceChildren();if(!query){select.classList.add('hidden');$('search-status').textContent='Enter a gene symbol or peak ID. Search covers the displayed subset.';return;}const hits=[];for(let i=0;i<N;i++){const p=DATA.peaks[i];if([peakId(i),p.nearest_gene,p.linked_gene].some(v=>v&&String(v).toLowerCase().includes(query)))hits.push(i);}select.classList.toggle('hidden',!hits.length);hits.slice(0,100).forEach(i=>option(select,i,peakLabel(i)));$('search-status').textContent=hits.length?`${hits.length.toLocaleString()} matching displayed peaks${hits.length>100?'; showing the first 100':''}. Gene annotations are associations, not verified targets.`:'No matching peak in the displayed subset. The full atlas is not searched here.';if(hits.length){selected=hits[0];renderAll();}}
function init(){
 if(DATA.interpretation_notice){$('interpretation-notice').textContent=DATA.interpretation_notice;$('interpretation-notice').classList.remove('hidden');}
 $('title').textContent=DATA.title||'Chromatin accessibility on the existing peak atlas';document.title=$('title').textContent;$('dimensions').textContent=`${N.toLocaleString()} selectable profiles / ${Number(DATA.total_peaks||N).toLocaleString()} peaks · ${C} celltypes × ${T} stages`;
 for(const [key,method] of Object.entries(DATA.methods))option($('method'),key,method.label||key);$('method').value=DATA.default_method||Object.keys(DATA.methods)[0];updateFields();DATA.celltypes.forEach((c,i)=>option($('celltype'),i,c));for(let t=0;t<T-1;t++)option($('interval'),t,`${DATA.stages[t]} → ${DATA.stages[t+1]} hpf`);$('interval').value=String(Math.max(0,Math.min(T-2,DATA.default_interval??2)));
 if(!DATA.umap3d)$('dimension').querySelector('option[value="3"]').remove();$('dimension').value=String(DATA.umap3d?(DATA.default_dimension||2):2);
 for(const key of ['celltype','leiden_coarse','lineage']){const labels=[...new Set(DATA.peaks.map(p=>label(p[key])))].sort();annotations[key]={labels,colors:{}};labels.forEach((value,i)=>{annotations[key].colors[value]=DATA.annotation_colors?.[key]?.[value]||(value==='unassigned'?'#9ca8b4':`hsl(${(i*137.508)%360},52%,48%)`);});if(labels.length===1&&labels[0]==='unassigned')$('metric').querySelector(`option[value="annotation:${key}"]`).remove();}
 if(!DATA.peaks.some(p=>p.broad_reference))$('metric').querySelector('option[value="broad_reference"]').remove();if(DATA.default_metric&&[...$('metric').options].some(o=>o.value===DATA.default_metric))$('metric').value=DATA.default_metric;
 selected=Math.max(0,Math.min(N-1,DATA.default_peak||0));$('notes').textContent=Array.isArray(DATA.notes)?DATA.notes.join(' '):(DATA.notes||'');$('panel-note').textContent=DATA.specificity_celltypes?`Specificity uses a fixed panel of ${DATA.specificity_celltypes.length} celltypes at all stages: ${DATA.specificity_celltypes.join(', ')}. Accessibility and opening/closing displays can use all ${C} celltypes when the supplied condition mask supports both endpoints.`:'Specificity uses the celltype panel specified by the analysis. Missing or excluded conditions are not interpreted as zero accessibility.';
 $('method').addEventListener('change',()=>{updateFields();renderAll();});for(const id of ['field','dimension','metric','celltype','interval'])$(id).addEventListener('change',renderAll);for(const id of ['density','length','coherence'])$(id).addEventListener('input',renderAtlas);$('show-arrows').addEventListener('change',renderAtlas);$('all-celltypes').addEventListener('change',renderCurves);$('reset').addEventListener('click',()=>{viewReset++;renderAtlas();});$('png').addEventListener('click',()=>Plotly.downloadImage('atlas',{format:'png',width:1600,height:1200,scale:1,filename:'real_peak_atlas_'+state().methodId+'_'+state().field.id+'_'+state().dim+'d'}));$('search-button').addEventListener('click',searchPeaks);$('search').addEventListener('keydown',e=>{if(e.key==='Enter')searchPeaks();});$('search-results').addEventListener('change',()=>{selected=Number($('search-results').value);renderAll();});renderAll();
}
init();
'''

"""Export a self-contained interactive peak-specialization arrow atlas.

Every dot is a peak; arrows are supplied by the analysis and never computed by
this viewer. Display subsampling, coherence filtering, and arrow-length scaling
change presentation only. Plotly is embedded, so the HTML needs no network.

Public API: ``export_viewer(payload: dict, output_path: Path) -> Path``.
Matrices may be lists or NumPy arrays. State columns must be celltype-major,
with stages ordered within each celltype. ``fields`` contain static ``N x d`` or
interval-specific ``N x (T - 1) x d`` vectors for both embeddings. Per-field
``support``, ``coherence``, and optional ``arrow_mask`` may be N or N x (T - 1).
Dimension-specific variants (e.g. ``coherence2d`` and ``arrow_mask3d``) override
the corresponding general field for that embedding. Optional ``color_limits``
sets a shared symmetric color bound for each rate metric across all intervals.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from plotly.offline import get_plotlyjs


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot JSON-encode {type(value).__name__}")


def _validate(payload):
    n = len(payload["peaks"])
    c = len(payload["celltypes"])
    t = len(payload["stages"])
    if n < 1 or c < 1 or t < 2:
        raise ValueError("The viewer needs peaks, celltypes, and at least two stages")
    for key, expected in (("umap2d", (n, 2)), ("umap3d", (n, 3)),
                          ("observed", (n, c * t)), ("truth_state", (n, c * t))):
        a = np.asarray(payload[key])
        if a.shape != expected or not np.isfinite(a).all():
            raise ValueError(f"{key} must be a finite array with shape {expected}")
    for key in ("specificity", "truth_specificity"):
        if np.asarray(payload[key]).shape != (n, t):
            raise ValueError(f"{key} must have shape {(n, t)}")
    for key in ("specificity_rate", "truth_specificity_rate"):
        if key in payload and np.asarray(payload[key]).shape != (n, t - 1):
            raise ValueError(f"{key} must have shape {(n, t - 1)}")
    if "opening_rate" in payload:
        shape = np.asarray(payload["opening_rate"]).shape
        if shape not in ((n, c * t), (n, c * (t - 1))):
            raise ValueError("opening_rate must have condition or interval columns")
    for key, limit in payload.get("color_limits", {}).items():
        if not np.isfinite(limit) or float(limit) <= 0:
            raise ValueError(f"color_limits[{key}] must be finite and positive")
    if not payload.get("fields"):
        raise ValueError("At least one supplied arrow field is required")
    ids = [f["id"] for f in payload["fields"]]
    if len(ids) != len(set(ids)):
        raise ValueError("Arrow field IDs must be unique")
    for f in payload["fields"]:
        for dim in (2, 3):
            a = np.asarray(f[f"vectors{dim}d"])
            if a.shape not in ((n, dim), (n, t - 1, dim)):
                raise ValueError(f"Invalid vector shape in field {f['id']}: {a.shape}")
        for key in ("support", "coherence", "arrow_mask", "support2d", "support3d", "coherence2d", "coherence3d", "arrow_mask2d", "arrow_mask3d"):
            if key in f and np.asarray(f[key]).shape not in ((n,), (n, t - 1)):
                raise ValueError(f"Invalid {key} shape in field {f['id']}")


def export_viewer(payload: dict, output_path: Path) -> Path:
    """Validate inputs and write one offline HTML containing data and Plotly."""
    _validate(payload)
    encoded = json.dumps(payload, default=_json_default, allow_nan=False,
                         separators=(",", ":"))
    # Prevent an annotation containing </script> from escaping the data block.
    encoded = encoded.replace("<", "\\u003c").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    html = _HTML.replace("__PLOTLY__", get_plotlyjs()).replace("__DATA__", encoded).replace("__APP__", _JS)
    out.write_text(html, encoding="utf-8")
    return out


_HTML = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Peak accessibility: specialization and developmental change</title>
<style>
:root{color-scheme:light;--ink:#19324a;--muted:#53677a;--line:#d8e1e9;--blue:#155eaa}
*{box-sizing:border-box}body{margin:0;background:#eef3f7;color:var(--ink);font:14px system-ui,sans-serif}
main{max-width:1700px;margin:auto;padding:24px}h1{font-size:27px;line-height:1.2;margin:0 0 10px}h2{font-size:16px;margin:0 0 8px}p{line-height:1.5;margin:8px 0}
.subtitle,.note,.small{color:var(--muted)}.subtitle{max-width:1100px}.badge{display:inline-block;padding:4px 8px;background:#e2edf7;border-radius:5px;font-size:12px;font-weight:650}
.controls{display:flex;gap:15px;flex-wrap:wrap;background:white;border:1px solid var(--line);border-radius:9px;padding:15px;margin:18px 0 12px;align-items:end}
label{display:flex;flex-direction:column;gap:5px;font-size:12px;font-weight:600}select,input[type=range],button{font:inherit}select{max-width:390px;padding:7px;border:1px solid #bccad7;border-radius:4px;background:white;color:var(--ink)}
input[type=range]{width:120px;accent-color:var(--blue)}.field-control{flex:1;min-width:250px}.field-control select{width:100%;max-width:100%}.check{display:flex;flex-direction:row;align-items:center;gap:5px;font-weight:400}
.toolbar{display:flex;gap:18px;align-items:center;flex-wrap:wrap;padding:5px 2px 12px}.toolbar label{min-width:130px}.toolbar .check{min-width:0}.display-note{font-size:12px;margin-left:auto;max-width:440px;color:var(--muted)}
.grid{display:grid;grid-template-columns:1.5fr 1fr;gap:15px}.panel{background:white;border:1px solid var(--line);border-radius:9px;padding:12px;min-width:0}
#atlas{height:690px}#curves{height:330px}#specificity-curve{height:280px}#field-note{background:#e7eff7;border-left:3px solid #3979ad;padding:10px 13px;margin:10px 0;font-size:13px;line-height:1.45}
#atlas-caption,#selected-note,#color-note,#render-status{font-size:12px;color:var(--muted)}#selected-note{min-height:40px;padding:0 10px}#render-status{font-variant-numeric:tabular-nums}#error{display:none;color:#8b1b1b;background:#fee9e9;padding:12px;white-space:pre-wrap}
button{border:1px solid #b9c8d6;background:white;border-radius:4px;padding:6px 10px;color:var(--ink);cursor:pointer}button:hover{background:#edf4fb}footer{font-size:12px;color:var(--muted);padding:12px 0}
@media(max-width:1100px){.grid{grid-template-columns:1fr}#atlas{height:560px}.display-note{margin-left:0}}@media(max-width:650px){main{padding:12px}h1{font-size:23px}#atlas{height:440px}.controls{gap:10px}select{max-width:280px}}
</style></head><body><main>
<h1 id="title">Peak accessibility: specialization and developmental change</h1>
<p class="subtitle"><strong>Every dot is one genomic peak.</strong> Fixed UMAP coordinates summarize each peak’s accessibility across celltypes and developmental stages.</p>
<p><span class="badge" id="dimensions"></span> <span class="badge">Synthetic ground truth</span></p>
<div id="field-note"></div><div id="error"></div>
<div class="controls">
<label class="field-control">Arrow interpretation<select id="field"></select></label>
<label>View<select id="dimension"><option value="2">2D peak UMAP</option><option value="3">3D peak UMAP</option></select></label>
<label>Color shows<select id="metric"><option value="family">Core / celltype family</option><option value="specificity">Estimated specificity at interval start</option><option value="truth_specificity">True specificity at interval start</option><option value="specificity_rate">Estimated specialization rate</option><option value="truth_specificity_rate">True specialization rate</option><option value="opening_rate">Estimated opening / closing rate</option><option value="observed">Measured accessibility at interval start</option><option value="truth_state">True accessibility at interval start</option></select></label>
<label>Celltype<select id="celltype"></select></label>
<label>Developmental interval<select id="interval"></select></label>
</div>
<div class="toolbar">
<label>Maximum arrows <span id="density-label"></span><input id="density" type="range" min="20" max="300" step="20" value="140"></label>
<label>Display length × <span id="length-label"></span><input id="length" type="range" min="0.25" max="2" step="0.05" value="1"></label>
<label>Minimum coherence <span id="coherence-label"></span><input id="coherence" type="range" min="0" max="1" step="0.05" value="0"></label>
<label class="check"><input id="show-arrows" type="checkbox" checked> Show arrows</label>
<button id="reset">Reset view</button><button id="png">Save atlas PNG</button>
<span class="display-note">Arrow count, filtering, and length affect display only. UMAP arrow length is not a physical speed.</span>
</div>
<div class="grid"><section class="panel"><div id="atlas"></div><p id="color-note"></p><p id="render-status"></p><p id="atlas-caption">Click a peak to inspect its trajectory. In 3D, drag to rotate; use the toolbar to zoom or reset.</p></section>
<section class="panel"><h2 id="selected-title">Selected peak</h2><div id="curves"></div><p id="selected-note"></p><label class="check"><input id="all-celltypes" type="checkbox"> Show true trajectories in other celltypes</label><div id="specificity-curve"></div></section></div>
<footer id="notes"></footer>
</main><script>__PLOTLY__</script><script>const DATA=__DATA__;</script><script>__APP__</script></body></html>'''

_JS = r'''
'use strict';
const $=id=>document.getElementById(id);
const FAMILY=['#d5a32d','#3274ad','#32a28f','#db8553','#9564ad','#728292','#bd709b'];
const DIVERGE=[[0,'#2166ac'],[.5,'#d8dee4'],[1,'#b2182b']];
const SEQUENTIAL=[[0,'#f5f7f9'],[.25,'#b8d9e8'],[.6,'#4c9dbb'],[1,'#174a78']];
const config={responsive:true,displaylogo:false,toImageButtonOptions:{format:'png',scale:2}};
let selected=0,atlasInitialized=false,viewReset=0;
const N=DATA.peaks.length,T=DATA.stages.length,C=DATA.celltypes.length;
const finite=x=>Number.isFinite(x);
function scalarAt(values,i,t,fallback=0){if(values===undefined||values===null)return fallback;let v=values[i];return Array.isArray(v)?v[t]:v;}
function vectorAt(values,i,t){let v=values[i];return Array.isArray(v[0])?v[t]:v;}
function familyCode(p){if(p.true_program==='constitutive_core')return 0;let f=p.true_family===null?-1:Number(p.true_family);return finite(f)&&f>=0?f+1:-1;}
function familyColor(p){let k=familyCode(p);return k<0?'#8c98a5':FAMILY[k%FAMILY.length];}
function peakId(i){let p=DATA.peaks[i];return p.peak||p.peak_id||p.name||('peak '+i);}
function peakProgram(i){return String(DATA.peaks[i].true_program||'unlabeled program').replaceAll('_',' ');}
function targetLabel(p){const target=p.primary_celltype;if(target&&DATA.celltypes.includes(target))return target;const index=Number(p.true_celltype_index);if(p.true_celltype_index!==null&&Number.isInteger(index)&&index>=0&&index<C)return DATA.celltypes[index];return 'all celltypes / no target';}
function number(x,n=3){return finite(x)?x.toFixed(n):'unavailable';}
function option(select,value,label){let o=document.createElement('option');o.value=value;o.textContent=label;select.appendChild(o);}
function state(){return {dim:Number($('dimension').value),ct:Number($('celltype').value),t:Number($('interval').value),metric:$('metric').value,field:DATA.fields.find(f=>f.id===$('field').value),density:Number($('density').value),length:Number($('length').value),coherence:Number($('coherence').value)};}
function openingRate(i,c,t){let row=DATA.opening_rate?.[i];if(!row)return null;return row[c*(row.length===C*T?T:T-1)+t];}
function metricValues(s){
 if(s.metric==='family')return DATA.peaks.map(familyCode);
 if(s.metric==='opening_rate')return DATA.peaks.map((p,i)=>openingRate(i,s.ct,s.t));
 if(['observed','truth_state'].includes(s.metric))return DATA[s.metric].map(row=>row[s.ct*T+s.t]);
 return DATA[s.metric].map(row=>row[s.t]);
}
function robustLimit(values){let a=values.filter(finite).map(Math.abs).sort((x,y)=>x-y);return a.length?Math.max(1e-5,a[Math.floor(.97*(a.length-1))]):1;}
function colorSpec(s,values){
 if(s.metric==='family')return {color:DATA.peaks.map(familyColor),showscale:false};
 const rate=s.metric.includes('rate'),fixed=DATA.color_limits?.[s.metric];const z=rate?(finite(fixed)&&fixed>0?fixed:robustLimit(values)):1;
 return {color:values,colorscale:rate?DIVERGE:SEQUENTIAL,cmin:rate?-z:0,cmax:z,showscale:true,colorbar:{title:{text:rate?'rate / hpf':'score'},thickness:13,len:.62,x:1.01}};
}
function rateColorNote(s){const limit=DATA.color_limits?.[s.metric];return finite(limit)&&limit>0?` Colors share a fixed ±${limit.toPrecision(3)} /hpf scale across ${s.metric==='opening_rate'?'celltypes and ':''}intervals; values beyond this limit saturate.`:' Colors saturate at the 97th percentile of absolute rates in this view.';}
function colorNote(s,values){
 const stage=DATA.stages[s.t],end=DATA.stages[s.t+1];
 if(s.metric==='family')return 'Gold = constitutively accessible core. Blue, teal, orange, and purple = synthetic celltype families. Gray = controls without a family assignment. Family annotations do not determine UMAP coordinates.';
 if(s.metric==='specificity_rate'||s.metric==='truth_specificity_rate')return `Specificity change from ${stage} to ${end} hpf: red = increasing specificity, blue = decreasing specificity. This can arise through opening, closing, or both.`+rateColorNote(s);
 if(s.metric==='opening_rate')return `Accessibility change in ${DATA.celltypes[s.ct]}, ${stage}–${end} hpf: red = opening, blue = closing. A peak may close in one celltype while becoming more specific to another.`+rateColorNote(s);
 if(s.metric.includes('specificity'))return `Celltype specificity at ${stage} hpf. Zero denotes broad accessibility; larger scores denote more restricted accessibility. Very weak signals and missing comparisons require separate masking in the analysis.`;
 return `Accessibility in ${DATA.celltypes[s.ct]} at ${stage} hpf. The atlas coordinates stay fixed when the selected celltype or interval changes.`;
}
// Greedy farthest-point sampling covers the displayed atlas without random jitter.
// Selection changes only the rendered arrow subset, never the supplied field.
function spatialSample(indices,coords,count,weights){
 if(indices.length<=count)return indices;
 let first=0;for(let j=1;j<indices.length;j++)if(weights[indices[j]]>weights[indices[first]])first=j;
 const distances=new Float64Array(indices.length);distances.fill(Infinity);
 const chosen=[],used=new Uint8Array(indices.length);let next=first;
 for(let k=0;k<count;k++){
   chosen.push(indices[next]);used[next]=1;let best=-1,bestDistance=-1;const p=coords[indices[next]];
   for(let j=0;j<indices.length;j++){
     if(used[j])continue;let d=0;for(let a=0;a<p.length;a++){let q=coords[indices[j]][a]-p[a];d+=q*q;}
     if(d<distances[j])distances[j]=d;
     if(distances[j]>bestDistance){bestDistance=distances[j];best=j;}
   }
   if(best<0)break;next=best;
 }
 return chosen;
}
function arrowEligibility(s,coords){
 const vectors=s.field[`vectors${s.dim}d`],eligible=[],weights=new Float64Array(N);let supplied=0;
 for(let i=0;i<N;i++){
   const v=vectorAt(vectors,i,s.t),support=scalarAt(s.field[`support${s.dim}d`]??s.field.support,i,s.t,1),coherence=scalarAt(s.field[`coherence${s.dim}d`]??s.field.coherence,i,s.t,1),mask=scalarAt(s.field[`arrow_mask${s.dim}d`]??s.field.arrow_mask,i,s.t,true);
   if(!mask||!finite(support)||support<=0||!v.every(finite)||v.every(x=>Math.abs(x)<1e-12))continue;
   supplied++;
   if(!finite(coherence)||coherence<s.coherence)continue;
   weights[i]=support*Math.max(0,coherence);eligible.push(i);
 }
 return {indices:spatialSample(eligible,coords,s.density,weights),supplied,eligible:eligible.length};
}
function arrowTraces(s,coords,indices){
 const xs=[],ys=[],zs=[],cx=[],cy=[],cz=[],cu=[],cv=[],cw=[];
 const vectors=s.field[`vectors${s.dim}d`];
 for(const i of indices){
   const p=coords[i],v=vectorAt(vectors,i,s.t).map(x=>x*s.length),q=p.map((x,a)=>x+v[a]);
   const len=Math.hypot(...v);if(!len)continue;
   xs.push(p[0],q[0],null);ys.push(p[1],q[1],null);
   if(s.dim===3){zs.push(p[2],q[2],null);cx.push(q[0]);cy.push(q[1]);cz.push(q[2]);cu.push(v[0]*.24);cv.push(v[1]*.24);cw.push(v[2]*.24);}
   else{
     const a=.25,b=.13,nx=-v[1],ny=v[0];
     xs.push(q[0]-a*v[0]+b*nx,q[0],q[0]-a*v[0]-b*nx,null);
     ys.push(q[1]-a*v[1]+b*ny,q[1],q[1]-a*v[1]-b*ny,null);
   }
 }
 const line={type:s.dim===3?'scatter3d':'scattergl',mode:'lines',x:xs,y:ys,line:{color:'#253c50',width:s.dim===3?3:1.4},hoverinfo:'skip',showlegend:false,name:'Supplied direction'};
 if(s.dim===2)return [line];
 line.z=zs; // Raw cone sizes keep each head proportional to its own shaft.
 return [line,{type:'cone',x:cx,y:cy,z:cz,u:cu,v:cv,w:cw,anchor:'tip',sizemode:'raw',sizeref:1,colorscale:[[0,'#253c50'],[1,'#253c50']],showscale:false,hoverinfo:'skip',showlegend:false,name:'Direction arrowheads',lighting:{ambient:1,diffuse:0}}];
}
function atlasLayout(s){
 const lay={margin:{l:15,r:48,t:12,b:25},paper_bgcolor:'white',plot_bgcolor:'white',font:{family:'system-ui, sans-serif',color:'#19324a'},showlegend:false,uirevision:`${s.dim}-${viewReset}`,hovermode:'closest'};
 if(s.dim===3)lay.scene={xaxis:{title:'UMAP 1',showgrid:false,zeroline:false},yaxis:{title:'UMAP 2',showgrid:false,zeroline:false},zaxis:{title:'UMAP 3',showgrid:false,zeroline:false},aspectmode:'data',camera:{eye:{x:1.45,y:1.45,z:1.0}}};
 else {lay.margin={l:58,r:75,t:12,b:48};lay.xaxis={title:'UMAP 1',showgrid:false,zeroline:false};lay.yaxis={title:'UMAP 2',showgrid:false,zeroline:false,scaleanchor:'x',scaleratio:1};}
 return lay;
}
async function renderAtlas(){
 const s=state(),coords=DATA[`umap${s.dim}d`],values=metricValues(s),colors=colorSpec(s,values);
 $('density-label').textContent=s.density;$('length-label').textContent=s.length.toFixed(2);$('coherence-label').textContent=s.coherence.toFixed(2);
 $('field-note').textContent=s.field.description||s.field.label;
 $('color-note').textContent=colorNote(s,values)+(s.metric!=='family'?' Gray points denote unavailable or masked values.':'');
 const traces=[{type:s.dim===3?'scatter3d':'scattergl',mode:'markers',x:coords.map(p=>p[0]),y:coords.map(p=>p[1]),customdata:DATA.peaks.map((p,i)=>[i,peakId(i),peakProgram(i),finite(values[i])?values[i]:null]),marker:{size:DATA.peaks.map(p=>familyCode(p)===0?(s.dim===3?3.4:5.1):(s.dim===3?2.5:3.9)),opacity:.75,...colors},hovertemplate:'%{customdata[1]}<br>%{customdata[2]}<br>Color value: %{customdata[3]:.4f}<extra></extra>',name:'Peaks',showlegend:false}];
 if(s.dim===3)traces[0].z=coords.map(p=>p[2]);
 const missing=values.map((v,i)=>finite(v)?-1:i).filter(i=>i>=0);
 if(s.metric!=='family'&&missing.length){
   missing.forEach(i=>traces[0].marker.size[i]=0);
   const unavailable={type:s.dim===3?'scatter3d':'scattergl',mode:'markers',x:missing.map(i=>coords[i][0]),y:missing.map(i=>coords[i][1]),customdata:missing.map(i=>[i,peakId(i),peakProgram(i)]),marker:{size:s.dim===3?2.5:3.9,color:'#aeb8c2',opacity:.7},hovertemplate:'%{customdata[1]}<br>%{customdata[2]}<br>Unavailable / masked value<extra></extra>',showlegend:false,name:'Unavailable'};
   if(s.dim===3)unavailable.z=missing.map(i=>coords[i][2]);traces.push(unavailable);
 }
 const eligible=arrowEligibility(s,coords);
 if($('show-arrows').checked&&eligible.indices.length)traces.push(...arrowTraces(s,coords,eligible.indices));
 const h={type:s.dim===3?'scatter3d':'scattergl',mode:'markers',x:[coords[selected][0]],y:[coords[selected][1]],marker:{size:s.dim===3?6.5:12,color:'#fff',line:{color:'#111827',width:2},symbol:s.dim===3?'circle':'circle-open'},hoverinfo:'skip',showlegend:false,name:'Selected peak'};
 if(s.dim===3)h.z=[coords[selected][2]];traces.push(h);
 await Plotly.react('atlas',traces,atlasLayout(s),config);
 $('render-status').textContent=`${N.toLocaleString()} peaks · ${eligible.eligible.toLocaleString()} supported directions after display filter · ${$('show-arrows').checked?eligible.indices.length:0} arrows shown. Spatial subsampling is deterministic.`;
 if(!atlasInitialized){$('atlas').on('plotly_click',e=>{const p=e.points.find(q=>Array.isArray(q.customdata));if(p){selected=Number(p.customdata[0]);renderAll();}});atlasInitialized=true;}
}
function curveLayout(title,ytitle,yrange){return {title:{text:title,font:{size:14},x:.04,y:.99,yanchor:'top'},margin:{l:50,r:12,t:68,b:42},xaxis:{title:'Developmental time (hpf)',tickvals:DATA.stages,range:[DATA.stages[0]-.4,DATA.stages[T-1]+.4]},yaxis:{title:ytitle,...(yrange?{range:yrange}:{})},legend:{orientation:'h',x:0,y:1.03,yanchor:'bottom',font:{size:10}},font:{family:'system-ui, sans-serif',color:'#19324a'},paper_bgcolor:'white',plot_bgcolor:'white',hovermode:'x unified',uirevision:'curves'};}
function renderCurves(){
 const s=state(),p=DATA.peaks[selected],stages=DATA.stages,ct=DATA.celltypes[s.ct],traces=[];
 if($('all-celltypes').checked)for(let c=0;c<C;c++)if(c!==s.ct)traces.push({type:'scatter',mode:'lines',x:stages,y:DATA.truth_state[selected].slice(c*T,(c+1)*T),line:{color:'#cdd6df',width:1},opacity:.55,showlegend:false,hoverinfo:'skip'});
 traces.push({type:'scatter',mode:'lines+markers',x:stages,y:DATA.truth_state[selected].slice(s.ct*T,(s.ct+1)*T),line:{color:'#173e63',width:2},marker:{size:5},name:'True accessibility'},
 {type:'scatter',mode:'lines+markers',x:stages,y:DATA.observed[selected].slice(s.ct*T,(s.ct+1)*T),line:{color:'#dd7a35',width:1.5,dash:'dot'},marker:{size:7,symbol:'diamond'},name:'Measured accessibility'});
 const lay=curveLayout(`${ct}: accessibility`,'Accessibility',[-.03,1.03]);lay.shapes=[{type:'rect',xref:'x',yref:'paper',x0:stages[s.t],x1:stages[s.t+1],y0:0,y1:1,fillcolor:'#dce8f3',opacity:.32,line:{width:0},layer:'below'}];
 Plotly.react('curves',traces,lay,config);
 const spec=[{type:'scatter',mode:'lines+markers',x:stages,y:DATA.truth_specificity[selected],line:{color:'#173e63',width:2},name:'True specificity'},
 {type:'scatter',mode:'lines+markers',x:stages,y:DATA.specificity[selected],line:{color:'#dd7a35',width:1.5,dash:'dot'},marker:{symbol:'diamond'},name:'Estimated specificity'}];
 const sl=curveLayout('Specificity across celltypes','Specificity',[-.03,1.03]);sl.shapes=lay.shapes;
 Plotly.react('specificity-curve',spec,sl,config);
 $('selected-title').textContent=peakId(selected);
 const rate=scalarAt(DATA.specificity_rate,selected,s.t,null),truthRate=scalarAt(DATA.truth_specificity_rate,selected,s.t,null),opening=openingRate(selected,s.ct,s.t);
 $('selected-note').textContent=`${peakProgram(selected)}; target: ${targetLabel(p)}. ${stages[s.t]}–${stages[s.t+1]} hpf: estimated specificity rate ${number(rate,4)} /hpf; true ${number(truthRate,4)} /hpf. Opening/closing rate in ${ct}: ${number(opening,4)} /hpf. Selected peak is outlined on the atlas.`;
}
async function renderAll(){try{await renderAtlas();renderCurves();$('error').style.display='none';}catch(e){$('error').textContent='Viewer error: '+e.message;$('error').style.display='block';console.error(e);}}
function init(){
 $('title').textContent=DATA.title||'Peak accessibility: specialization and developmental change';document.title=$('title').textContent;
 $('dimensions').textContent=`${N.toLocaleString()} peaks × ${C} celltypes × ${T} stages`;
 DATA.fields.forEach(f=>option($('field'),f.id,f.label||f.id));$('field').value=DATA.default_field||DATA.fields[0].id;
 DATA.celltypes.forEach((c,i)=>option($('celltype'),i,c));for(let t=0;t<T-1;t++)option($('interval'),t,`${DATA.stages[t]} → ${DATA.stages[t+1]} hpf`);
 $('dimension').value=String(DATA.default_dimension||2);$('interval').value=String(Math.min(T-2,Math.max(0,DATA.default_interval??2)));
 $('metric').value=DATA.default_metric||'family';
 if(DATA.default_peak!==undefined)selected=Number(DATA.default_peak);
 $('notes').textContent=(Array.isArray(DATA.notes)?DATA.notes.join(' '):DATA.notes)||'Synthetic benchmark. Structural arrows orient relationships between different peaks. Rate-guided arrows project specificity change onto these atlas paths; they do not establish peak transport or ancestry. No biological ground truth has been inferred from UMAP geometry.';
 for(const o of [...$('metric').options])if(o.value!=='family'&&!DATA[o.value])o.remove();
 for(const id of ['field','dimension','metric','celltype','interval'])$(id).addEventListener('change',renderAll);
 for(const id of ['density','length','coherence'])$(id).addEventListener('input',renderAtlas);
 $('show-arrows').addEventListener('change',renderAtlas);$('all-celltypes').addEventListener('change',renderCurves);
 $('reset').addEventListener('click',()=>{viewReset++;renderAtlas();});
 $('png').addEventListener('click',()=>Plotly.downloadImage('atlas',{format:'png',width:1600,height:1200,scale:1,filename:'peak_umap_'+state().field.id+'_'+state().dim+'d'}));
 renderAll();
}
init();
'''

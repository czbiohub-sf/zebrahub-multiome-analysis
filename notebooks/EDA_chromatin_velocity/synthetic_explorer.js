/* Continuous profiles mirror evaluate_truth; validated against saved Python truth. */
function hierarchyGate(t, center, width) {
  const sig = x => 1/(1+Math.exp(-(x-center)/width));
  return (sig(t)-sig(10))/(sig(24)-sig(10));
}
function latentValue(p, c, t) {
  if (p.model === 'continuous_hierarchy') {
    const g=hierarchyGate(t,p.true_family_center_hpf,p.true_family_width_hpf);
    const h=hierarchyGate(t,p.true_celltype_center_hpf,p.true_celltype_width_hpf);
    const df=Math.floor(c/5)===p.true_family?0.18:-0.55;
    const dc=c===p.true_celltype_index?0.23:-0.55;
    return p.true_baseline+p.true_amplitude_scale*(p.true_family_weight*df*g+p.true_celltype_weight*dc*h);
  }
  const z = (t - p.true_center_hpf - p.lags[c]) / p.true_width_hpf;
  const sigmoid = 1 / (1 + Math.exp(-z));
  const pulse = Math.exp(-0.5 * z * z);
  let f = sigmoid;
  if (p.true_program === 'closing') f = 1 - sigmoid;
  if (p.true_program === 'transient_opening') f = pulse;
  if (p.true_program === 'transient_closing') f = 1 - pulse;
  if (p.true_program === 'opposite_celltypes' && c === p.partner_index) f = 1 - sigmoid;
  if (p.true_program.startsWith('stable_')) f = 0.8;
  return p.baseline[c] + p.true_amplitude * p.loadings[c] * f;
}

function initExplorer(d) {
  const ct = document.getElementById('celltype');
  const dimension = document.getElementById('dimension');
  const metric = document.getElementById('metric');
  const stage = document.getElementById('stage');
  const example = document.getElementById('example');
  const all = document.getElementById('all-celltypes');
  const atlas = document.getElementById('atlas');
  let selected = d.examples[0];
  if (d.title) document.querySelector('h1').textContent=d.title;
  if (d.note) document.querySelector('main > p.label').textContent=d.note;
  if (d.footnote) document.querySelector('main > p.label:last-child').textContent=d.footnote;
  for (const [key,value] of Object.entries(d.annotations || {})) metric.add(new Option(value.label,key));
  if (d.default_dimension) dimension.value=String(d.default_dimension);
  if (d.default_metric) metric.value=d.default_metric;
  const colors = ['#3b82f6', '#14b8a6', '#f59e0b', '#a855f7'];
  d.celltypes.forEach((c, i) => ct.add(new Option(`${c} · family ${Math.floor(i/5)+1}`, i)));
  function targetLabel(p) {
    if (p.true_program === 'constitutive_core') return 'all celltypes';
    if (p.true_program === 'family_bridge') return `family ${p.true_family+1}`;
    return p.primary_celltype;
  }
  d.examples.forEach(i => example.add(new Option(`${d.peaks[i].true_program.replaceAll('_',' ')} · ${targetLabel(d.peaks[i])}`, i)));
  document.getElementById('dimensions').textContent = `${d.peaks.length.toLocaleString()} peaks × 120 conditions`;

  function column() { return Number(ct.value)*6 + Number(stage.value); }
  function annotation() { return (d.annotations || {})[metric.value]; }
  function isRate() { return !annotation() && metric.value !== 'truth_state' && metric.value !== 'observed'; }
  function updateLabel() {
    const j = Number(stage.value), t = d.stages[j];
    document.getElementById('stage-label').textContent = isRate()
      ? (j < 5 ? `${t} → ${d.stages[j+1]} hpf` : `${t} hpf · no following stage`)
      : `${t} hpf`;
    document.getElementById('color-note').textContent = annotation() ? annotation().note : isRate()
      ? 'Red = opening · blue = closing · white = near zero. Rates are accessibility-score change per hpf.'
      : 'Color shows continuous accessibility, from 0 to 1. Every point is a peak; its position stays fixed.';
  }
  async function renderAtlas() {
    updateLabel();
    const coords = dimension.value === '3' ? d.umap3d : d.umap2d;
    const j = column(), supported = !isRate() || d.valid[j];
    const ann=annotation();
    const values = ann ? ann.values : d[metric.value].map(row => row[j]);
    const hover = d.peaks.map((p,i) => `${p.peak}<br>${p.true_program.replaceAll('_',' ')}<br>target: ${targetLabel(p)}<br>${supported ? values[i].toFixed(4) : 'No defined interval'}`);
    const trace = {
      type: dimension.value === '3' ? 'scatter3d' : 'scattergl', mode:'markers',
      x:coords.map(r=>r[0]), y:coords.map(r=>r[1]), text:hover,
      hovertemplate:'%{text}<extra></extra>',
      marker: {size:dimension.value==='3'?3:5, opacity:0.85,
        color:supported?values:'#a1aab8', showscale:supported,
        colorscale:isRate()?'RdBu':'Viridis', reversescale:isRate(),
        cmin:isRate()?-d.rate_limit:0, cmax:isRate()?d.rate_limit:1,
        colorbar:{title:{text:isRate()?'Score / hpf':'Accessibility'}, thickness:12}},
    };
    if (ann) {
      Object.assign(trace.marker,{colorscale:ann.colorscale,reversescale:false,cmin:ann.min,cmax:ann.max});
      trace.marker.colorbar={title:{text:ann.label},thickness:12};
      if (ann.tickvals) Object.assign(trace.marker.colorbar,{tickvals:ann.tickvals,ticktext:ann.ticktext});
    }
    if (d.emphasize_core) trace.marker.size=d.peaks.map(p=>p.true_program==='constitutive_core'?(dimension.value==='3'?5:8):(dimension.value==='3'?2.5:4));
    if (dimension.value === '3') trace.z = coords.map(r=>r[2]);
    const title = metric.options[metric.selectedIndex].text + (ann ? '' : ` · ${d.celltypes[Number(ct.value)]}`);
    const layout = {title:{text:title,font:{size:16}}, margin:{t:45,l:25,r:20,b:25},
      paper_bgcolor:'white', plot_bgcolor:'white', hovermode:'closest',
      xaxis:{title:'Peak UMAP 1',showticklabels:false,zeroline:false},
      yaxis:{title:'Peak UMAP 2',showticklabels:false,zeroline:false},
      scene:{aspectmode:'data', xaxis:{title:'Peak UMAP 1'},yaxis:{title:'Peak UMAP 2'},zaxis:{title:'Peak UMAP 3'}},
      uirevision:dimension.value};
    await Plotly.react(atlas, [trace], layout, {responsive:true,displaylogo:false});
  }
  function renderCurves() {
    const p = d.peaks[selected], c = Number(ct.value);
    const times = Array.from({length:141}, (_,i)=>10+i/10);
    const traces = [];
    if (all.checked) {
      d.celltypes.forEach((name,k) => {
        if (k === c) return;
        traces.push({x:times,y:times.map(t=>latentValue(p,k,t)), type:'scatter',mode:'lines',
          name, line:{color:colors[Math.floor(k/5)],width:1},opacity:0.28,showlegend:false,
          hovertemplate:`${name} truth<br>%{x:.1f} hpf: %{y:.3f}<extra></extra>`});
      });
    }
    traces.push({x:times, y:times.map(t=>latentValue(p,c,t)),type:'scatter',mode:'lines',
      name:`${d.celltypes[c]} truth`,line:{color:'#155eaa',width:3}});
    traces.push({x:d.stages,y:d.observed[selected].slice(c*6,c*6+6),type:'scatter',mode:'markers',
      name:`${d.celltypes[c]} measured mean`,marker:{color:'#0f172a',size:8,symbol:'diamond'}});
    Plotly.react('curves',traces,{title:{text:`Selected peak · ${p.true_program.replaceAll('_',' ')}`,font:{size:16}},
      xaxis:{title:'Developmental time (hpf)',tickvals:d.stages},yaxis:{title:'Accessibility score',range:[0,1]},
      margin:{l:55,r:15,t:45,b:65},legend:{orientation:'h',y:-0.2},paper_bgcolor:'white',
    },{responsive:true,displaylogo:false});
    document.getElementById('peak-note').textContent = `${p.peak} · ${p.true_scope.replaceAll('_',' ')} program · ${targetLabel(p)}` +
      (p.true_program==='opposite_celltypes'?` opens while ${p.partner_celltype} closes`:'') +
      (p.true_program==='constitutive_core'?' · constant true accessibility; zero true temporal rate':'');
    const start = c*6;
    const names = ['truth_velocity','velocity_temporal','velocity'];
    Plotly.react('rates', names.map((name,k)=>({x:d.stages.slice(0,5).map((t,j)=>`${t}→${d.stages[j+1]}`),
      y:d[name][selected].slice(start,start+5), type:'scatter',mode:'lines+markers',
      name:['True interval rate','Temporal estimate','Cis-smoothed estimate'][k],
      line:{color:['#155eaa','#64748b','#c04a32'][k],width:k===0?3:1.6}})),
      {title:{text:`Opening/closing in ${d.celltypes[c]}`,font:{size:16}},
        xaxis:{title:'Interval (hpf)'},yaxis:{title:'Accessibility / hpf',zeroline:true,zerolinecolor:'#94a3b8'},
        margin:{l:60,r:15,t:45,b:80},legend:{orientation:'h',y:-0.35}},
      {responsive:true,displaylogo:false});
  }
  function choosePeak(i) { selected=i; renderCurves(); }
  example.onchange = () => { const p=d.peaks[Number(example.value)]; ct.value=d.celltypes.indexOf(p.primary_celltype); choosePeak(Number(example.value)); renderAtlas(); };
  ct.onchange=()=>{renderAtlas();renderCurves();};
  dimension.onchange=renderAtlas;
  metric.onchange=renderAtlas;
  stage.oninput=renderAtlas;
  all.onchange=renderCurves;
  ct.value=d.celltypes.indexOf(d.peaks[selected].primary_celltype);
  renderAtlas().then(()=>atlas.on('plotly_click',event=>choosePeak(event.points[0].pointNumber)));
  renderCurves();
}

if (typeof module !== 'undefined') module.exports = {latentValue};
if (typeof window !== 'undefined') initExplorer(DATA);

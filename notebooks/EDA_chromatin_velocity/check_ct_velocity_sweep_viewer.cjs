/* Offline parameter explorer: payload-to-H5 audit and all-setting browser QA. */
const fs=require('fs'),path=require('path'),crypto=require('crypto'),child=require('child_process');
const {chromium}=require(process.env.CHROMATIN_PLAYWRIGHT_MODULE || 'playwright');
const out=path.resolve(process.env.CT_PARAMETER_VIEWER_OUT||'notebooks/EDA_chromatin_velocity/outputs/ct_velocity_experiments_v1');
const qa=path.join(out,'parameter_browser_validation');fs.mkdirSync(qa,{recursive:true});
const htmlPath=path.join(out,'parameter_explorer.html');
const source=path.resolve('notebooks/EDA_chromatin_velocity/outputs/ct_velocity_flow_sweep_v1/sweep_vectors.h5');
const py=process.env.CHROMATIN_PYTHON || 'python';
const hash=()=>crypto.createHash('sha256').update(fs.readFileSync(htmlPath)).digest('hex');
const initialHash=hash();
const auditCode=`import sys,json,h5py,numpy as np
from pathlib import Path
html=Path(sys.argv[1]).read_text();d=json.loads(html.rsplit('const D=',1)[1].split(';window.velocityData=D;',1)[0]);checks=[]
def equal(got,want,name):
 a=np.asarray(got,dtype=float);b=np.asarray(want,dtype=float)
 assert a.shape==b.shape,name+' shape'
 assert np.array_equal(a,b,equal_nan=True),name+' mismatch'
 checks.append(name)
with h5py.File(sys.argv[2],'r') as f:
 equal(d['xy'],f['umap2d'][:],'coordinates');equal(d['source_indices'],f['source_indices'][:],'source_indices');assert d['peak_ids']==f['peak_ids'].asstr()[:].tolist();checks.append('peak_ids');equal(d['broad'],np.flatnonzero(f['broad_reference'][:]),'broad_reference')
 for t in range(5):
  it=f[f'interval_{t}'];selection=np.asarray(d['arrow_indices'][str(t)],int)
  assert len(selection)==len(np.unique(selection));assert np.all(it['source_active'][:][selection])
  for key in ['source_active','gradual','intrinsic_net_rate']:equal(d[key][str(t)],it[key][:],f'{t}_{key}')
  for k in [10,30,60,100]:
   for b in [0,25,50]:
    g=it[f'k{k}_b{b:03d}']
    for ctl in ['actual','reversed']+(['permutation_0'] if b==0 else []):
     item=g if ctl=='actual' else g[ctl];key=f'{t}_{k}_{b}_{ctl}';field=d['fields'][key]
     equal(field['vectors'],item['vectors2d'][:][selection],key+'_vectors');equal(field['valid'],item['valid2d'][:][selection],key+'_valid');equal(field['excess'],item['root_excess_delta'][:][selection],key+'_root_excess');assert field['eligible']==int(item['valid2d'][:].sum())
 assert len(d['fields'])==140
 vals=np.concatenate([np.abs(it['intrinsic_net_rate'][:][it['source_active'][:].astype(bool)]) for it in [f[f'interval_{t}'] for t in range(5)]])
 assert d['rate_limit']==float(np.nanquantile(vals,.98))
 assert d['span']==float(np.linalg.norm(np.ptp(f['umap2d'][:],axis=0)))
print(json.dumps(dict(status='passed',exact_array_checks=len(checks),fields=140,n_queries=len(d['xy']),checks=checks,source=str(Path(sys.argv[2]).resolve()),all_values_and_masks_equal=True)))`;
const audited=child.spawnSync(py,['-c',auditCode,htmlPath,source],{encoding:'utf8',maxBuffer:16*1024*1024});
if(audited.status!==0)throw Error('Independent payload/H5 audit failed: '+audited.stderr);
const numerical=JSON.parse(audited.stdout);fs.writeFileSync(path.join(qa,'payload_source_audit.json'),JSON.stringify(numerical,null,2));
(async()=>{
 const browser=await chromium.launch({headless:true,args:['--no-sandbox','--disable-dev-shm-usage','--enable-unsafe-swiftshader','--use-gl=angle','--use-angle=swiftshader']});
 const page=await browser.newPage({viewport:{width:1650,height:1200},deviceScaleFactor:1});
 const errors=[],requests=[],checks=[];page.on('pageerror',e=>errors.push(e.message));page.on('request',r=>{if(/^https?:/.test(r.url()))requests.push(r.url())});
 await page.goto('file://'+htmlPath,{waitUntil:'load',timeout:120000});await page.evaluate(()=>window.viewerReady);
 async function check(label){
  await page.evaluate(()=>queue);
  const result=await page.evaluate(()=>{
   const d=window.velocityData,$=id=>document.getElementById(id),t=$('interval').value,k=$('k').value,b=$('beta').value,ctl=$('control').value;
   const field=d.fields[`${t}_${k}_${b}_${ctl}`],pick=d.arrow_indices[t],plot=$('plot'),selected=[],length=d.span*+$('length').value/1000;
   for(let a=0;a<pick.length;a++){if(!field.valid[a]||($('gradual').checked&&!d.gradual[t][pick[a]]))continue;if(a%100<+$('density').value)selected.push(a)}
   const x=[],y=[];let error=0,finite=true;
   for(const a of selected){const i=pick[a],p=d.xy[i],v=field.vectors[a],n=Math.hypot(...v);if(!(n>0)||!Number.isFinite(n))finite=false;const u=v.map(z=>z/n),q=[p[0]+length*u[0],p[1]+length*u[1]],side=[-u[1],u[0]],back=.3*length,wide=.15*length;for(const pair of [[p,q],[q,[q[0]-back*u[0]+wide*side[0],q[1]-back*u[1]+wide*side[1]]],[q,[q[0]-back*u[0]-wide*side[0],q[1]-back*u[1]-wide*side[1]]]]){x.push(pair[0][0],pair[1][0],null);y.push(pair[0][1],pair[1][1],null)}}
   function compare(got,want){if(got.length!==want.length)return Infinity;let e=0;for(let i=0;i<want.length;i++){if(want[i]===null){if(got[i]!==null)return Infinity}else e=Math.max(e,Math.abs(got[i]-want[i]))}return e}
   const colored=d.source_active[t].map((v,i)=>v&&Number.isFinite(d.intrinsic_net_rate[t][i])?i:-1).filter(i=>i>=0),reference=$('reference').checked?d.broad:[];
   for(let j=0;j<2;j++){const axis=['x','y'][j];error=Math.max(error,compare(plot.data[0][axis],d.xy.map(v=>v[j])),compare(plot.data[1][axis],colored.map(i=>d.xy[i][j])),compare(plot.data[2][axis],reference.map(i=>d.xy[i][j])),compare(plot.data[3][axis],j===0?x:y));}
   error=Math.max(error,compare(plot.data[1].marker.color,colored.map(i=>d.intrinsic_net_rate[t][i])));
   const status=$('status').textContent,render=window.lastRender;
   return {t,k,b,control:ctl,gradual:$('gradual').checked,reference:$('reference').checked,density:+$('density').value,length:+$('length').value,expected_arrows:selected.length,rendered_arrows:plot.data[3].x.length/9,render_metadata:render,eligible:field.eligible,max_render_error:error,all_arrow_vectors_finite:finite,color_scale:plot.data[1].marker.colorscale,color_reversed:plot.data[1].marker.reversescale,status,range_x:plot.layout.xaxis.range,range_y:plot.layout.yaxis.range};
  });
  if(result.max_render_error>1e-10||!result.all_arrow_vectors_finite||result.expected_arrows!==result.rendered_arrows||result.render_metadata.n!==result.expected_arrows||!result.status.includes(result.eligible.toLocaleString()+' eligible'))throw Error(label+': '+JSON.stringify(result));
  const cs=result.color_scale;if(!Array.isArray(cs)||cs[0][0]!==0||cs[0][1].toLowerCase()!=='#2166ac'||cs[cs.length-1][0]!==1||cs[cs.length-1][1].toLowerCase()!=='#b2182b'||result.color_reversed)throw Error('Signed color orientation mismatch: '+JSON.stringify(cs));
  checks.push({label,...result});console.log('PASS '+label);return result;
 }
 await check('Default actual rates, 10→12 hpf, k30, beta0');await page.screenshot({path:path.join(qa,'default_parameter_view.png'),fullPage:true});
 for(let t=0;t<5;t++){
  await page.selectOption('#interval',String(t));
  for(const k of['10','30','60','100']){
   await page.selectOption('#k',k);
   for(const b of['0','25','50']){
    await page.selectOption('#beta',b);
    for(const ctl of['actual','reversed',...(b==='0'?['permutation_0']:[])]){await page.selectOption('#control',ctl);await check(`interval${t} k${k} beta${b} ${ctl}`)}
   }
  }
 }
 await page.selectOption('#interval','3');await page.selectOption('#k','30');await page.selectOption('#beta','0');await page.selectOption('#control','permutation_0');await check('Control screenshot');await page.screenshot({path:path.join(qa,'permuted_control_16_19.png'),fullPage:true});
 await page.selectOption('#control','actual');const all=await check('All supported arrows before gradual filter');await page.check('#gradual');const gradual=await check('Sustained-trend arrow subset');if(gradual.expected_arrows>all.expected_arrows)throw Error('Gradual filter increased arrow count');await page.uncheck('#gradual');
 await page.uncheck('#reference');await check('Reference hidden');await page.check('#reference');await check('Reference restored');
 const density=[];for(const v of['10','50','100']){await page.locator('#density').evaluate((el,v)=>{el.value=v;el.dispatchEvent(new Event('change',{bubbles:true}))},v);density.push((await check('Density '+v)).expected_arrows)}if(density.some((v,i)=>i>0&&v<density[i-1]))throw Error('Density control non-monotone');
 for(const v of['3','25','12']){await page.locator('#length').evaluate((el,v)=>{el.value=v;el.dispatchEvent(new Event('change',{bubbles:true}))},v);await check('Arrow length '+v)}
 await page.selectOption('#control','permutation_0');await page.selectOption('#beta','25');const resetControl=await check('Unavailable permuted beta resets to actual');if(resetControl.control!=='actual')throw Error('Invalid permutation/beta combination not reset');
 await page.evaluate(()=>Plotly.relayout('plot',{'xaxis.range':[0,1],'yaxis.range':[0,1]}));await page.click('#reset');const reset=await check('Reset view after manual zoom');if(reset.range_x&&reset.range_x[0]===0&&reset.range_x[1]===1)throw Error('Reset did not clear manual zoom');
 await page.evaluate(()=>{for(const [id,v] of[['interval','0'],['beta','50'],['control','reversed'],['k','100'],['interval','4'],['beta','0'],['control','actual']]){const el=document.getElementById(id);el.value=v;el.dispatchEvent(new Event('change',{bubbles:true}))}});await check('Rapid changes settle consistently');
 await page.selectOption('#interval','0');await page.selectOption('#k','30');await page.selectOption('#beta','0');await page.selectOption('#control','actual');await page.locator('#density').evaluate(el=>{el.value='70';el.dispatchEvent(new Event('change',{bubbles:true}))});await check('Restored default settings');
 if(hash()!==initialHash)throw Error('HTML changed during browser validation');
 const report={passed:errors.length===0&&requests.length===0,html_path:htmlPath,html_sha256:initialHash,html_unchanged:true,completed_utc:new Date().toISOString(),n_checks:checks.length,n_complete_parameter_fields:140,numerical_source_audit:numerical,checks,page_errors:errors,external_requests:requests};fs.writeFileSync(path.join(qa,'browser_validation.json'),JSON.stringify(report,null,2));console.log(JSON.stringify({passed:report.passed,n_checks:checks.length,page_errors:errors,external_requests:requests}));await browser.close();if(!report.passed)process.exitCode=1;
})().catch(e=>{console.error(e);process.exit(1)});

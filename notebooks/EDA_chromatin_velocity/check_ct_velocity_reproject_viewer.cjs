const fs=require('fs'),path=require('path'),crypto=require('crypto');
const {chromium}=require(process.env.CHROMATIN_PLAYWRIGHT_MODULE || 'playwright');
const out=path.resolve(process.env.CT_VELOCITY_VIEWER_OUT||'notebooks/EDA_chromatin_velocity/outputs/ct_velocity_reprojection_v1');
const qa=path.join(out,'viewer_browser_validation');fs.mkdirSync(qa,{recursive:true});
const htmlPath=path.join(out,'ct_velocity_explorer.html');const sourceHash=()=>crypto.createHash('sha256').update(fs.readFileSync(htmlPath)).digest('hex');const initialHash=sourceHash();
(async()=>{
 const browser=await chromium.launch({headless:true,args:['--no-sandbox','--disable-dev-shm-usage','--enable-unsafe-swiftshader','--use-gl=angle','--use-angle=swiftshader']});
 const page=await browser.newPage({viewport:{width:1650,height:1180},deviceScaleFactor:1});
 const errors=[],requests=[],checks=[];page.on('pageerror',e=>errors.push(e.message));page.on('request',r=>{if(/^https?:/.test(r.url()))requests.push(r.url())});
 await page.goto('file://'+path.join(out,'ct_velocity_explorer.html'),{waitUntil:'load',timeout:120000});
 await page.waitForFunction(()=>window.ctVelocityReady&&!window.rendering,null,{timeout:120000});
 async function check(label){
  await page.waitForFunction(()=>!window.rendering,null,{timeout:120000});
  const result=await page.evaluate(()=>{
   const d=window.ctVelocityDebug,s=d.getState(),ar=d.getArrowGeometry(s),map=document.getElementById('map'),curve=document.getElementById('rate-chart'),xy=d.MAPS[String(s.d)],ci=d.getSelected(),color=d.getColor(s);let coordError=0,arrowError=0,curveError=0,colorError=0;
   for(let i=0;i<d.DATA.peak_ids.length;i++){for(let k=0;k<s.d;k++)coordError=Math.max(coordError,Math.abs(map.data[0][['x','y','z'][k]][i]-xy[i*s.d+k]));if(Number.isFinite(color.values[i]))colorError=Math.max(colorError,Math.abs(map.data[0].marker.color[i]-color.values[i]));else if(map.data[0].marker.color[i]!==null)colorError=Infinity}
   const stride=s.d===2?9:15;for(let i=0;i<ar.count;i++)for(let k=0;k<s.d;k++){const axis=['x','y','z'][k];arrowError=Math.max(arrowError,Math.abs(map.data[1][axis][stride*i]-ar.a.origins[i*s.d+k]),Math.abs(map.data[1][axis][stride*i+1]-ar.ends[i][k]));}
   for(let t=0;t<5;t++){const expected=d.METHODS[s.m][t].net_rate[ci];if(Number.isFinite(expected))curveError=Math.max(curveError,Math.abs(curve.data[0].y[t]-expected));else if(curve.data[0].y[t]!==null)curveError=Infinity}
   return{method:s.m,interval:s.t,dimension:s.d,color:s.color,arrow_mode:s.mode,arrows_shown:ar.count,background_peaks:map.data[0].x.length,coordinate_max_error:coordError,arrow_max_error:arrowError,curve_max_error:curveError,color_max_error:colorError,selected_peak:d.DATA.peak_ids[ci],error:document.getElementById('error').textContent};
  });
  if(result.coordinate_max_error!==0||result.arrow_max_error!==0||result.curve_max_error!==0||result.color_max_error!==0||result.error)throw Error(label+': '+JSON.stringify(result));
  checks.push({label,...result});console.log('PASS '+label);
 }
 await check('Default updated 19→24hpf 2D');
 await page.evaluate(()=>{for(const [id,value] of[['interval','0'],['dimension','3'],['method','reused'],['interval','4'],['method','updated'],['dimension','2']]){const el=document.getElementById(id);el.value=value;el.dispatchEvent(new Event('change',{bubbles:true}))}});await check('Rapid control changes settle to consistent state');
 await page.screenshot({path:path.join(qa,'updated_19_24_2d.png'),fullPage:true});
 for(const method of['updated','reused']){
  await page.selectOption('#method',method);
  for(const dim of['2','3']){
   await page.selectOption('#dimension',dim);
   for(let interval=0;interval<5;interval++){await page.selectOption('#interval',String(interval));await check(`${method} ${interval} ${dim}D`)}
  }
 }
 await page.selectOption('#method','updated');await page.selectOption('#interval','4');
 await page.screenshot({path:path.join(qa,'updated_19_24_3d.png'),fullPage:true});
 await page.selectOption('#dimension','2');
 for(const color of['mean','tau','rms','rate']){await page.selectOption('#color',color);await check('Color '+color)}
 await page.selectOption('#arrow-mode','weighted');await check('Coherence-weighted arrows');
 await page.locator('#scale').evaluate(el=>{el.value='2';el.dispatchEvent(new Event('change',{bubbles:true}))});await check('Twofold display scale');
 await page.locator('#density').evaluate(el=>{el.value='0';el.dispatchEvent(new Event('change',{bubbles:true}))});await check('Zero arrows');
 await page.locator('#density').evaluate(el=>{el.value='600';el.dispatchEvent(new Event('change',{bubbles:true}))});await check('All 600 candidate arrows');
 const id=await page.evaluate(()=>window.ctVelocityDebug.DATA.peak_ids[123]);await page.fill('#search',id);await page.click('#find');await check('Exact peak search');
 if(await page.evaluate(()=>window.ctVelocityDebug.DATA.peak_ids[window.ctVelocityDebug.getSelected()])!==id)throw Error('Exact peak search failed');
 await page.fill('#search','slc4a1a');await page.click('#find');await check('Nearest gene search');
 if(!(await page.evaluate(()=>window.ctVelocityDebug.DATA.genes[window.ctVelocityDebug.getSelected()])).toLowerCase().includes('slc4a1a'))throw Error('Gene search failed');
 const missingId=await page.evaluate(()=>{const d=window.ctVelocityDebug;return d.DATA.peak_ids[Array.from(d.METHODS.updated[0].net_rate).findIndex(x=>!Number.isFinite(x))]});await page.fill('#search',missingId);await page.click('#find');await check('Unavailable rates preserved as gaps');if(!(await page.evaluate(()=>document.getElementById('rate-chart').data[0].y.every(x=>x===null))))throw Error('Missing rates were not preserved');await page.fill('#search','slc4a1a');await page.click('#find');
 await page.selectOption('#arrow-mode','direction');await page.selectOption('#color','mean');
 await page.locator('#scale').evaluate(el=>{el.value='1';el.dispatchEvent(new Event('change',{bubbles:true}))});
 await page.locator('#density').evaluate(el=>{el.value='350';el.dispatchEvent(new Event('change',{bubbles:true}))});await check('Final restored display');
 await page.screenshot({path:path.join(qa,'selected_gene_mean_accessibility.png'),fullPage:true});
 if(sourceHash()!==initialHash)throw Error('HTML changed during browser validation');const result={passed:errors.length===0&&requests.length===0,html_path:htmlPath,html_sha256:initialHash,html_unchanged_during_validation:true,completed_utc:new Date().toISOString(),checks,page_errors:errors,external_requests:requests};
 fs.writeFileSync(path.join(qa,'browser_validation.json'),JSON.stringify(result,null,2));
 console.log(JSON.stringify({passed:result.passed,n_checks:checks.length,page_errors:errors,external_requests:requests},null,2));
 await browser.close();if(!result.passed)process.exitCode=1;
})().catch(e=>{console.error(e);process.exit(1)});

"""Portable report/figure/code bundle for the authorized chromatin email thread."""
import os
from pathlib import Path,PurePosixPath
from html.parser import HTMLParser
import json,hashlib,zipfile,posixpath,re
import pandas as pd
HERE=Path(__file__).resolve().parent;ROOT=HERE/'outputs';OUT=ROOT/'ct_velocity_experiments_v1';DEST=OUT/'email_package/deliverables'
FOLDERS=['ct_velocity_experiments_v1','ct_velocity_flow_sweep_v1','ct_cis_forecast_v1','ct_cis_forecast_lagged_v1']
class Links(HTMLParser):
 def __init__(self):super().__init__();self.links=[]
 def handle_starttag(self,tag,attrs):self.links.extend(v for k,v in attrs if k in ['src','href'])
def dump(p,d):p.write_text(json.dumps(d,indent=2)+'\n')
def main():
 DEST.mkdir(parents=True,exist_ok=True)
 for p in [OUT/'report.html',OUT/'parameter_explorer.html',OUT/'parameter_viewer_validation.json',OUT/'parameter_browser_validation/browser_validation.json',OUT/'forecast_independent_audit.json',OUT/'lagged_forecast_independent_audit.json',OUT/'sweep_summary_and_distance_audit.json']:
  assert p.exists(),p
 archive=DEST/'chromatin_velocity_parameter_and_cis_experiments.zip';rewrites=[]
 with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
  for folder in FOLDERS:
   for p in sorted((ROOT/folder).iterdir()):
    if p.suffix not in ['.png','.csv','.json','.html'] or not p.is_file():continue
    name=f'{folder}/{p.name}'
    if p.suffix=='.html' and p.name!='parameter_explorer.html':
     content=p.read_text()
     def replace_link(match):
      href,label=match.groups()
      if href.endswith(('.h5','.h5ad','.npy','.npz')):
       rewrites.append(dict(report=name,omitted_array=href,workspace_path=str((p.parent/href).resolve())))
       return '<code>'+str((p.parent/href).resolve())+'</code> (large array on shared workspace)'
      return match.group(0)
     content=re.sub(r'<a href="([^"]+)">([^<]*)</a>',replace_link,content);z.writestr(name,content)
    else:z.write(p,name)
  for p in sorted((OUT/'parameter_browser_validation').glob('*.json')):z.write(p,'ct_velocity_experiments_v1/parameter_browser_validation/'+p.name)
  code=['ct_velocity_flow_sweep.py','ct_velocity_display_sweep.py','ct_velocity_sweep_viewer.py','ct_velocity_sweep_viewer_template.html','check_ct_velocity_sweep_viewer.cjs','ct_velocity_experiments_report.py','ct_velocity_experiments_delivery.py','ct_velocity_experiment_audit.py','ct_cis_forecast.py','ct_cis_forecast_lagged.py','ct_cis_forecast_report.py','ct_velocity_reproject.py','real_trajectory_velocity.py','real_specialization_benchmark.py','real_specialization_report.py']
  for name in code:z.write(HERE/name,'code/notebooks/EDA_chromatin_velocity/'+name)
  for p in sorted((HERE.parents[1]/'scripts/chrom_velo').glob('*.py')):z.write(p,'code/scripts/chrom_velo/'+p.name)
  z.writestr('OPEN_FIRST.txt','Extract the ZIP and open ct_velocity_experiments_v1/report.html. The offline parameter_explorer.html lets you compare neighbor count, rate smoothing, intervals and negative controls. Each point is a peak. All PNG figures, CSV tables, validation records and implementation code are included. Large prediction H5 files, graph arrays, display grids and PDF figures remain in the shared workspace at the paths in the reports. Code expects those original data/analysis paths; this archive is not a stand-alone raw-data bundle.\n')
 with zipfile.ZipFile(archive) as z:
  assert z.testzip() is None;names=set(z.namelist());assert len(names)==len(z.namelist());broken=[]
  for name in names:
   if name.endswith('.html') and not name.startswith('code/'):
    links=Links();links.feed(z.read(name).decode())
    for link in links.links:
     if ':' in link or link.startswith('#'):continue
     dest=posixpath.normpath(str(PurePosixPath(name).parent/link))
     if dest not in names:broken.append((name,link))
  assert not broken,broken
 checks=dict(status='passed',zip_crc_verified=True,relative_html_links_resolve=True,n_files=len(names),png_figures=sum(n.endswith('.png') for n in names),rewritten_large_array_links=rewrites,archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest());dump(DEST/'package_validation.json',checks)
 candidates=[OUT/'04_projection_parameter_arrows_10_12.png',ROOT/'ct_cis_forecast_v1/01_cis_future_change.png'];selected=[];size=archive.stat().st_size
 for p in candidates:
  if size+p.stat().st_size<18_000_000:selected.append(p);size+=p.stat().st_size
 assert size<18_000_000,'Email package requires splitting'
 attachments=[dict(path=str(p.resolve()),filename=p.name,mime_type='application/zip' if p.suffix=='.zip' else 'image/png',bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in selected+[archive]]
 manifest=dict(to=os.environ.get('CHROMATIN_REPORT_RECIPIENT', ''),subject='Re: Real chromatin velocity: peak UMAPs, interactive atlas, and depth-matching interpretation',attachments=attachments,binary_bytes=size);dump(DEST/'delivery_manifest.json',manifest)
 body='''<p>Hello,</p><p><strong>The parameter experiments and Cicero future-state tests are finished.</strong> All figures, tables, reports, code, and an offline interactive parameter explorer are attached in the ZIP.</p><h3>1. Do parameter changes reveal flow from or toward the root?</h3><p>I tested projection k=10,30,60,100 and rate-neighbor blends of 0,0.25,0.5 across all five intervals, using 20,000 fixed random query peaks with targets from the full 640,830-peak atlas. Separate stream-display tests varied neighbor count and smoothing width.</p><ul><li>Local arrows are quite stable: among arrows eligible under both settings, median direction cosine stays ≥0.973 and fewer than 3% turn by more than 90°.</li><li>The mean raw graph-distance tendency is <strong>toward the broad-accessibility reference</strong> under every tested setting. The unguided neighbor graph also points inward, so much of this is geometric.</li><li>After subtracting that graph tendency, later fields are less inward. At the default setting, 12–14 and 14–16 hpf exceed all three matched shuffles on a common-projectable cohort. Late-stage contrasts weaken substantially when the same peaks must be projectable under every setting. This does not establish a global outward biological stream.</li></ul><p>The reference consists of 20,905 broadly accessible peaks spread across the atlas; I used fixed graph-hop distance to this set, rather than one UMAP center. The gradual subset requires at least three supported intervals in a monotonic celltype trajectory. No parameter was chosen because it looked more root-to-branch.</p><h3>2. Can Cicero-weighted accessibility predict the future?</h3><p>I tested weighted current neighbor accessibility minus own accessibility, added to current own state and previous own slope. Models train only on earlier transitions and other chromosomes, and predict next-stage changes in an independent cell half. I also tested recent neighbor slopes, matched-distance neighbors, feature shuffles, persistence, and direct neighbor averaging.</p><p><strong>No predictive gain was demonstrated.</strong> At cutoff 0.25, current Cicero information increased squared-change error by about 0.13% versus own history; adding recent neighbor changes increased it by about 1.40% in the point estimate (its chromosome-bootstrap interval crosses zero). Own-history prediction itself was about 7.1% worse than persistence overall. The planted positive control successfully recovered a strong true neighbor-leading signal.</p><p>This scopes the conclusion to these models and the available graph. At cutoff 0.25 only 4,525 peaks are connected (0.706% of the atlas). The graph was learned from all stages/cells, so validation is conditional on that graph; cell halves are not biological replicates. Co-accessibility may still help denoising or module summaries, but synchronous association is not automatically a future-state model.</p><h3>My interpretation</h3><p>The most defensible result remains the fitted interval-specific opening/closing trajectories. The local arrow patterns survive reasonable parameter changes, but rootward geometric bias and projection coverage limit a global-flow claim. A stronger predictive follow-up would need training-only lineage-specific graphs, broader connected-peak coverage, and independent biological validation.</p><p>Extract the ZIP and open <code>ct_velocity_experiments_v1/report.html</code> or its <code>parameter_explorer.html</code>. The explorer switches interval, k, rate blend, gradual subset, and reversed/permuted controls. All checks passed: projection reconstruction, fixed graph distances, forecast coefficients/predictions, holdout partitions, synthetic controls, and browser operation.</p><p>Full workspace report:<br><code>'''+str(OUT/'report.html')+'''</code><br>Large prediction arrays and PDFs remain in the sibling analysis folders listed in that report.</p>'''
 (DEST/'email_body.html').write_text(body);print(json.dumps(manifest,indent=2))
if __name__=='__main__':main()

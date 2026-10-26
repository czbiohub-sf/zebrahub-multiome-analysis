"""Package finalized CT peak-UMAP velocity views for the authorized email thread."""
import os
from pathlib import Path,PurePosixPath
from html.parser import HTMLParser
import hashlib,html,json,zipfile
import pandas as pd
from ct_velocity_reproject import HERE,OUT,dump
SUBJECT='Re: Real chromatin velocity: peak UMAPs, interactive atlas, and depth-matching interpretation'
class Links(HTMLParser):
 def __init__(self):super().__init__();self.links=[]
 def handle_starttag(self,tag,attrs):self.links.extend(v for k,v in attrs if k in ['src','href'])

def main():
 folder=OUT/'email_package/deliverables';folder.mkdir(parents=True,exist_ok=True)
 required=['report.html','ct_velocity_explorer.html','new_graph_projection_validation.json','reused_validation.json','export_validation.json','figure_manifest.json','viewer_payload_validation.json']
 for name in required:assert (OUT/name).exists(),name
 # Package figures, portable browser viewer, tables and implementation; large arrays remain in workspace.
 archive=folder/'ct_umap_chromatin_velocity_results.zip'
 files=sorted(p for p in set(list(OUT.glob('*.png'))+list(OUT.glob('*.csv'))+list(OUT.glob('*.json'))+[OUT/'report.html',OUT/'ct_velocity_explorer.html']) if 'pre_exact' not in p.name)
 with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
  for p in files:z.write(p,p.name)
  for p in sorted(OUT.glob('viewer_browser_validation/*.json')):z.write(p,'viewer_browser_validation/'+p.name)
  code=['ct_velocity_reproject.py','ct_velocity_reproject_core.py','ct_velocity_reproject_figures.py','ct_velocity_reproject_viewer.py','ct_velocity_reproject_export.py','ct_velocity_reproject_report.py','ct_velocity_reproject_delivery.py','real_trajectory_velocity.py','real_specialization_benchmark.py','real_specialization_report.py','check_ct_velocity_reproject_viewer.cjs']
  for name in code:z.write(HERE/name,'code/notebooks/EDA_chromatin_velocity/'+name)
  for p in HERE.glob('ct_velocity*template.html'):z.write(p,'code/notebooks/EDA_chromatin_velocity/'+p.name)
  for name in ['trajectory_projection.py','trajectory_modes.py','trajectory_classification.py','streamline_field.py','streamline_paths.py','real_specialization.py','specialization_flow.py','real_sparse_projection.py','temporal_velocity.py']:
   z.write(HERE.parents[1]/'scripts/chrom_velo'/name,'code/scripts/chrom_velo/'+name)
  z.writestr('OPEN_FIRST.txt','Extract this ZIP and open report.html, or ct_velocity_explorer.html for the offline interactive 2D/3D view. All five intervals, primary updated-neighbor arrows, and coordinate-only controls are included. Each point is a genomic peak. The original fitted CPM/hour values were reused unchanged. Large H5/AnnData arrays and PDF versions remain in the shared workspace.\n')
 with zipfile.ZipFile(archive) as z:
  assert z.testzip() is None;names=set(z.namelist());assert len(names)==len(z.namelist());broken=[]
  for name in ['report.html','ct_velocity_explorer.html']:
   parser=Links();parser.feed(z.read(name).decode())
   for link in parser.links:
    if ':' in link or link.startswith('#'):continue
    if str(PurePosixPath(name).parent/link) not in names:broken.append((name,link))
  assert not broken,broken
 dump(folder/'package_validation.json',dict(status='passed',zip_crc_verified=True,relative_html_links_resolve=True,n_entries=len(names),n_pngs=sum(n.endswith('.png') for n in names),archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest()))
 # Attach the two most useful PNGs directly when size permits; all PNGs are inside the ZIP.
 selected=[OUT/'01_ct_umap_velocity_arrows.png',OUT/'03_ct_umap_velocity_streams.png']
 if sum(p.stat().st_size for p in selected)+archive.stat().st_size>=18_000_000:selected=selected[:1]
 assert sum(p.stat().st_size for p in selected)+archive.stat().st_size<18_000_000,'Split email package required'
 attachments=[dict(path=str(p.resolve()),filename=p.name,mime_type='application/zip' if p.suffix=='.zip' else 'image/png',bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in selected+[archive]]
 manifest=dict(to=os.environ.get('CHROMATIN_REPORT_RECIPIENT', ''),subject=SUBJECT,attachments=attachments,binary_bytes=sum(p['bytes'] for p in attachments));dump(folder/'delivery_manifest.json',manifest)
 cov=pd.read_csv(OUT/'projection_coverage.csv');q=cov.query('method=="updated" and embedding=="umap2d"');nums=', '.join(f'{int(r.start_hpf)}→{int(r.end_hpf)}: {int(r.projected):,}' for r in q.itertuples())
 agreement=pd.read_csv(OUT/'projection_method_agreement.csv').query('embedding=="umap2d"');rev=agreement.fraction_reversed
 geom=pd.read_csv(OUT/'edge_locality_and_geometry.csv').query('method=="updated"').median_baseline_direction_cosine
 geometry_text=f'Median cosine alignment with the unfiltered graph direction is {geom.min():.3f}–{geom.max():.3f}, so neighborhood geometry contributes strongly to these arrows. This diagnostic does not by itself establish biological flow.'
 body='''<p>Hello,</p><p><strong>The chromatin velocities are now reprojected onto the new celltype × timepoint UMAP in2D and3D.</strong> I reused the previous six-stage fitted accessibility curves and CPM/hour rates, since the measured data and intervals have not changed.</p><ul><li><strong>Primary view:</strong> rebuilt30-neighbor candidates among all640,830 peaks in the new50-PC log-accessibility space, then recomputed transition probabilities and2D/3D directions using the same linear CPM states/rates and source gates.</li><li><strong>Coordinate-only control:</strong> retained the previous transition weights and recomputed their directions on the new map. This isolates changing the coordinates from changing neighborhoods.</li><li><strong>Five separate interval fields:</strong>10→12,12→14,14→16,16→19 and19→24hpf. No root direction, entropy weighting, or extra cis smoothing was imposed.</li></ul><p><strong>Plots:</strong> primary2D arrows, original/new-map controls, locally supported streamplots,3D arrows, and separate signed-rate/magnitude maps. The interactive HTML switches intervals, methods, dimensions and display settings.</p><p>Eligible2D arrows before display subsampling:<br>'''+nums+'''</p><p><strong>How to interpret them:</strong> red/blue shows mean relative opening/closing in CPM/hour. Arrows show local profile correspondence between fixed genomic peaks; inward/outward arrows do not directly encode closing/opening. Arrow lengths are normalized for display. The two projection methods differ by more than 90° for '''+f'{100*rev.min():.1f}–{100*rev.max():.1f}%'+''' of peaks with eligible arrows under both methods, depending on interval, so the neighborhood definition matters.</p><p>'''+geometry_text+'''</p><p>The earlier independent-cell and count-noise findings still apply: these figures provide the requested new projection, without establishing a universal root-to-branch biological stream. Source gates are conditional fit/count support, not biological confidence.</p><p><strong>Attachments:</strong> key PNGs and a ZIP containing all five PNG figures, the offline HTML explorer, report, tables, validation records and code. Extract the ZIP and open <code>report.html</code> or <code>ct_velocity_explorer.html</code>.</p><p>Full AnnData (unchanged high-dimensional biological layers; new projected fields):<br><code>'''+html.escape(str(OUT/'chromatin_velocity_ct_umap.h5ad'))+'''</code><br>Report:<br><code>'''+html.escape(str(OUT/'report.html'))+'''</code></p><p>Peak alignment, exact-rate preservation, sparse transition calculations, neighbor recall, browser behavior and AnnData export were checked. Original fits and source atlases remain unchanged.</p>'''
 for a,b in [('in2D','in 2D'),('and3D','and 3D'),('rebuilt30','rebuilt 30'),('all640','all 640'),('new50','new 50'),('and2D','and 2D'),('fields:</strong>10','fields:</strong> 10'),('→12,12','→12, 12'),('→14,14','→14, 14'),('→16,16','→16, 16'),('and19','and 19'),('24hpf','24 hpf'),('primary2D','primary 2D'),('streamplots,3D','streamplots, 3D'),('Eligible2D','Eligible 2D')]:body=body.replace(a,b)
 (folder/'email_body.html').write_text(body);print(json.dumps(manifest,indent=2))

if __name__=='__main__':main()

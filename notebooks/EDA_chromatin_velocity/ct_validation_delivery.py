"""Build portable, reviewable delivery packages for the CT atlas and flow tests."""
import os
from pathlib import Path
import argparse, hashlib, html, json, shutil, zipfile
HERE=Path(__file__).resolve().parent
OUT=HERE/'outputs/ct_validation_delivery_v1'
SUBJECT='Re: Real chromatin velocity: peak UMAPs, interactive atlas, and depth-matching interpretation'
STYLE='<style>body{font:16px system-ui,sans-serif;max-width:1150px;margin:32px auto;padding:0 20px;line-height:1.55;color:#233044}img{max-width:100%}table{border-collapse:collapse}th,td{border:1px solid #ddd;padding:8px;text-align:left}pre{white-space:pre-wrap;background:#f3f5f7;padding:16px}</style>'

def manifest(folder, attachments):
 records=[]
 for p,mime in attachments:
  records.append(dict(path=str(p),filename=p.name,mime_type=mime,bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
 result=dict(to=os.environ.get('CHROMATIN_REPORT_RECIPIENT', ''),subject=SUBJECT,attachments=records,binary_bytes=sum(x['bytes'] for x in records))
 assert result['binary_bytes']<18_000_000,result['binary_bytes']
 (folder/'delivery_manifest.json').write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps(result,indent=2))

def atlas():
 src=HERE/'outputs/ct_peak_atlas_v1';folder=OUT/'atlas_milestone';folder.mkdir(parents=True,exist_ok=True)
 figures=[src/f for f in ['01_fixed_peak_atlas_comparison.png','02_ct_peak_atlas_3d.png','03_embedding_diagnostics.png']]
 files=[src/'ct_peak_atlas_explorer.html',src/'report.html']+list(src.glob('*.csv'))+list(src.glob('*.json'))
 files=[p for p in files if not p.name.startswith('full_')]
 archive=folder/'ct_peak_atlas_explorer_and_analysis.zip'
 with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
  for p in files:z.write(p,p.name)
  for p in sorted((src/'coordinates').glob('*.json')):z.write(p,'coordinates/'+p.name)
  for p in sorted((src/'browser_validation').glob('*.json')):z.write(p,'browser_validation/'+p.name)
  for name in ['ct_peak_atlas.py','ct_peak_atlas_viewer.py','ct_peak_atlas_viewer_template.html','ct_peak_atlas_report.py','validate_ct_peak_atlas.py','ct_atlas_independent_check.py','check_ct_peak_atlas_viewer.cjs']:
   z.write(HERE/name,'code/'+name)
  z.writestr('OPEN_FIRST.txt','Save all three PNG email attachments into the same folder where you extract this ZIP. Open report.html for the analysis and ct_peak_atlas_explorer.html for the offline interactive atlas. Each dot is one peak, not a cell. The explorer displays 8,038 peaks from the 25,056-peak comparison; training used 20,000 peaks. Full 640,830-peak frozen-model projection is a separate milestone. No network connection is required for the explorer.\n')
 body='''<p>Hello,</p><p><strong>Milestone 1: the celltype × timepoint peak atlas is ready.</strong> Your suspected mismatch was real: the celltype and Leiden pseudobulk matrices differ, but their saved PCA and UMAP coordinates were identical. I rebuilt the geometry from the actual celltype accessibility matrix.</p><ul><li><strong>Each point is one genomic peak.</strong> Features are the 190 observed celltype × stage combinations. The two absent combinations remain absent.</li><li>Compared eight settings on the same 20,000 training peaks and 5,000 held-out peaks (+56 marker peaks): matched celltype/Leiden log1p(CPM), linear CPM, excluding columns with fewer than 20 cells, broader UMAP neighborhoods, a second seed, all PCs, and 3D.</li><li><strong>Log scaling changes the layout more than aggregation alone.</strong> Celltype and Leiden log maps both form a broad curved continuum with fine fingers. Removing low-cell columns simplifies several fingers, but also removes real rare-celltype features, so simpler does not necessarily mean biologically better.</li><li>The 3D log map preserves local and global relationships somewhat better than 2D. The axes are arbitrary and the map has no imposed root or temporal direction.</li></ul><p><strong>How to view:</strong> save the three PNG attachments beside the extracted ZIP, then open <code>report.html</code> or <code>ct_peak_atlas_explorer.html</code>. The offline explorer contains 8,038 searchable peaks, alternative maps, CT/stage coloring, and measured accessibility curves with missing/low-cell flags.</p><p><strong>Validation:</strong> all eight model arrays independently checked against source counts and transformations; 21 browser checks passed with zero external requests or page errors. The source objects were left unchanged.</p><p>The projection of all 640,830 peaks through the frozen landmark-trained models is finishing separately. I will follow with that milestone and the independent-cell flow/noise and cis-smoothing results. The initial flow tests favor using log accessibility for geometry while keeping biological changes in CPM/hour.</p><p>Analysis folder:<br><code>'''+html.escape(str(src))+'''</code></p>'''
 (folder/'email_body.html').write_text(body)
 manifest(folder,[(p,'image/png') for p in figures]+[(archive,'application/zip')])

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['atlas']);a=p.parse_args();globals()[a.stage]()

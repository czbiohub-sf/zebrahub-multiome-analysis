"""Package checked activation figures and an offline viewer for the existing thread."""
from pathlib import Path
import hashlib,json,re,zipfile,sys,platform
import numpy,scipy,pandas,h5py,matplotlib
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
OUT=HERE/'outputs/activation_progression_v1'

def entry(p,mime):
 return dict(path=str(p.resolve()),filename=p.name,mime_type=mime,bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())

def main():
 p=OUT/'email_package';p.mkdir(exist_ok=True)
 versions={name:module.__version__ for name,module in [('numpy',numpy),('scipy',scipy),('pandas',pandas),('h5py',h5py),('matplotlib',matplotlib)]}
 (OUT/'software_versions.json').write_text(json.dumps(dict(python=sys.version,executable=sys.executable,platform=platform.platform(),packages=versions),indent=2)+'\n')
 # Current sources plus local imported helpers, keeping repository hierarchy.
 sources=sorted(set(HERE.glob('*.py')) | set((ROOT/'scripts/chrom_velo').glob('*.py')))
 (OUT/'source_manifest.json').write_text(json.dumps([dict(path=str(f.relative_to(ROOT)),sha256=hashlib.sha256(f.read_bytes()).hexdigest()) for f in sources],indent=2)+'\n')
 note='''ACTIVATION PROGRESSION: REVIEW PACKAGE
Open analysis_summary.html for figures and interpretation.
Unzip the separate activation_peak_explorer.zip into the same directory to enable its viewer link. The viewer displays a fixed 12,000-peak subset; numerical analyses use all 640,830 atlas peaks.
The 32 individual celltype PNG/PDF sheets and large H5/NPZ arrays are saved in the workspace and omitted from this email-sized ZIP.

REPRODUCIBILITY
Use the existing repository and prior outputs/trajectory_velocity_v1/trajectory_fits.h5, plus the original atlas paths recorded in geometry_summary.json. Source files retain repository-relative layout under source/. These inputs are not included in the ZIP.
From the repository root with the single-cell-base environment:
  python notebooks/EDA_chromatin_velocity/prepare_activation_geometry.py
  python notebooks/EDA_chromatin_velocity/prepare_activation_geometry.py --core-reference-only
  python notebooks/EDA_chromatin_velocity/activation_progression.py --stage all
  python notebooks/EDA_chromatin_velocity/activation_report.py --geometry notebooks/EDA_chromatin_velocity/outputs/activation_progression_v1/geometry.h5
  python notebooks/EDA_chromatin_velocity/activation_report.py --summaries-only
  python notebooks/EDA_chromatin_velocity/activation_report.py --global-only
  python notebooks/EDA_chromatin_velocity/regional_activation_order.py
  python notebooks/EDA_chromatin_velocity/finish_activation_analysis.py --stage all
Validation records and software_versions.json document the executed checks/environment.

FULL OUTPUT DIRECTORY
'''+str(OUT)+'\n'
 (OUT/'START_HERE.txt').write_text(note)
 report=(OUT/'analysis_summary.html').read_text()
 report=re.sub(r'<h2>All celltype sheets</h2><ul>.*?</ul>', '<h2>Individual celltype sheets</h2><p>All 32 sheets are available in the shared workspace under celltypes/. They are omitted from this email-sized ZIP.</p>',report,flags=re.S)
 review=p/'activation_analysis_review.zip'
 files=sorted([f for f in OUT.iterdir() if f.suffix in ['.png','.json','.csv','.txt']])
 files+=sorted((OUT/'browser_validation').glob('*.json'))
 with zipfile.ZipFile(review,'w',zipfile.ZIP_DEFLATED,compresslevel=8) as z:
  for f in files:z.write(f,str(f.relative_to(OUT)))
  z.writestr('analysis_summary.html',report)
  for f in sources:z.write(f,'source/'+str(f.relative_to(ROOT)))
 with zipfile.ZipFile(review) as z:assert z.testzip() is None
 viewer=p/'activation_peak_explorer.zip'
 with zipfile.ZipFile(viewer,'w',zipfile.ZIP_DEFLATED,compresslevel=8) as z:z.write(OUT/'activation_peak_explorer.html','activation_peak_explorer.html')
 with zipfile.ZipFile(viewer) as z:
  assert z.testzip() is None
  assert z.read('activation_peak_explorer.html')==(OUT/'activation_peak_explorer.html').read_bytes()
 direct=['01_celltype_stage_accessibility.png','02_celltype_interval_changes.png','08_global_ordering_controls.png','09_opening_window_positions.png','regional_activation_order.png']
 attachments=[entry(OUT/f,'image/png') for f in direct]+[entry(review,'application/zip')]
 manifest=dict(email1=attachments,email1_binary_bytes=sum(f['bytes'] for f in attachments),email2=[entry(viewer,'application/zip')],email2_binary_bytes=viewer.stat().st_size)
 assert manifest['email1_binary_bytes']<18_000_000 and manifest['email2_binary_bytes']<18_000_000,manifest
 (p/'delivery_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
 print(json.dumps({k:v for k,v in manifest.items() if k.endswith('bytes')}))
if __name__=='__main__':main()

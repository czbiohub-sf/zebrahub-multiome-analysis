#!/usr/bin/env python3
"""Package the reviewed publication figures; this script does not send mail."""
import os
from pathlib import Path
import json,hashlib,zipfile,html
HERE=Path(__file__).resolve().parent;OUT=HERE/'outputs/chromatin_velocity_figures_170mm_v1'
MAIL=OUT/'email_package/deliverables';MAIL.mkdir(parents=True,exist_ok=True)
main='MAIN_temporal_chromatin_dynamics';si='SI_parameter_robustness_and_cicero'
summary=(OUT/'results_summary.txt').read_text().strip()
paragraphs='\n'.join('<p>'+html.escape(p)+'</p>' for p in summary.split('\n\n'))
fullcaptions='\n\n'.join((OUT/f'{stem}_caption.txt').read_text().strip() for stem in [main,si])
(OUT/'figure_captions.txt').write_text(fullcaptions+'\n')
body='''<!doctype html><html><head><meta charset="utf-8"></head><body style="font:15px/1.55 Arial,sans-serif;max-width:900px;color:#18252d">
<p>Hello,</p><p>I assembled a main-style overview and a supplementary validation figure, both at <b>170 mm full width</b>. The overview is 180 mm high; the SI page is 205 mm high. Each has a vector PDF, editable SVG, and 600 dpi PNG.</p>
<h3>Approach, results and interpretation</h3>'''+paragraphs+'''<h3>Attached</h3>
<ul><li><b>Main-style overview — PNG and PDF:</b> workflow, distributed broad-accessibility reference, two interval arrow maps, graph-bias diagnostic, and Cicero forecasting test.</li>
<li><b>Supplementary figure — PNG and PDF:</b> parameter sensitivity, projection coverage, shuffled controls, Cicero coverage, held-out prediction, and synthetic controls.</li>
<li><b>Complete figure package — ZIP:</b> both pages in PNG/PDF/SVG, full captions, this two-paragraph summary, plotted CSV/NPZ data, figure scripts, validation manifests, and the interactive parameter explorer.</li></ul>
<p><b>My recommendation:</b> use the overview to explain the method and the SI page to document its limits. For a biological manuscript, I would describe these as retrospective chromatin dynamics and keep a predictive or universal root-flow interpretation exploratory.</p>
<p>PDF canvas dimensions, embedded fonts and text boundaries were checked, and both pages were visually reviewed at publication width. Print/place the PDFs at 100% to retain 170 mm width. These are new figure assemblies from the existing results; no models were refitted for this presentation.</p></body></html>'''
(MAIL/'email_body.html').write_text(body)
(OUT/'README.txt').write_text('''Chromatin dynamics: 170 mm figure package

MAIN_temporal_chromatin_dynamics: 170 x 180 mm overview.
SI_parameter_robustness_and_cicero: 170 x 205 mm validation page.
PDF = fixed publication canvas with embedded fonts.
SVG = editable text and vector axes/arrows; dense scatter is embedded raster.
PNG = 600 dpi (nominal 170 mm; pixel rounding applies).
Place PDFs at 100% without fit-to-page to retain exact physical width.

results_summary.txt contains the requested two paragraphs.
figure_captions.txt contains complete panel captions.
Source_data/ holds the exact map arrays and quantitative plot values.
Scripts/ contains figure and packaging code. Run figure scripts in the original repository's notebooks/EDA_chromatin_velocity directory to access the existing model outputs; large original H5 analysis inputs are not duplicated here. Source paths and validation checks are recorded in manifests.
parameter_explorer.html is the previously generated self-contained interactive sensitivity viewer. It uses 20,000 query peaks and full-atlas candidate neighbors. Display-normalized arrows describe profile correspondence; opening/closing colors are separate fitted rates. The viewer is exploratory, not validation of physical peak movement.

No models or input analyses were changed during figure assembly. No biological-replicate uncertainty is estimated. Read the captions before interpreting root-distance diagnostics or forecast intervals.
''')
links=[]
for stem,title in [(main,'Overview'),(si,'Supplementary validation')]:
 links.append('<section><h2>'+title+'</h2><p>'+ ' · '.join(f'<a href="{stem}.{e}">{e.upper()}</a>' for e in ['pdf','png','svg'])+'</p><img style="max-width:100%" src="'+stem+'.png" alt="'+title+' figure"><p>'+html.escape((OUT/f'{stem}_caption.txt').read_text())+'</p></section>')
(OUT/'index.html').write_text('<!doctype html><html><head><meta charset="utf-8"><title>Chromatin dynamics figures</title><style>body{max-width:1050px;margin:35px auto;font:16px/1.5 Arial;color:#18252d;padding:0 20px}h1{font-size:28px}section{margin:40px 0}section p{font-size:14px}a{color:#176681}</style></head><body><h1>Chromatin dynamics: publication figures</h1><p>Both figures are 170 mm wide. PNG: 600 dpi. PDF: exact physical canvas. SVG: editable text.</p>'+paragraphs+''.join(links)+'<p><a href="parameter_explorer.html">Interactive parameter explorer</a> · <a href="figure_captions.txt">Full captions</a> · <a href="results_summary.txt">Two-paragraph summary</a></p></body></html>')
# Local index and ZIP use the same self-contained parameter viewer.
viewer=HERE/'outputs/ct_velocity_experiments_v1/parameter_explorer.html'
(OUT/'parameter_explorer.html').write_bytes(viewer.read_bytes())
archive=MAIL/'chromatin_dynamics_170mm_complete.zip'
entries=[]
for stem in [main,si]:
 for e in ['pdf','svg','png']:entries.append((OUT/f'{stem}.{e}',f'{stem}.{e}'))
 entries.append((OUT/f'{stem}_manifest.json',f'Validation/{stem}_manifest.json'))
for name in ['results_summary.txt','figure_captions.txt','README.txt','index.html','parameter_explorer.html']:entries.append((OUT/name,name))
for p in sorted(OUT.glob('MAIN_*'))+sorted(OUT.glob('SI_*')):
 if p.suffix in ['csv','.csv','.npz']:entries.append((p,'Source_data/'+p.name))
for name in ['chromatin_velocity_summary_170mm.py','chromatin_velocity_si_170mm.py','chromatin_velocity_figure_delivery.py']:entries.append((HERE/name,'Scripts/'+name))
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
 for p,name in entries:z.write(p,name)
with zipfile.ZipFile(archive) as z:
 assert z.testzip() is None
 assert len(z.namelist())==len(set(z.namelist()))
 names=set(z.namelist());assert all(f'{s}.{e}' in names for s in [main,si] for e in ['pdf','png','svg'])
attachments=[]
for p,mime in [(OUT/f'{main}.png','image/png'),(OUT/f'{si}.png','image/png'),(OUT/f'{main}.pdf','application/pdf'),(OUT/f'{si}.pdf','application/pdf'),(archive,'application/zip')]:
 attachments.append({'path':str(p),'filename':p.name,'mime_type':mime,'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
manifest={'to':os.environ.get('CHROMATIN_REPORT_RECIPIENT', ''),'subject':'Re: Real chromatin velocity: peak UMAPs, interactive atlas, and depth-matching interpretation','attachments':attachments,'binary_bytes':sum(a['bytes'] for a in attachments),'archive_entries':len(entries),'summary_paragraphs':len(summary.split('\n\n')),'archive_test':'passed'}
assert manifest['binary_bytes']<18000000
(MAIL/'delivery_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps(manifest,indent=2))

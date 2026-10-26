"""Package review-sized artifacts; full model arrays stay in the shared workspace."""
from pathlib import Path
import json,zipfile,hashlib
from real_trajectory_velocity import OUT,HERE

def main():
    out=OUT;p=out/'email_package';p.mkdir(exist_ok=True)
    images=sorted(out.glob('*.png'))+sorted((out/'synthetic').glob('*.png'))+sorted((out/'weight_sensitivity').glob('*.png'))
    review=p/'trajectory_analysis_review.zip'
    small=[x for x in out.iterdir() if x.suffix in ['.json','.csv','.txt']]
    for folder in ['synthetic','weight_sensitivity']:
        small.extend(x for x in (out/folder).iterdir() if x.suffix in ['.json','.csv'])
    sources=list((HERE.parents[1]/'scripts/chrom_velo').glob('trajectory_*.py'))+[HERE/x for x in ['real_trajectory_velocity.py','trajectory_report.py','trajectory_benchmark.py','trajectory_artifacts.py','verify_trajectory_velocity.py','trajectory_weight_sensitivity.py','plot_trajectory_weight_sensitivity.py','finish_trajectory_report.py','package_trajectory_delivery.py']]
    sources += list(HERE.glob('test_trajectory_*.py'))
    for folder in ['browser_validation','stream_display']:
        small.extend(x for x in (out/folder).glob('*.json'))
    # Include imported local helpers needed to understand/reproduce reading and display.
    sources += [HERE/x for x in ['real_specialization_benchmark.py','real_specialization_viewer.py']]
    sources += [HERE.parents[1]/'scripts/chrom_velo'/x for x in ['real_specialization.py','streamline_field.py','streamline_paths.py']]
    with zipfile.ZipFile(review,'w',zipfile.ZIP_DEFLATED,compresslevel=8) as z:
        for f in images+small:z.write(f,arcname=str(f.relative_to(out)))
        z.writestr('analysis_summary.html',(out/'analysis_summary.html').read_text().replace('href="trajectory_peak_explorer.html"','href="trajectory_peak_explorer_email.html"'))
        for f in sources:z.write(f,arcname='source/'+str(f.relative_to(HERE.parents[1])))
        z.writestr('START_HERE.txt','Open analysis_summary.html for the interpretation and figures.\nThe separate trajectory_peak_explorer_email.zip contains the offline interactive viewer; extract it into this folder if you want the report link to work. Its 15,000 displayed peaks are a subset of the same 640,830-peak model.\nFull arrays, larger viewer, PDFs and source scripts are saved in the shared workspace:\n'+str(out)+'\nThe full model files are trajectory_fits.h5 (~1.5GB) and trajectory_velocity.h5ad (~0.95GB). They are omitted from email-sized archives.\nSource scripts retain the repository-relative hierarchy under source/ and use the existing repository helpers and local input atlas paths.\n')
    with zipfile.ZipFile(review) as z:assert z.testzip() is None
    attachments=[dict(path=str(f.resolve()),filename=f.name,mime_type='image/png',bytes=f.stat().st_size,sha256=hashlib.sha256(f.read_bytes()).hexdigest()) for f in images]
    attachments.append(dict(path=str(review.resolve()),filename=review.name,mime_type='application/zip',bytes=review.stat().st_size,sha256=hashlib.sha256(review.read_bytes()).hexdigest()))
    viewer=p/'trajectory_peak_explorer_email.zip'
    with zipfile.ZipFile(viewer) as z:
        assert z.testzip() is None
        assert z.read('trajectory_peak_explorer_email.html')==(p/'trajectory_peak_explorer_email.html').read_bytes()
    vm=dict(path=str(viewer.resolve()),filename=viewer.name,mime_type='application/zip',bytes=viewer.stat().st_size,sha256=hashlib.sha256(viewer.read_bytes()).hexdigest())
    result=dict(email1=attachments,email1_binary_bytes=sum(x['bytes'] for x in attachments),email2=[vm],email2_binary_bytes=vm['bytes'])
    assert result['email1_binary_bytes']<18_000_000 and result['email2_binary_bytes']<18_000_000
    (p/'delivery_manifest.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'pngs':len(images),'email1_binary_bytes':result['email1_binary_bytes'],'email2_binary_bytes':result['email2_binary_bytes']}))
if __name__=='__main__':main()

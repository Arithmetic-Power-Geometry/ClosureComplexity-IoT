from pathlib import Path
import subprocess,sys,hashlib,json,shutil,os
ROOT=Path(__file__).resolve().parent
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ.setdefault(key,'1')
RESULTS=ROOT/'results'
if RESULTS.exists():
    shutil.rmtree(RESULTS)
for rel in ['raw','tables','figures']:
    (RESULTS/rel).mkdir(parents=True,exist_ok=True)
for stage in ['scripts/main_benchmark.py','scripts/analyze.py','scripts/make_figures.py','scripts/validate_results.py']:
    print(f'==> {stage}', flush=True)
    subprocess.run([sys.executable,str(ROOT/stage)],cwd=ROOT,check=True)
manifest={}
files=[p for p in ROOT.rglob('*') if p.is_file() and '.git' not in p.parts and '__pycache__' not in p.parts and p.suffix not in {'.pyc'} and p.name!='manifest.json']
for p in sorted(files,key=lambda q:str(q.relative_to(ROOT))):
    if p.parts[-2] == '.pytest_cache':
        continue
    manifest[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
(RESULTS/'manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
print('Complete reproducibility workflow finished successfully.')

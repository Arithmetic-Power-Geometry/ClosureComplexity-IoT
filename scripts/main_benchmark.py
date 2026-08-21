from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import subprocess,sys,json,os
ROOT=Path(__file__).resolve().parents[1]
config=json.loads((ROOT/'configs'/'default.json').read_text())

def run_partition(n,dam):
    cmd=[sys.executable,str(ROOT/'scripts'/'benchmark_worker.py'),'--nodes',str(n),'--damage',str(dam)]
    env=os.environ.copy()
    for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:
        env[key]='1'
    cp=subprocess.run(cmd,cwd=ROOT,check=True,capture_output=True,text=True,env=env)
    return n,dam,cp.stdout.strip()

jobs=[(n,dam) for n in config['network_sizes'] for dam in config['damage_rates']]
with ThreadPoolExecutor(max_workers=min(3,len(jobs))) as ex:
    futs=[ex.submit(run_partition,*j) for j in jobs]
    for fut in as_completed(futs):
        n,dam,msg=fut.result(); print(msg or f'completed n={n}, damage={dam}',flush=True)
subprocess.run([sys.executable,str(ROOT/'scripts'/'combine_benchmark.py')],cwd=ROOT,check=True)

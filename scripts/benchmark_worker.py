from pathlib import Path
import sys,json,argparse
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from cc_iot import *
import numpy as np,pandas as pd
ap=argparse.ArgumentParser();ap.add_argument('--nodes',type=int,required=True);ap.add_argument('--damage',type=float,required=True);args=ap.parse_args()
config=json.loads((ROOT/'configs'/'default.json').read_text());T=Target(**config['target']);n=args.nodes;dam=args.damage;rows=[]
for mode in config['failure_modes']:
  for seed in config['seeds']:
   clear_cache();scenario_seed=int(seed+1000*n+10000*round(dam,2));s=generate_scenario(scenario_seed,n,dam,mode,int(config['candidate_actions']));base=metrics(s,0);healthy=healthy_metrics(s);exact_mask,exact_m=exact_cc(s,T);exact_feas=satisfies(exact_m,T) and exact_mask>=0;exact_cost=cost(s,exact_mask) if exact_feas else np.nan
   vals={'CC-Exact':(exact_mask,exact_m),'Greedy':greedy(s,T),'GA':genetic(s,T,seed+77),'PSO':pso_binary(s,T,seed+88),'NSGA-II':nsga2_binary(s,T,seed+66),'Random':random_search(s,T,seed+99)}
   for name,(mask,m) in vals.items():
    feas=(mask>=0 and satisfies(m,T));rc=cost(s,mask) if mask>=0 else np.nan;cog=(rc-exact_cost) if (feas and exact_feas) else np.nan;ncog=(cog/exact_cost) if (feas and exact_feas and exact_cost>1e-12) else (0.0 if feas and exact_feas and abs(rc-exact_cost)<1e-12 else np.nan)
    rows.append(dict(nodes=n,damage=dam,failure_mode=mode,seed=seed,scenario_seed=scenario_seed,method=name,feasible=feas,repair_cost=rc,steps=steps(mask),exact_cc=exact_cost,cog=cog,ncog=ncog,optimum_hit=bool(feas and exact_feas and abs(rc-exact_cost)<1e-9),healthy_coverage=healthy['coverage'],healthy_reliability=healthy['reliability'],base_coverage=base['coverage'],base_reliability=base['reliability'],base_latency_ms=base['latency_ms'],base_energy_mj=base['energy_mj'],**m))
out=ROOT/'results'/'raw'/f'benchmark_n{n}_d{int(round(dam*100))}.csv';pd.DataFrame(rows).to_csv(out,index=False);print(out.name,len(rows))

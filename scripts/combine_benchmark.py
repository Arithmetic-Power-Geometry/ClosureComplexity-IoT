from pathlib import Path
import json,pandas as pd
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results'/'raw';TAB=ROOT/'results'/'tables';TAB.mkdir(parents=True,exist_ok=True);cfg=json.loads((ROOT/'configs'/'default.json').read_text())
parts=[]
for n in cfg['network_sizes']:
 for dam in cfg['damage_rates']:
  p=OUT/f'benchmark_n{n}_d{int(round(dam*100))}.csv'
  if not p.exists():raise FileNotFoundError(p)
  parts.append(pd.read_csv(p))
df=pd.concat(parts,ignore_index=True);df.to_csv(OUT/'benchmark_results.csv',index=False)
order=['CC-Exact','Greedy','GA','PSO','NSGA-II','Random']
rows=[]
for method in order:
    g=df[df.method==method].copy()
    gf=g[g.feasible].copy()
    paired=gf[gf.exact_cc.notna()].copy()
    rows.append(dict(
        method=method,
        n_feasible=int(len(gf)),
        success_rate=float(g.feasible.mean()),
        mean_feasible_cost=float(gf.repair_cost.mean()) if len(gf) else float('nan'),
        median_feasible_cost=float(gf.repair_cost.median()) if len(gf) else float('nan'),
        mean_feasible_steps=float(gf.steps.mean()) if len(gf) else float('nan'),
        mean_ncog=float(paired.ncog.mean()) if len(paired) else float('nan'),
        paired_optimum_hit_rate=float(paired.optimum_hit.mean()) if len(paired) else float('nan'),
        mean_coverage=float(gf.coverage.mean()) if len(gf) else float('nan'),
        mean_reliability=float(gf.reliability.mean()) if len(gf) else float('nan'),
        mean_latency_ms=float(gf.latency_ms.mean()) if len(gf) else float('nan'),
        mean_energy_mj=float(gf.energy_mj.mean()) if len(gf) else float('nan')
    ))
pd.DataFrame(rows).to_csv(TAB/'table_main_results.csv',index=False)
print('Combined',len(parts),'benchmark partitions into',len(df),'rows')

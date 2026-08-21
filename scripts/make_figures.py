from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from cc_iot import *
import pandas as pd,numpy as np
import matplotlib.pyplot as plt
OUT=ROOT/'results'/'raw';TAB=ROOT/'results'/'tables';FIG=ROOT/'results'/'figures';FIG.mkdir(parents=True,exist_ok=True);config=json.loads((ROOT/'configs'/'default.json').read_text());T=Target(**config['target'])
df=pd.read_csv(OUT/'benchmark_results.csv');cc=df[(df.method=='CC-Exact')&df.feasible];idf=pd.read_csv(TAB/'table_interaction.csv');states=pd.read_csv(OUT/'representative_state_space.csv');pf=pd.read_csv(TAB/'table_pareto_front.csv');scale=pd.read_csv(TAB/'table_scalability.csv');pair=pd.read_csv(TAB/'table_matched_pair.csv') if (TAB/'table_matched_pair.csv').stat().st_size>1 else pd.DataFrame()
methods=['CC-Exact','Greedy','GA','PSO','NSGA-II','Random']
plt.figure(figsize=(7.2,4.6))
for method in methods:
 g=df[(df.method==method)&df.feasible];x=g.groupby('damage').repair_cost.mean();plt.plot(x.index*100,x.values,marker='o',label=method)
plt.xlabel('Node failure rate (%)');plt.ylabel('Mean repair cost (normalized units)');plt.legend(ncol=2,fontsize=8);plt.tight_layout();plt.savefig(FIG/'fig_cost_vs_damage.pdf');plt.savefig(FIG/'fig_cost_vs_damage.png',dpi=220);plt.close()
plt.figure(figsize=(7.2,4.6))
for mode,g in cc.groupby('failure_mode'):
 x=g.groupby('damage').repair_cost.mean();plt.plot(x.index*100,x.values,marker='o',label=mode)
plt.xlabel('Node failure rate (%)');plt.ylabel('Exact closure complexity');plt.legend(fontsize=8);plt.tight_layout();plt.savefig(FIG/'fig_failure_geometry.pdf');plt.savefig(FIG/'fig_failure_geometry.png',dpi=220);plt.close()
plt.figure(figsize=(6.7,4.5));plt.scatter(cc.base_reliability,cc.repair_cost,s=24,alpha=.65);plt.xlabel('Degraded-state expected reliability');plt.ylabel('Exact closure complexity');plt.tight_layout();plt.savefig(FIG/'fig_recoverability.pdf');plt.savefig(FIG/'fig_recoverability.png',dpi=220);plt.close()
plt.figure(figsize=(6.7,4.5))
for b in ['Greedy','GA','PSO','NSGA-II','Random']:
 g=df[(df.method==b)&df.feasible&df.exact_cc.notna()];plt.scatter(g.exact_cc,g.ncog,s=18,alpha=.5,label=b)
plt.axhline(0,lw=.8);plt.xlabel('Exact closure complexity');plt.ylabel('Normalized closure optimality gap');plt.legend(fontsize=8);plt.tight_layout();plt.savefig(FIG/'fig_ncog_vs_cc.pdf');plt.savefig(FIG/'fig_ncog_vs_cc.png',dpi=220);plt.close()
plt.figure(figsize=(6.8,4.5));labels=[f"{r.target_a}-{r.target_b}" for r in idf.itertuples()];plt.bar(labels,idf.closure_interaction);plt.axhline(0,lw=.8);plt.ylabel('Closure interaction (cost units)');plt.xticks(rotation=30,ha='right');plt.tight_layout();plt.savefig(FIG/'fig_interaction.pdf');plt.savefig(FIG/'fig_interaction.png',dpi=220);plt.close()
plt.figure(figsize=(6.5,4.5));plt.scatter(states.cost,states.coverage,s=10,alpha=.25,label='All repair states');
if len(pf):plt.scatter(pf.cost,pf.coverage,s=32,label='Pareto target-satisfying states')
plt.axhline(T.coverage,ls='--',lw=1,label='Coverage target');plt.xlabel('Repair cost');plt.ylabel('Coverage');plt.legend(fontsize=8);plt.tight_layout();plt.savefig(FIG/'fig_pareto.pdf');plt.savefig(FIG/'fig_pareto.png',dpi=220);plt.close()
plt.figure(figsize=(7.0,4.5))
for method in ['Greedy','GA','PSO','NSGA-II','Random']:
 g=scale[scale.method==method].sort_values('actions');plt.plot(g.actions,g.median_ncog,marker='o',label=method)
 if method not in ['Greedy'] and len(g): plt.fill_between(g.actions,g.q25_ncog,g.q75_ncog,alpha=.12)
plt.xlabel('Candidate repair actions');plt.ylabel('Median normalized closure optimality gap');plt.xticks(config['scalability_actions']);plt.legend(fontsize=8);plt.tight_layout();plt.savefig(FIG/'fig_scalability_gap.pdf');plt.savefig(FIG/'fig_scalability_gap.png',dpi=220);plt.close()
if len(pair)==2:
 fig,axes=plt.subplots(1,2,figsize=(9.2,4.2),sharex=True,sharey=True)
 for ax,row in zip(axes,pair.itertuples()):
  ss=generate_scenario(int(row.scenario_seed),int(row.nodes),float(row.damage),row.failure_mode,8);ax.scatter(ss.nodes[ss.active,0],ss.nodes[ss.active,1],s=14,label='active');ax.scatter(ss.nodes[ss.failed,0],ss.nodes[ss.failed,1],s=22,marker='x',label='failed');ax.scatter(ss.gateways[:,0],ss.gateways[:,1],s=50,marker='s',label='gateway');ax.set_title(f"Case {row.case}: {row.failure_mode}\nCC={row.CC:.2f}, C={row.base_coverage:.3f}, R={row.base_reliability:.3f}");ax.set_xlim(0,AREA);ax.set_ylim(0,AREA);ax.set_xlabel('x (m)')
 axes[0].set_ylabel('y (m)');axes[0].legend(fontsize=7,loc='lower left');fig.tight_layout();fig.savefig(FIG/'fig_matched_pair.pdf');fig.savefig(FIG/'fig_matched_pair.png',dpi=220);plt.close(fig)
print('Figure stage complete')

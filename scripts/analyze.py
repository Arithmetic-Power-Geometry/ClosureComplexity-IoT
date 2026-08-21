from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from cc_iot import *
import numpy as np,pandas as pd
from scipy.stats import wilcoxon,spearmanr
OUT=ROOT/'results'/'raw';TAB=ROOT/'results'/'tables';FIG=ROOT/'results'/'figures'; TAB.mkdir(parents=True,exist_ok=True); FIG.mkdir(parents=True,exist_ok=True)
config=json.loads((ROOT/'configs'/'default.json').read_text());T=Target(**config['target'])
df=pd.read_csv(OUT/'benchmark_results.csv');cc=df[df.method=='CC-Exact'].copy();cc_feas=cc[cc.feasible].copy()
wide=df[df.feasible].pivot_table(index=['nodes','damage','failure_mode','seed'],columns='method',values='repair_cost')
stats=[]
for b in ['Greedy','GA','PSO','NSGA-II','Random']:
 z=wide[['CC-Exact',b]].dropna();diff=z[b]-z['CC-Exact'] if len(z) else pd.Series(dtype=float)
 if len(diff)==0:w,p=np.nan,np.nan
 elif np.allclose(diff.to_numpy(float),0):w,p=np.nan,np.nan
 else:
  try:w,p=wilcoxon(diff,alternative='greater')
  except Exception:w,p=np.nan,np.nan
 stats.append(dict(baseline=b,n=len(z),mean_excess_cost=float(diff.mean()) if len(diff) else np.nan,median_excess_cost=float(diff.median()) if len(diff) else np.nan,mean_normalized_gap=float(((z[b]-z['CC-Exact'])/z['CC-Exact'].replace(0,np.nan)).mean()) if len(z) else np.nan,optimum_hit_rate=float(np.isclose(z[b],z['CC-Exact']).mean()) if len(z) else np.nan,wilcoxon_W=w,p_value=p))
pd.DataFrame(stats).to_csv(TAB/'table_algorithm_gap.csv',index=False)
geom=cc_feas.groupby(['damage','failure_mode']).repair_cost.agg(['count','mean','std','median']).reset_index();geom.to_csv(TAB/'table_failure_geometry.csv',index=False)
# Failure-family vulnerability: distinguish finite repair burden from unreachable modes.
basefam=cc.groupby(['nodes','damage','seed'])
fam=basefam.apply(lambda g: pd.Series({
 'cvhat_finite': float(g.loc[g.feasible,'repair_cost'].max()) if g.feasible.any() else np.nan,
 'min_cc_finite': float(g.loc[g.feasible,'repair_cost'].min()) if g.feasible.any() else np.nan,
 'mean_cc_finite': float(g.loc[g.feasible,'repair_cost'].mean()) if g.feasible.any() else np.nan,
 'spread_finite': float(g.loc[g.feasible,'repair_cost'].max()-g.loc[g.feasible,'repair_cost'].min()) if g.feasible.any() else np.nan,
 'finite_modes': int(g.feasible.sum()),
 'unreachable_modes': int((~g.feasible).sum())
}), include_groups=False).reset_index()
fam.to_csv(TAB/'table_failure_family_vulnerability.csv',index=False)
assoc=[]
for variable in ['damage','nodes','base_coverage','base_reliability','base_latency_ms','base_energy_mj']:
 x=cc_feas[[variable,'repair_cost']].dropna();rho,pv=spearmanr(x[variable],x.repair_cost) if len(x)>2 else (np.nan,np.nan);assoc.append(dict(variable=variable,n=len(x),spearman_rho=float(rho),p_value=float(pv)))
pd.DataFrame(assoc).to_csv(TAB/'table_recoverability_associations.csv',index=False)
gap_assoc=[]
for b in ['Greedy','GA','PSO','NSGA-II','Random']:
 g=df[(df.method==b)&df.feasible&df.exact_cc.notna()].copy(); gx=g.exact_cc.to_numpy(float); gy=g.ncog.fillna(0).to_numpy(float); rho,pv=((np.nan,np.nan) if len(g)<=2 or np.allclose(gx,gx[0]) or np.allclose(gy,gy[0]) else spearmanr(gx,gy)); gap_assoc.append(dict(method=b,n=len(g),spearman_rho_cc_vs_ncog=float(rho),p_value=float(pv),mean_ncog=float(g.ncog.mean()),optimum_hit_rate=float(g.optimum_hit.mean())))
pd.DataFrame(gap_assoc).to_csv(TAB/'table_cc_algorithmic_difficulty.csv',index=False)
# matched pair
cand=[]
for _,g in cc_feas.groupby(['nodes','damage','seed']):
 rr=list(g.itertuples())
 for i in range(len(rr)):
  for j in range(i+1,len(rr)):
   a,b=rr[i],rr[j];qdist=abs(a.base_coverage-b.base_coverage)+abs(a.base_reliability-b.base_reliability);gap=abs(a.repair_cost-b.repair_cost);cand.append((qdist,-gap,a,b))
eligible=[x for x in cand if x[0]<=0.08] or cand
eligible.sort(key=lambda x:(x[0],x[1]));minq=eligible[0][0] if eligible else np.nan;pool=[x for x in eligible if x[0]<=minq+0.03] if eligible else []
best=max(pool,key=lambda x:-x[1]) if pool else None
if best:
 _,_,a,b=best;pair=pd.DataFrame([dict(case='A',nodes=a.nodes,damage=a.damage,seed=a.seed,scenario_seed=a.scenario_seed,failure_mode=a.failure_mode,base_coverage=a.base_coverage,base_reliability=a.base_reliability,CC=a.repair_cost),dict(case='B',nodes=b.nodes,damage=b.damage,seed=b.seed,scenario_seed=b.scenario_seed,failure_mode=b.failure_mode,base_coverage=b.base_coverage,base_reliability=b.base_reliability,CC=b.repair_cost)])
else:pair=pd.DataFrame()
pair.to_csv(TAB/'table_matched_pair.csv',index=False)
# Nested linear models: does failure geometry add explanatory value beyond damage/QoS/size?
reg=cc_feas[['repair_cost','damage','nodes','base_coverage','base_reliability','base_latency_ms','base_energy_mj','failure_mode']].dropna().copy()
y=reg.repair_cost.to_numpy(float)
def fit_ols(X,y):
 X=np.asarray(X,float); X=np.column_stack([np.ones(len(X)),X]); beta=np.linalg.lstsq(X,y,rcond=None)[0]; pred=X@beta; rss=float(np.sum((y-pred)**2)); tss=float(np.sum((y-y.mean())**2)); r2=1-rss/tss if tss>0 else np.nan; p=X.shape[1]; n=len(y); adj=1-(1-r2)*(n-1)/(n-p) if n>p and np.isfinite(r2) else np.nan; return rss,r2,adj,p

def vif_values(X, names):
    X=np.asarray(X,float); rows=[]
    for j,name in enumerate(names):
        yj=X[:,j]; others=np.delete(X,j,axis=1)
        if others.shape[1]==0:
            r2=0.0
        else:
            _,r2,_,_=fit_ols(others,yj)
        vif=np.inf if np.isfinite(r2) and r2>=1-1e-12 else (1.0/(1.0-r2) if np.isfinite(r2) else np.nan)
        rows.append(dict(predictor=name,vif=float(vif)))
    return pd.DataFrame(rows)
# Base-state latency is an affine function of reliability under Eq. (15) because no relay is active before repair; omit it to avoid exact collinearity.
X0=reg[['damage','nodes','base_coverage','base_reliability','base_energy_mj']].to_numpy(float)
dummies=pd.get_dummies(reg.failure_mode,drop_first=True,dtype=float)
X1=np.column_stack([X0,dummies.to_numpy(float)])
rss0,r20,adj0,p0=fit_ols(X0,y); rss1,r21,adj1,p1=fit_ols(X1,y)
df1=p1-p0; df2=len(y)-p1
from scipy.stats import f as fdist
F=((rss0-rss1)/df1)/(rss1/df2) if df1>0 and df2>0 and rss1>0 else np.nan
pF=float(fdist.sf(F,df1,df2)) if np.isfinite(F) else np.nan
pd.DataFrame([dict(model='QoS+damage+size',n=len(y),r2=r20,adjusted_r2=adj0,delta_r2=0.0,F_geometry=np.nan,p_geometry=np.nan),dict(model='QoS+damage+size+failure_geometry',n=len(y),r2=r21,adjusted_r2=adj1,delta_r2=r21-r20,F_geometry=F,p_geometry=pF)]).to_csv(TAB/'table_nested_models.csv',index=False)
base_names=['damage','nodes','coverage','reliability','energy']
vif_values(X0,base_names).to_csv(TAB/'table_vif.csv',index=False)
# Systematic damage-equivalent matching within identical size/damage/seed.
pairs=[]
for key,g in cc_feas.groupby(['nodes','damage','seed']):
 rr=list(g.itertuples())
 for i in range(len(rr)):
  for j in range(i+1,len(rr)):
   a,b=rr[i],rr[j]
   dc=abs(a.base_coverage-b.base_coverage); dr=abs(a.base_reliability-b.base_reliability); dcc=abs(a.repair_cost-b.repair_cost)
   pairs.append(dict(nodes=key[0],damage=key[1],seed=key[2],mode_a=a.failure_mode,mode_b=b.failure_mode,delta_coverage=dc,delta_reliability=dr,delta_cc=dcc,matched_001=bool(dc<0.01 and dr<0.01),matched_002=bool(dc<0.02 and dr<0.02),matched_005=bool(dc<0.05 and dr<0.05)))
pairdf=pd.DataFrame(pairs); pairdf.to_csv(OUT/'matched_pairs_all.csv',index=False)
summary=[]
for col,tol in [('matched_001',.01),('matched_002',.02),('matched_005',.05)]:
 z=pairdf[pairdf[col]]
 summary.append(dict(qos_tolerance=tol,n_pairs=len(z),median_abs_delta_cc=float(z.delta_cc.median()) if len(z) else np.nan,mean_abs_delta_cc=float(z.delta_cc.mean()) if len(z) else np.nan,p90_abs_delta_cc=float(z.delta_cc.quantile(.9)) if len(z) else np.nan,share_delta_cc_ge_1=float((z.delta_cc>=1).mean()) if len(z) else np.nan))
pd.DataFrame(summary).to_csv(TAB/'table_matched_pair_summary.csv',index=False)

# representative state space/CI/Pareto
clear_cache();s=generate_scenario(565,30,.15,'clustered',8);states=enumerate_states(s,T);states.to_csv(OUT/'representative_state_space.csv',index=False);pf=pareto(states);pf.to_csv(TAB/'table_pareto_front.csv',index=False)
ints=[]
for a,b in [('coverage','reliability'),('coverage','latency'),('reliability','latency'),('coverage','energy'),('reliability','energy')]:
 ma,_=exact_single_target(s,T,a);mb,_=exact_single_target(s,T,b);jt=Target(coverage=T.coverage if 'coverage' in (a,b) else 0,reliability=T.reliability if 'reliability' in (a,b) else 0,latency_ms=T.latency_ms if 'latency' in (a,b) else 1e9,energy_mj=T.energy_mj if 'energy' in (a,b) else 1e9);mj,_=exact_cc(s,jt);ca,cb,cj=cost(s,ma),cost(s,mb),cost(s,mj);ints.append(dict(target_a=a,target_b=b,CC_a=ca,CC_b=cb,CC_joint=cj,closure_interaction=ca+cb-cj))
pd.DataFrame(ints).to_csv(TAB/'table_interaction.csv',index=False)
# CC vs CER: deterministic representative case showing step-optimality and cost-optimality can disagree
cer_records=[]
clear_cache(); ss=generate_scenario(32016,30,.15,'random',8); st=enumerate_states(ss,T); sat=st[st.satisfies]
if not sat.empty:
 ccrow=sat.sort_values(['cost','steps']).iloc[0]; cerrow=sat.sort_values(['steps','cost']).iloc[0]
 for label,row in [('CC minimum cost',ccrow),('CER minimum steps',cerrow)]:
  mask=int(row['mask']); labels=selected_actions(ss,mask); kinds=selected_action_kinds(ss,mask)
  cer_records.append(dict(selection=label,mask=mask,cost=float(row['cost']),steps=int(row['steps']),selected_actions=' + '.join(labels),action_kinds=' + '.join(kinds),coverage=float(row['coverage']),reliability=float(row['reliability']),latency_ms=float(row['latency_ms']),energy_mj=float(row['energy_mj'])))
pd.DataFrame(cer_records).to_csv(TAB/'table_cc_vs_cer.csv',index=False)
# scalability: repeated stochastic runs on a fixed geometry
import time
scale=[]
reps=int(config.get('scalability_replicates',30)); seed_base=int(config.get('scalability_seed_base',9000))
for ma in config['scalability_actions']:
 clear_cache(); scenario_seed=7777; ss=generate_scenario(scenario_seed,30,.30,'gateway-near',int(ma))
 t0=time.perf_counter(); exm,exq=exact_cc(ss,T); tex=time.perf_counter()-t0; exok=exm>=0 and satisfies(exq,T); excc=cost(ss,exm) if exok else np.nan
 # Exact reference and deterministic greedy are each evaluated once.
 scale.append(dict(actions=ma,state_space=2**ma,scenario_seed=scenario_seed,algorithm_seed=np.nan,replicate=0,method='CC-Exact',feasible=exok,cost=excc,exact_cc=excc,cog=0.0 if exok else np.nan,ncog=0.0 if exok and excc>0 else np.nan,optimum_hit=bool(exok),runtime_s=tex))
 t0=time.perf_counter(); gm,gq=greedy(ss,T); grt=time.perf_counter()-t0; gok=gm>=0 and satisfies(gq,T); gc=cost(ss,gm) if gok else np.nan; gap=(gc-excc) if gok and exok else np.nan
 scale.append(dict(actions=ma,state_space=2**ma,scenario_seed=scenario_seed,algorithm_seed=np.nan,replicate=0,method='Greedy',feasible=gok,cost=gc,exact_cc=excc,cog=gap,ncog=(gap/excc if gok and exok and excc>0 else np.nan),optimum_hit=bool(gok and exok and np.isclose(gc,excc)),runtime_s=grt))
 for rep in range(reps):
  seeds={'GA':seed_base+100000*ma+rep,'PSO':seed_base+200000*ma+rep,'NSGA-II':seed_base+300000*ma+rep,'Random':seed_base+400000*ma+rep}
  funcs={'GA':genetic,'PSO':pso_binary,'NSGA-II':nsga2_binary,'Random':random_search}
  for name,fn in funcs.items():
   alg_seed=int(seeds[name]); t0=time.perf_counter(); mask,q=fn(ss,T,alg_seed); runtime=time.perf_counter()-t0
   ok=mask>=0 and satisfies(q,T); cst=cost(ss,mask) if ok else np.nan; gap=(cst-excc) if ok and exok else np.nan; ncog=(gap/excc if ok and exok and excc>0 else np.nan)
   scale.append(dict(actions=ma,state_space=2**ma,scenario_seed=scenario_seed,algorithm_seed=alg_seed,replicate=rep+1,method=name,feasible=ok,cost=cst,exact_cc=excc,cog=gap,ncog=ncog,optimum_hit=bool(ok and exok and np.isclose(cst,excc)),runtime_s=runtime))
scaledf=pd.DataFrame(scale);scaledf.to_csv(OUT/'scalability_results.csv',index=False)
summary=[]
for (ma,statespace,method),g in scaledf.groupby(['actions','state_space','method'],sort=True):
 vals=g.loc[g.feasible,'ncog'].dropna(); runt=g.runtime_s.dropna()
 summary.append(dict(actions=ma,state_space=statespace,method=method,runs=len(g),success_rate=float(g.feasible.mean()),exact_cc=float(g.exact_cc.dropna().iloc[0]) if g.exact_cc.notna().any() else np.nan,mean_ncog=float(vals.mean()) if len(vals) else np.nan,median_ncog=float(vals.median()) if len(vals) else np.nan,q25_ncog=float(vals.quantile(.25)) if len(vals) else np.nan,q75_ncog=float(vals.quantile(.75)) if len(vals) else np.nan,sd_ncog=float(vals.std(ddof=1)) if len(vals)>1 else 0.0 if len(vals)==1 else np.nan,optimum_hit_rate=float(g.optimum_hit.mean()),mean_runtime_s=float(runt.mean()) if len(runt) else np.nan,median_runtime_s=float(runt.median()) if len(runt) else np.nan))
pd.DataFrame(summary).to_csv(TAB/'table_scalability.csv',index=False)
print('Analysis stage complete')

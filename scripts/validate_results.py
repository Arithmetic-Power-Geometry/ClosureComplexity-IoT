from pathlib import Path
import json, math
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
RAW=ROOT/'results'/'raw'; TAB=ROOT/'results'/'tables'; FIG=ROOT/'results'/'figures'
errors=[]

def req(path):
    if not path.exists() or path.stat().st_size==0: errors.append(f'missing/empty: {path.relative_to(ROOT)}')

for name in ['benchmark_results.csv','scalability_results.csv','representative_state_space.csv','matched_pairs_all.csv']:
    req(RAW/name)
for name in ['table_main_results.csv','table_algorithm_gap.csv','table_nested_models.csv','table_vif.csv','table_cc_vs_cer.csv','table_scalability.csv']:
    req(TAB/name)
for name in ['fig_cost_vs_damage.pdf','fig_failure_geometry.pdf','fig_recoverability.pdf','fig_matched_pair.pdf','fig_interaction.pdf','fig_pareto.pdf','fig_scalability_gap.pdf','fig_ncog_vs_cc.pdf']:
    req(FIG/name)
if errors: raise SystemExit('\n'.join(errors))

df=pd.read_csv(RAW/'benchmark_results.csv')
assert len(df)==810, len(df)
# Exact solver must never be beaten by any feasible paired method.
paired=df[df['exact_cc'].notna() & df['feasible']]
if ((paired['repair_cost']-paired['exact_cc']) < -1e-9).any():
    bad=paired.loc[(paired['repair_cost']-paired['exact_cc']) < -1e-9,['method','repair_cost','exact_cc']]
    raise AssertionError(f'algorithm below exact CC:\n{bad}')
# Main summary must use each method's feasible subset, not all returned masks.
main=pd.read_csv(TAB/'table_main_results.csv')
for r in main.itertuples():
    g=df[(df.method==r.method)&df.feasible]
    assert int(r.n_feasible)==len(g)
    if len(g):
        assert abs(r.mean_feasible_cost-g.repair_cost.mean())<1e-10
        assert abs(r.mean_feasible_steps-g.steps.mean())<1e-10

# Gap table must reproduce paired differences exactly.
gap=pd.read_csv(TAB/'table_algorithm_gap.csv')
for r in gap.itertuples():
    z=df[(df.method.isin(['CC-Exact',r.baseline])) & df.feasible].pivot_table(index=['nodes','damage','failure_mode','seed'],columns='method',values='repair_cost')
    if 'CC-Exact' not in z.columns or r.baseline not in z.columns: continue
    z=z[['CC-Exact',r.baseline]].dropna(); diff=z[r.baseline]-z['CC-Exact']
    assert len(z)==int(r.n)
    assert abs(diff.mean()-r.mean_excess_cost)<1e-10
# Table 10 masks must match action counts and exported labels.
cccer=pd.read_csv(TAB/'table_cc_vs_cer.csv')
for r in cccer.itertuples():
    labels=[x.strip() for x in str(r.selected_actions).split('+') if x.strip()]
    assert len(labels)==int(r.steps)
# Regression R2 and adjusted R2 must be in sensible bounds; VIF >= 1.
nested=pd.read_csv(TAB/'table_nested_models.csv'); assert ((nested.r2>=0)&(nested.r2<=1)).all(); assert ((nested.adjusted_r2<=1)).all()
vif=pd.read_csv(TAB/'table_vif.csv'); assert (vif.vif>=1-1e-9).all()
# Scaling: 30 replicates for each stochastic algorithm/action count.
cfg=json.loads((ROOT/'configs'/'default.json').read_text()); reps=int(cfg['scalability_replicates'])
sc=pd.read_csv(RAW/'scalability_results.csv')
for ma in cfg['scalability_actions']:
    for method in ['GA','PSO','NSGA-II','Random']:
        assert len(sc[(sc.actions==ma)&(sc.method==method)])==reps
print('Result validation passed: tables, paired gaps, CC/CER actions, regression diagnostics, and repeated scaling are internally consistent.')

import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from cc_iot import *

def test_exact_not_worse_than_greedy():
    t=Target();s=generate_scenario(1234,30,.30,'clustered',8)
    em,eq=exact_cc(s,t);gm,gq=greedy(s,t)
    assert em>=0 and satisfies(eq,t)
    if satisfies(gq,t): assert cost(s,em)<=cost(s,gm)+1e-12

def test_failure_modes_preserve_failure_count():
    for mode in ['random','clustered','gateway-near','peripheral','coverage-critical']:
        s=generate_scenario(99,60,.30,mode,8)
        assert len(s.failed)==18
        assert int((~s.active).sum())==18

def test_nsga2_deterministic():
    t=Target();s=generate_scenario(2468,30,.15,'random',8)
    a,_=nsga2_binary(s,t,777);b,_=nsga2_binary(s,t,777)
    assert a==b

def test_exact_reproducible_across_cache_clear():
    t=Target();s=generate_scenario(55,30,.30,'coverage-critical',8)
    a,_=exact_cc(s,t);ca=cost(s,a);clear_cache();b,_=exact_cc(s,t);cb=cost(s,b)
    assert a==b
    if a>=0: assert abs(ca-cb)<1e-12

def test_exact_matches_bruteforce_on_small_lattice():
    t=Target(); s=generate_scenario(4242,30,.15,'random',8)
    em,eq=exact_cc(s,t)
    states=enumerate_states(s,t); sat=states[states.satisfies]
    if sat.empty:
        assert em<0
    else:
        assert em>=0
        assert abs(cost(s,em)-float(sat.cost.min()))<1e-12

def test_action_labels_match_mask_and_steps():
    s=generate_scenario(32016,30,.15,'random',8)
    mask=(1<<0)|(1<<2)|(1<<5)
    assert len(selected_actions(s,mask))==steps(mask)==3
    assert len(selected_action_kinds(s,mask))==3

def test_cost_is_sum_of_selected_action_costs():
    s=generate_scenario(32016,30,.15,'random',8)
    for mask in [0,1,3,17,255]:
        expected=sum(a.cost for i,a in enumerate(s.actions) if (mask>>i)&1)
        assert abs(cost(s,mask)-expected)<1e-12

#!/usr/bin/env python3
"""Run and grade Protective Personal Space without rendering."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from pps_metrics import gates,summarize
from rollout_pps import PpsRollout


def main():
    p=argparse.ArgumentParser();p.add_argument('--policy',default=str(ROOT/'onnx/alpha_walking.onnx'))
    p.add_argument('--seconds',type=float,default=150.0);p.add_argument('--json',default='')
    p.add_argument('--trace',default='');p.add_argument('--quiet',action='store_true');a=p.parse_args()
    r=PpsRollout(a.policy,a.seconds)
    def progress(i,x):
        if not a.quiet: print(f"t={x['t']:6.2f} {x['state']:<18} active={str(x['active']):<6} ward={x['ward_range_m']:.2f} cmd={x['command_peak']:.2f}")
    r.run(progress if not a.quiet else None);s=summarize(r);results=gates(s)
    s['gate_results']=[{'gate':n,'pass':ok,'evidence':e} for n,ok,e in results]
    s['gates_passed']=sum(ok for _,ok,_ in results);s['gates_total']=len(results);s['all_gates_pass']=all(ok for _,ok,_ in results)
    print('\nTRANSITIONS');[print(f" {x['t']:6.2f} {x['from']:<18} -> {x['to']:<18}") for x in s['transitions']]
    print('\nACCEPTANCE GATES')
    for n,ok,e in results:print(f" [{'OK' if ok else 'FAIL'}] {n}: {e}")
    print(f"\n{s['gates_passed']}/{s['gates_total']} gates; episodes={s['episode_kinds']}; path={s['path_m']}m")
    print('ALL GATES PASS' if s['all_gates_pass'] else 'GATES FAILED')
    if a.json:Path(a.json).write_text(json.dumps(s,indent=2))
    if a.trace:Path(a.trace).write_text(json.dumps(r.records))
    return 0 if s['all_gates_pass'] else 1
if __name__=='__main__':raise SystemExit(main())

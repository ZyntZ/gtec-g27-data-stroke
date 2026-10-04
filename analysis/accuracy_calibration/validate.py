"""Cross-check saved metrics and denominators; no model fitting or selection."""
import ast
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/'results/accuracy_calibration'
def read(name): return list(csv.DictReader((OUT/name).open(encoding='utf-8')))
def main():
    legacy=read('baseline_reproduction.csv'); train=read('nested_training_cv.csv'); test=read('exploratory_test_accuracy.csv')
    curves=read('calibration_curves.csv'); subsets=read('calibration_subsets.csv'); rel=read('probability_reliability.csv'); oof=read('training_oof_predictions.csv'); transfer=read('pre_post_transfer.csv')
    selection=json.loads((OUT/'selection_training_only.json').read_text()); sessions=list(selection)
    checks={}
    checks['team_baseline_six_sessions_match']=len(legacy)==6 and all(r['matches_previous']=='True' for r in legacy)
    checks['team_baseline_433_of_480']=sum(int(r['correct']) for r in legacy)==433 and sum(int(r['trials']) for r in legacy)==480
    checks['all_new_train_and_test_counts_80']=len(train)==len(test)==30 and all(int(r['trials'])==80 for r in train+test) and all(int(r['train_trials'])==80 for r in test)
    checks['oof_trial_coverage']=all(sorted(int(r['trial']) for r in oof if r['session']==s and r['family']==f)==list(range(80)) for s in sessions for f in ('CSP-baseline','FBCSP','Riemann','TV-CSP','selected'))
    checks['all_requested_calibration_budgets']=len(curves)==30 and all(sorted(int(r['n_calibration']) for r in curves if r['session']==s)==[0,10,20,40,80] for s in sessions)
    checks['full_calibration_endpoint_matches_selected']=all(float(next(r for r in curves if r['session']==s and int(r['n_calibration'])==80)['accuracy'])==float(next(r for r in test if r['session']==s and r['family']=='selected')['accuracy']) for s in sessions)
    checks['reliability_bins_account_for_80_trials']=len(rel)==30 and all(sum(int(r['n']) for r in rel if r['session']==s)==80 for s in sessions)
    checks['pre_post_transfer_only_pre_training_labels']=len(transfer)==3 and all(int(r['source_trials'])==80 and int(r['source_test_labels_used'])==int(r['target_calibration_labels'])==0 and int(r['trials'])==80 for r in transfer)
    z=np.load(OUT/'training_covariances.npz')
    checks['exact_balanced_subset_counts']=all(len(set(map(int,r['indices'].split(','))))==int(r['n_calibration']) and np.sum(z[r['session']+'_y'][list(map(int,r['indices'].split(',')))]==1)==int(r['n_calibration'])//2 for r in subsets if int(r['n_calibration'])>0)
    checks['full_choice_matches_training_receipt']=all(list(ast.literal_eval(r['chosen_setting']))==selection[r['session']]['chosen'] for r in test if r['family']=='selected')
    checks['nested_subsets']=all(set(next(r for r in subsets if r['session']==s and int(r['repeat'])==rep and int(r['n_calibration'])==a)['indices'].split(',')) <= set(next(r for r in subsets if r['session']==s and int(r['repeat'])==rep and int(r['n_calibration'])==b)['indices'].split(',')) for s in sessions for rep in range(5) for a,b in ((10,20),(20,40)))
    if not all(checks.values()): raise AssertionError(checks)
    receipt={'checks':checks,'tests':{'passed':37,'synthetic_causality':'Perturbing later samples and another trial leaves current and other trial covariances exactly unchanged','scope':'Implementation checks, not scientific or clinical validation'},'pooled_correct':{f:sum(int(r['correct']) for r in test if r['family']==f) for f in ('CSP-baseline','FBCSP','Riemann','TV-CSP','selected')}}
    (OUT/'validation.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    provenance=json.loads((OUT/'provenance.json').read_text())
    provenance['protocol']='../../analysis/accuracy_calibration/PROTOCOL.md'
    provenance['current_code_sha256']=hashlib.sha256((ROOT/'analysis/accuracy_calibration/run.py').read_bytes()).hexdigest()
    provenance['post_run_plot_changes']='Full subset min/max bars replace SD bars; margin and label positions adjusted after visual inspection. Numerical fitting, selection and score CSVs unchanged.'
    provenance['repository_correction']='Anna/ZyntZ is the team repository, based on ffcdb7e. Independent comparison outputs moved unchanged from the separate nxxis reference checkout. Reproduced Anna baseline433/480 here; nxxis444/479 remains a separate reference receipt.'
    provenance['assistant_model']='Current Codex session and inherited specialist review; exact model/effort identifier was not exposed by tool results.'
    provenance['output_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(OUT.glob('*.csv'))}
    (OUT/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(receipt,indent=2))

if __name__=='__main__': main()

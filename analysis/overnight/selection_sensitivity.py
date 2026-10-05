"""One prespecified training-only inner-split sensitivity; never load tests."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import numpy as np
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits

def temporal_inner(original_ids, labels, k=3, purge=1):
    """Return local positions, purging by original IDs across outer-fold holes."""
    ids=np.asarray(original_ids)
    y=np.asarray(labels)
    if len(ids)!=len(y) or len(ids)<k or np.any(np.diff(ids)<=0):
        raise ValueError('Expected sorted unique trial IDs and corresponding labels')
    folds=[]
    positions=np.arange(len(ids))
    for val in np.array_split(positions,k):
        distance=np.abs(ids[:,None]-ids[val][None,:]).min(axis=1)
        fit=positions[distance>purge]
        if set(y[fit])!={-1,1} or set(y[val])!={-1,1}:
            raise ValueError('Each training and validation partition needs both classes')
        folds.append((fit,val))
    return folds

def training_covariances(path):
    permitted={f'{p}_{s}_training.mat' for p in ('P1','P2','P3') for s in ('pre','post')}
    if Path(path).name not in permitted:
        raise ValueError('Sensitivity permits only the six declared training recordings')
    from analysis.accuracy_calibration import run as track
    return track.covariances(path)

def run(data,out):
    import scipy,sklearn,pyriemann
    from analysis.accuracy_calibration import run as track
    out.mkdir(parents=True,exist_ok=True)
    protocol=Path(__file__).with_name('LITERATURE_REVIEW.md')
    receipt=dict(source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        upstream_commit=subprocess.check_output(['git','rev-parse','origin/main'],cwd=ROOT,text=True).strip(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        protocol_sha256=hashlib.sha256(protocol.read_bytes()).hexdigest(),
        runtime=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,
                     sklearn=sklearn.__version__,pyriemann=pyriemann.__version__),
        scope='Single prespecified split sensitivity; full six-candidate selector; training files only',
        grid=track.GRID,outer_folds=5,purge_trials=1,inner_folds=3,
        window_seconds=[2.5,6.5],inner_seed=27,inputs=[],memberships=[],full_fit_choices={})
    rows=[];predictions=[]
    for session in track.SESSIONS:
        source=data/(session+'_training.mat')
        digest=hashlib.sha256(source.read_bytes()).hexdigest()
        receipt['inputs'].append(dict(file=source.name,sha256=digest))
        X,y=training_covariances(source)
        for kind in ('shuffled_inner','blocked_inner'):
            p=np.empty(80); choices=[]
            for number,(outer_fit,outer_val) in enumerate(track.blocked_folds()):
                inner=(temporal_inner(outer_fit,y[outer_fit]) if kind=='blocked_inner'
                       else list(StratifiedKFold(3,shuffle=True,random_state=27).split(X[outer_fit],y[outer_fit])))
                selected,scores=track.select(X[outer_fit],y[outer_fit],folds=inner)
                p[outer_val]=track.probability(track.Decoder(*selected).fit(X[outer_fit],y[outer_fit]),X[outer_val])
                choices.append(selected)
                receipt['memberships'].append(dict(session=session,protocol=kind,outer_fold=number,
                    outer_fit_ids=outer_fit.tolist(),outer_validation_ids=outer_val.tolist(),
                    chosen=selected,candidate_inner_accuracy=scores,
                    inner=[dict(fit_ids=outer_fit[a].tolist(),validation_ids=outer_fit[b].tolist(),
                                fit_class_counts={str(lab):int((y[outer_fit[a]]==lab).sum()) for lab in (-1,1)},
                                validation_class_counts={str(lab):int((y[outer_fit[b]]==lab).sum()) for lab in (-1,1)})
                           for a,b in inner]))
            full_inner=temporal_inner(np.arange(80),y) if kind=='blocked_inner' else None
            chosen,scores=track.select(X,y,folds=full_inner)
            receipt['full_fit_choices'][session+'_'+kind]=dict(chosen=chosen,candidate_inner_accuracy=scores)
            rows.append(dict(session=session,protocol=kind,outer_choices=json.dumps(choices),**track.metrics(y,p)))
            predictions.extend(dict(session=session,protocol=kind,trial=int(i),label=int(y[i]),p_left=float(p[i])) for i in range(80))
            print(session,kind,rows[-1]['correct'],'/80',flush=True)
        assert hashlib.sha256(source.read_bytes()).hexdigest()==digest
    # This is a replication check for the comparator, not a new test evaluation.
    import pandas as pd
    old=pd.read_csv(ROOT/'results/accuracy_calibration/nested_training_cv.csv')
    receipt['original_selected_training_comparisons']=[]
    for row in rows:
        if row['protocol']!='shuffled_inner':continue
        previous=old[(old.session==row['session'])&(old.family=='selected')].iloc[0]
        matches=all(abs(float(previous[key])-row[key])<1e-8
                    for key in ('correct','trials','accuracy','balanced_accuracy','brier','log_loss'))
        receipt['original_selected_training_comparisons'].append(dict(session=row['session'],matches=matches))
        assert matches,'Original comparator mismatch; stop before drawing a split conclusion'
    track.csv_write(out/'selection_sensitivity.csv',rows)
    track.csv_write(out/'selection_sensitivity_oof.csv',predictions)
    receipt['inputs_unchanged']=True
    receipt['io_policy']='Only declared training paths are permitted; source-audited, not an observed file counter'
    receipt['completed']=True
    destination=out/'selection_sensitivity.json'
    memberships=receipt.pop('memberships')
    text=json.dumps(receipt,indent=2)[:-2]+',\n  \"memberships\": [\n'
    text+=',\n'.join('    '+json.dumps(row,separators=(',',':')) for row in memberships)
    destination.write_text(text+'\n  ]\n}\n',encoding='utf-8')

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--data-dir',type=Path,required=True)
    parser.add_argument('--output',type=Path,default=ROOT/'results/overnight')
    args=parser.parse_args()
    with threadpool_limits(limits=1):run(args.data_dir.resolve(),args.output.resolve())

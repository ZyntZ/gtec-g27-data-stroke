"""Small, training-selected EEG comparison; exposed tests are exploratory.

Run from any directory with --data-dir and --output. No test file is opened
by the training stage. One independently filtered trial is one example.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import platform
import sys
from pathlib import Path
import numpy as np
import scipy
from scipy.io import loadmat
from scipy.signal import butter, sosfilt, sosfilt_zi, sosfiltfilt
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, brier_score_loss, log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from pyriemann.estimation import Covariances
from pyriemann.spatialfilters import CSP
from pyriemann.tangentspace import TangentSpace
import sklearn
import pyriemann

SESSIONS = [f'{p}_{s}' for p in ('P1','P2','P3') for s in ('pre','post')]
BANDS = ((8,12),(12,20),(20,30),(8,30))
WINDOWS = ((2.5,6.5),(2.5,3.5),(3.5,4.5),(4.5,5.5),(5.5,6.5))
GRID = [('FBCSP',2),('FBCSP',4),('Riemann',0.01),('Riemann',0.1),('TV-CSP',2),('TV-CSP',4)]
COUNTS = (0,10,20,40,80)

def dump(path, obj):
    Path(path).write_text(json.dumps(obj,indent=2)+'\n',encoding='utf-8')

def csv_write(path, rows):
    with Path(path).open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

def recording(path):
    m=loadmat(path,variable_names=('y','trig','fs'))
    signal=np.asarray(m['y'],float); trig=np.asarray(m['trig']).ravel(); fs=int(m['fs'].item())
    active=trig!=0
    on=np.flatnonzero(active & np.r_[True,~active[:-1]])
    off=np.flatnonzero(active & np.r_[~active[1:],True])+1
    y=trig[on].astype(int)
    if fs!=256 or signal.shape!=(len(trig),16) or not np.isfinite(signal).all():
        raise ValueError(f'Invalid signal: {path}')
    if len(on)!=80 or not np.all(off-on==8*fs) or not np.array_equal(np.unique(trig),[-1,0,1]):
        raise ValueError(f'Invalid trigger blocks: {path}')
    if any(not np.all(trig[a:b]==lab) for a,b,lab in zip(on,off,y)) or not all((y==k).sum()==40 for k in (-1,1)):
        raise ValueError(f'Invalid labels: {path}')
    return signal,on,y,fs

def covariances(path):
    signal,on,y,fs=recording(path)
    result=np.empty((80,len(BANDS),len(WINDOWS),16,16))
    # Filter state is reset per trial. No validation/test trial or post-window
    # EEG can enter a feature. Only samples from [0,6.5) s are read per trial.
    for b,band in enumerate(BANDS):
        sos=butter(4,band,btype='bandpass',fs=fs,output='sos')
        epochs=[]
        for t in on:
            raw=signal[t:t+int(6.5*fs)]
            filtered=sosfilt(sos,raw,axis=0,zi=sosfilt_zi(sos)[:,:,None]*raw[0][None,None,:])[0]
            epochs.append(filtered.T)
        epochs=np.stack(epochs)
        for w,(a,z) in enumerate(WINDOWS):
            result[:,b,w]=Covariances('oas').transform(epochs[:,:,int(a*fs):int(z*fs)])
    return result,y

class Decoder(ClassifierMixin,BaseEstimator):
    def __init__(self,family='FBCSP',setting=4):
        self.family=family; self.setting=setting
    def fit(self,X,y):
        self.classes_=np.unique(y); self.models_=[]
        bands=[3] if self.family=='CSP-baseline' else range(3)
        windows=range(1,5) if self.family=='TV-CSP' else [0]
        self.blocks_=[(b,w) for w in windows for b in bands]
        self.transforms_=[]; features=[]
        for b,w in self.blocks_:
            tr=TangentSpace(metric='riemann',tsupdate=False) if self.family=='Riemann' else CSP(nfilter=int(self.setting),log=True)
            features.append(tr.fit_transform(X[:,b,w],y)); self.transforms_.append(tr)
        if self.family=='TV-CSP':
            # A different CSP and LDA at each one-second decision time.
            for w in range(4):
                f=np.hstack(features[w*3:(w+1)*3])
                self.models_.append(make_pipeline(StandardScaler(),LinearDiscriminantAnalysis(solver='lsqr',shrinkage='auto')).fit(f,y))
        else:
            clf=LogisticRegression(C=float(self.setting),max_iter=2000) if self.family=='Riemann' else LinearDiscriminantAnalysis(solver='lsqr',shrinkage='auto')
            self.models_.append(make_pipeline(StandardScaler(),clf).fit(np.hstack(features),y))
        return self
    def predict_proba(self,X):
        f=[tr.transform(X[:,b,w]) for tr,(b,w) in zip(self.transforms_,self.blocks_)]
        if self.family=='TV-CSP':
            return np.mean([m.predict_proba(np.hstack(f[i*3:(i+1)*3])) for i,m in enumerate(self.models_)],axis=0)
        return self.models_[0].predict_proba(np.hstack(f))
    def predict(self,X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]

def blocked_folds(n=80,k=5,purge=1):
    allidx=np.arange(n)
    for val in np.array_split(allidx,k):
        fit=allidx[(allidx<val[0]-purge)|(allidx>val[-1]+purge)]
        yield fit,val

def probability(model,X):
    return model.predict_proba(X)[:,list(model.classes_).index(1)]

def metrics(y,p):
    pred=np.where(p>=.5,1,-1)
    return dict(correct=int((pred==y).sum()),trials=len(y),accuracy=float(accuracy_score(y,pred)),
                balanced_accuracy=float(balanced_accuracy_score(y,pred)),
                brier=float(brier_score_loss(y==1,p)),log_loss=float(log_loss(y==1,np.c_[1-p,p],labels=[False,True])))

def select(X,y,grid=GRID,folds=None):
    if folds is None:
        k=2 if len(y)<=10 else 3
        folds=list(StratifiedKFold(k,shuffle=True,random_state=27).split(X,y))
    scores=[]
    for family,setting in grid:
        p=np.empty(len(y))
        for fit,val in folds:
            p[val]=probability(Decoder(family,setting).fit(X[fit],y[fit]),X[val])
        scores.append(metrics(y,p)['accuracy'])
    # Fixed order wins ties. Test data never participate in this choice.
    best=int(np.argmax(scores))
    return grid[best],scores

def reproduce_baseline(data,out):
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
    from stroke_rehab.experiment import evaluate_all
    results=evaluate_all(data,out/'baseline_reproduction',seed=27,n_splits=5)
    with (Path(__file__).resolve().parents[2]/'results/session_results.csv').open() as f:
        previous={r['patient']+'_'+r['session']:r for r in csv.DictReader(f)}
    rows=[]
    for r in results:
        session=r['patient']+'_'+r['session']; old=previous[session]
        match=all(str(r[k])==old[k] if isinstance(r[k],str) else abs(float(r[k])-float(old[k]))<1e-10 for k in r)
        row=dict(session=session,train_trials=r['train_trials'],correct=r['correct_test_trials'],trials=r['test_trials'],accuracy=r['test_accuracy'],previous_accuracy=float(old['test_accuracy']),matches_previous=match)
        rows.append(row); print('TEAM BASELINE',session,row['correct'],row['trials'],match,flush=True)
    csv_write(out/'baseline_reproduction.csv',rows)
    return rows

def train_stage(data,out):
    cache={}; selection={}; rows=[]; predictions=[]
    for session in SESSIONS:
        X,y=covariances(data/f'{session}_training.mat'); cache[session]=(X,y)
        full,scores=select(X,y)
        selection[session]={'chosen':full,'grid':GRID,'cv_accuracy':scores,'families':{}}
        for family in ('CSP-baseline','FBCSP','Riemann','TV-CSP','selected'):
            grid=[('CSP-baseline',4)] if family=='CSP-baseline' else GRID if family=='selected' else [g for g in GRID if g[0]==family]
            p=np.empty(80); outer_choices=[]
            for fit,val in blocked_folds():
                choice,_=select(X[fit],y[fit],grid)
                p[val]=probability(Decoder(*choice).fit(X[fit],y[fit]),X[val]); outer_choices.append(choice)
            chosen,_=select(X,y,grid)
            selection[session]['families'][family]=chosen
            rows.append(dict(session=session,family=family,full_fit_setting=str(chosen),outer_fit_settings=json.dumps(outer_choices),**metrics(y,p)))
            predictions.extend(dict(session=session,family=family,trial=int(i),label=int(y[i]),p_left=float(p[i])) for i in range(80))
        print('TRAIN',session,'selected',full,flush=True)
    # Immutable selection receipt is written before the test stage starts.
    csv_write(out/'nested_training_cv.csv',rows); csv_write(out/'training_oof_predictions.csv',predictions)
    dump(out/'selection_training_only.json',selection)
    return cache,selection

def balanced_order(y,seed):
    rng=np.random.default_rng(seed)
    a=rng.permutation(np.flatnonzero(y==-1)); b=rng.permutation(np.flatnonzero(y==1))
    return np.c_[a,b].ravel()

def evaluate_stage(data,out,cache,selection,repeats=5):
    test={}; rows=[]; curves=[]; detail=[]; reliability=[]; transfer=[]
    for session in SESSIONS:
        test[session]=covariances(data/f'{session}_test.mat')
        X,y=cache[session]; T,yt=test[session]
        for family,choice in selection[session]['families'].items():
            m=Decoder(*choice).fit(X,y); p=probability(m,T)
            rows.append(dict(session=session,family=family,chosen_setting=str(choice),train_trials=len(y),**metrics(yt,p)))
            if family=='selected':
                for lo in np.arange(0,1,.2):
                    keep=(p>=lo)&(p<(lo+.2) if lo<.8 else p<=1)
                    reliability.append(dict(session=session,bin_low=float(lo),bin_high=float(lo+.2),n=int(keep.sum()),
                        mean_probability=float(p[keep].mean()) if keep.any() else '',fraction_left=float((yt[keep]==1).mean()) if keep.any() else ''))
        for n in COUNTS:
            sources=[s for s in SESSIONS if s[:2]!=session[:2]]
            if n==0:
                # Zero calibration excludes BOTH sessions of the target person.
                Z=np.concatenate([cache[s][0] for s in sources]); zy=np.concatenate([cache[s][1] for s in sources])
                # Leave-one-source-participant-out selection; the validation
                # participant's other session must also be excluded from fit.
                folds=[]
                for patient in sorted({s[:2] for s in sources}):
                    val=np.concatenate([np.arange(i*80,(i+1)*80) for i,s in enumerate(sources) if s[:2]==patient])
                    folds.append((np.setdiff1d(np.arange(320),val),val))
                choice,_=select(Z,zy,folds=folds); model=Decoder(*choice).fit(Z,zy)
                p=probability(model,T); ms=metrics(yt,p); detail.append(dict(session=session,n_calibration=0,repeat=0,fit_trials=320,chosen_setting=str(choice),indices='',**ms))
                curves.append(dict(session=session,n_calibration=0,repeats=1,accuracy=ms['accuracy'],subset_sd=0.,minimum=ms['accuracy'],maximum=ms['accuracy'],test_trials=80,brier=ms['brier']))
                continue
            acc=[]; bs=[]
            for repeat in range(1 if n==80 else repeats):
                idx=np.arange(80) if n==80 else balanced_order(y,27+repeat)[:n]
                choice=selection[session]['chosen'] if n==80 else select(X[idx],y[idx])[0]
                p=probability(Decoder(*choice).fit(X[idx],y[idx]),T); ms=metrics(yt,p)
                acc.append(ms['accuracy']); bs.append(ms['brier'])
                detail.append(dict(session=session,n_calibration=n,repeat=repeat,fit_trials=n,chosen_setting=str(choice),indices=','.join(map(str,idx)),**ms))
            curves.append(dict(session=session,n_calibration=n,repeats=len(acc),accuracy=float(np.mean(acc)),subset_sd=float(np.std(acc,ddof=1)) if len(acc)>1 else 0.,minimum=min(acc),maximum=max(acc),test_trials=80,brier=float(np.mean(bs))))
        print('TEST/CALIBRATION',session,flush=True)
    for patient in ('P1','P2','P3'):
        source=patient+'_pre'; target=patient+'_post'
        X,y=cache[source]; choice=selection[source]['chosen']; T,yt=test[target]
        transfer.append(dict(patient=patient,direction='PRE training -> POST test',source_trials=80,source_test_labels_used=0,target_calibration_labels=0,chosen_setting=str(choice),**metrics(yt,probability(Decoder(*choice).fit(X,y),T))))
    csv_write(out/'exploratory_test_accuracy.csv',rows); csv_write(out/'calibration_curves.csv',curves)
    csv_write(out/'calibration_subsets.csv',detail); csv_write(out/'probability_reliability.csv',reliability)
    csv_write(out/'pre_post_transfer.csv',transfer)
    return rows,curves,transfer

def plots(out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    def read(name): return list(csv.DictReader((out/name).open()))
    curves=read('calibration_curves.csv'); rel=read('probability_reliability.csv')
    fig,axs=plt.subplots(2,3,figsize=(12,6.6),sharex=True,sharey=True)
    for ax,s in zip(axs.flat,SESSIONS):
        rr=[r for r in curves if r['session']==s]
        x=[int(r['n_calibration']) for r in rr]; y=[100*float(r['accuracy']) for r in rr]
        spread=np.array([[v-100*float(r['minimum']) for v,r in zip(y,rr)],
                         [100*float(r['maximum'])-v for v,r in zip(y,rr)]])
        ax.errorbar(x,y,yerr=spread,fmt='o-',capsize=3,color='#176b91'); ax.axhline(50,color='.65',ls='--',lw=1)
        ax.set(title=s.replace('_',' ').upper(),xlim=(0,80),ylim=(0,100),xticks=COUNTS); ax.grid(alpha=.15)
    fig.supxlabel('Target-session labelled calibration trials',y=.065); fig.supylabel('Exploratory test accuracy (%)')
    fig.suptitle('Retrospective label budgets: 80 exposed test trials',fontsize=13)
    fig.text(.5,.035,'10–40: five nested random, class-balanced subsets from the full run; not chronological prefixes.',ha='center',fontsize=9)
    fig.text(.5,.012,'Bars: subset minimum–maximum, not confidence intervals. 0: 320 source trials. 80: full fit. Same exposed tests.',ha='center',fontsize=9)
    fig.tight_layout(rect=(.02,.12,1,.94))
    for ext in ('png','svg'): fig.savefig(out/f'calibration_curves.{ext}',dpi=180)
    plt.close(fig)
    fig,axs=plt.subplots(2,3,figsize=(10,6.5),sharex=True,sharey=True)
    for ax,s in zip(axs.flat,SESSIONS):
        rr=[r for r in rel if r['session']==s and int(r['n'])]
        ax.plot([0,1],[0,1],color='.6',ls='--'); ax.plot([float(r['mean_probability']) for r in rr],[float(r['fraction_left']) for r in rr],'o-',color='#176b91')
        for r in rr:
            px=float(r['mean_probability']); py=float(r['fraction_left'])
            ax.annotate(r['n'],(px,py),xytext=(-16 if px>.85 else 4,-12 if py>.9 else 5),textcoords='offset points',fontsize=8)
        ax.set(title=s.replace('_',' ').upper(),xlim=(0,1),ylim=(0,1)); ax.grid(alpha=.15)
    fig.supxlabel('Mean predicted P(left)',y=.065); fig.supylabel('Observed left-trial fraction'); fig.suptitle('Probability reliability: training-selected model, exposed tests',fontsize=12)
    fig.text(.5,.018,'Five fixed bins; numbers are trial counts. Descriptive curves; no probability calibrator was fitted.',ha='center',fontsize=9)
    fig.tight_layout(rect=(.02,.12,1,.94))
    for ext in ('png','svg'): fig.savefig(out/f'probability_reliability.{ext}',dpi=180)
    plt.close(fig)
    # Matplotlib leaves spaces in multiline SVG path attributes; keep exports
    # diff-clean without changing their geometry or pixel rendering.
    for path in out.glob('*.svg'):
        path.write_text('\n'.join(line.rstrip() for line in path.read_text(encoding='utf-8').splitlines())+'\n',encoding='utf-8')

def main():
    p=argparse.ArgumentParser(); p.add_argument('--data-dir',type=Path,required=True); p.add_argument('--output',type=Path,required=True); p.add_argument('--stage',choices=['all','train','evaluate','baseline','plot'],default='all'); args=p.parse_args()
    out=args.output; out.mkdir(parents=True,exist_ok=True)
    with threadpool_limits(limits=1):
        if args.stage in ('all','baseline'): reproduce_baseline(args.data_dir,out)
        if args.stage in ('all','train'):
            cache,selection=train_stage(args.data_dir,out)
            np.savez_compressed(out/'training_covariances.npz',**{s+'_X':v[0] for s,v in cache.items()},**{s+'_y':v[1] for s,v in cache.items()})
        if args.stage in ('all','evaluate'):
            if args.stage=='evaluate':
                z=np.load(out/'training_covariances.npz'); cache={s:(z[s+'_X'],z[s+'_y']) for s in SESSIONS}; selection=json.loads((out/'selection_training_only.json').read_text())
            evaluate_stage(args.data_dir,out,cache,selection)
        if args.stage in ('all','plot'): plots(out)
    if args.stage=='all':
        dump(out/'provenance.json',{'protocol':'PROTOCOL.md','date':'2026-10-04','python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,'sklearn':sklearn.__version__,'pyriemann':pyriemann.__version__,
             'selection_receipt_sha256':hashlib.sha256((out/'selection_training_only.json').read_bytes()).hexdigest(),
             'files':[{'name':q.name,'sha256':hashlib.sha256(q.read_bytes()).hexdigest()} for q in sorted(args.data_dir.glob('*.mat'))],
             'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})

if __name__=='__main__': main()

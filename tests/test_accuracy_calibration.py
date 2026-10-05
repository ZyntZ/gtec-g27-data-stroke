"""Check causality and trial-budget isolation, rather than copying scores."""
import importlib.util
from pathlib import Path
import numpy as np
from scipy.io import savemat
import pytest

pytest.importorskip('pyriemann', reason='Install the optional accuracy extra to test these models')

spec=importlib.util.spec_from_file_location('accuracy_track',Path(__file__).parents[1]/'analysis/accuracy_calibration/run.py')
track=importlib.util.module_from_spec(spec); spec.loader.exec_module(track)

def test_future_and_other_trial_samples_cannot_change_features(tmp_path):
    fs=256; length=80*9*fs
    rng=np.random.default_rng(13); signal=rng.normal(size=(length,16)); trig=np.zeros(length)
    for i in range(80): trig[i*9*fs:i*9*fs+8*fs]=1 if i%2 else -1
    base=tmp_path/'a.mat'; changed=tmp_path/'b.mat'
    savemat(base,dict(fs=fs,y=signal,trig=trig))
    # Alter trial 0 after decision time and every sample of trial 1.
    signal[int(6.5*fs):9*fs]*=1e6; signal[9*fs:18*fs]*=1e6
    savemat(changed,dict(fs=fs,y=signal,trig=trig))
    A,y=track.covariances(base); B,z=track.covariances(changed)
    np.testing.assert_array_equal(A[0],B[0])
    np.testing.assert_array_equal(A[2:],B[2:])
    np.testing.assert_array_equal(y,z)
    assert not np.allclose(A[1],B[1])

def test_budget_subsets_are_balanced_nested_and_reproducible():
    y=np.r_[np.ones(40),-np.ones(40)]
    order=track.balanced_order(y,27)
    assert len(np.unique(order))==80
    np.testing.assert_array_equal(order,track.balanced_order(y,27))
    for n in (10,20,40,80):
        assert (y[order[:n]]==1).sum()==n//2
        assert (y[order[:n]]==-1).sum()==n//2

def test_chronological_outer_folds_keep_trials_and_purge_neighbours():
    folds=list(track.blocked_folds())
    np.testing.assert_array_equal(np.concatenate([v for _,v in folds]),np.arange(80))
    assert [len(f) for f,_ in folds]==[63,62,62,62,63]
    for fit,val in folds:
        assert not np.intersect1d(fit,val).size
        assert np.min(np.abs(fit[:,None]-val[None,:]))>=2

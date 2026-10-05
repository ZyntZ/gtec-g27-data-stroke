"""Temporal separation must survive missing outer-validation trial IDs."""
import numpy as np
import pytest
from analysis.overnight.selection_sensitivity import temporal_inner,training_covariances

def test_purge_uses_original_ids_across_a_hole():
    ids=np.r_[np.arange(20),np.arange(40,80)]
    y=np.where(ids%2,1,-1)
    folds=temporal_inner(ids,y)
    np.testing.assert_array_equal(np.concatenate([v for _,v in folds]),np.arange(len(ids)))
    for fit,val in folds:
        assert np.min(np.abs(ids[fit,None]-ids[val][None,:]))>1
        assert set(y[fit])=={-1,1}
    # Trial40 is twenty-one trial IDs away from trial19, not its adjacent trial.
    fit,val=folds[0]
    assert 40 in ids[fit] and 19 in ids[val]

def test_single_class_partitions_fail_explicitly():
    with pytest.raises(ValueError,match='both classes'):
        temporal_inner(np.arange(80),np.r_[np.ones(60),-np.ones(20)])

def test_exposed_test_is_rejected_before_any_read(tmp_path):
    with pytest.raises(ValueError,match='training recordings'):
        training_covariances(tmp_path/'P1_pre_test.mat')

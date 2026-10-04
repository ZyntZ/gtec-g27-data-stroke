"""A zero CLI exit must not conceal a failed numerical comparison."""
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from analysis.overnight import reproduce

TABLES=('data_inventory.csv','session_results.csv','timing_audit.csv',
        'train_diagnostics.csv','causal_train_cv.csv','causal_nested_summary.csv',
        'causal_nested_folds.csv','causal_latency_audit.csv','training_negative_controls.csv')

@pytest.mark.parametrize('case',('match','mismatch','missing_reference'))
def test_successful_process_still_requires_matching_saved_reference(tmp_path,monkeypatch,case):
    data=tmp_path/'organizer';data.mkdir()
    for i in range(12):(data/f'{i}.mat').write_bytes(b'synthetic placeholder')
    references=tmp_path/'results';references.mkdir()
    for name in TABLES:(references/name).write_text('value\n1\n')
    if case=='missing_reference':(references/TABLES[0]).unlink()
    monkeypatch.setattr(reproduce,'ROOT',tmp_path)
    monkeypatch.setattr(reproduce.subprocess,'check_output',lambda *a,**k:'snapshot\n')
    calls=[]
    def fake_cli(command,**kwargs):
        calls.append(command)
        output=tmp_path/'data/overnight-reproduction'
        for name in TABLES:
            value=2 if case=='mismatch' and name==TABLES[0] else 1
            (output/name).write_text(f'value\n{value}\n')
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(reproduce.subprocess,'run',fake_cli)
    if case=='match':reproduce.run(data)
    else:
        with pytest.raises(RuntimeError,match='numerical verification failed'):
            reproduce.run(data)
    receipt=json.loads((references/'overnight/reproduction.json').read_text())
    if case=='match':
        assert receipt['finished_stages']==9 and len(calls)==9
        assert all(s['verification_passed'] for s in receipt['stages'])
    else:
        stage=receipt['stages'][0]
        assert stage['exit_code']==0 and stage['status']=='failed'
        assert stage['verification_passed'] is False and len(calls)==1
        assert stage['comparisons'][0]['status']==('mismatch' if case=='mismatch' else 'no_committed_reference')

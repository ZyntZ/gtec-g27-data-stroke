"""Execute unchanged notebook sources in an ignored replica; check own exports."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_physiology_csvs(output, reference_manifest):
    """Fail closed when output-free notebooks have no tracked reference CSVs."""
    expected = {'mu_power/all_training_trial_mu.csv',
                'mu_power/p1_pre_training_trial_mu.csv',
                'mu_power/training_mu_summary.csv'}
    refs = reference_manifest['csv_checks']
    if len(refs) != len(expected) or {r['file'] for r in refs} != expected:
        raise ValueError('Expected all three frozen physiology references')
    checks = []
    for reference in refs:
        actual = Path(output) / reference['file']
        actual_hash = sha(actual)
        if actual_hash != reference['reference_sha256']:
            raise RuntimeError(f"Physiology CSV mismatch: {reference['file']}")
        checks.append(dict(file=reference['file'], status='match',
                           reference_sha256=reference['reference_sha256'],
                           actual_sha256=actual_hash,
                           reference_source=reference_manifest['source_commit']))
    return checks

def run(data):
    import jsonschema
    import pandas as pd
    from jupyter_client import KernelManager

    ignored = ROOT / 'data/overnight-reproduction'
    sandbox = ignored / 'notebook-sandbox'
    sandbox.mkdir(parents=True, exist_ok=True)
    for folder in ('stroke_rehab', 'analysis/mu_power', 'notebooks'):
        if (ROOT / folder).exists():
            shutil.copytree(ROOT / folder, sandbox / folder, dirs_exist_ok=True)
    output = sandbox / 'results'
    output.mkdir(exist_ok=True)
    for file in ignored.glob('*.csv'):
        shutil.copy2(file, output / file.name)
    schema = json.loads(urllib.request.urlopen(
        'https://raw.githubusercontent.com/jupyter/nbformat/main/nbformat/v4/nbformat.v4.schema.json',
        timeout=30).read())
    receipt = dict(source_commit=subprocess.check_output(
        ['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip(),
        upstream_commit=subprocess.check_output(['git','rev-parse','origin/main'],
                                              cwd=ROOT,text=True).strip(),
        method='Unchanged notebook sources; replica cwd and STROKE_REHAB_DATA only',
        physiology_notebook_present=(ROOT / 'notebooks/03_mu_power.ipynb').is_file(),
        notebooks=[], csv_checks=[], local_models=[])
    dest = ROOT / 'results/overnight/artifact_checks.json'
    def save():
        dest.write_text(json.dumps(receipt, indent=2)+'\n', encoding='utf-8')
    save()
    for source in sorted((ROOT / 'notebooks').glob('*.ipynb')):
        book = json.loads(source.read_text(encoding='utf-8'))
        jsonschema.validate(book, schema)
        for cell in book['cells']:
            if cell['cell_type'] == 'code':
                cell['outputs'] = []
                cell['execution_count'] = None
        manager = KernelManager(kernel_name='python3',
                                connection_file=str(sandbox / '.analysis-kernel.json'))
        manager.kernel_spec.argv[0] = sys.executable
        environment = dict(os.environ, STROKE_REHAB_DATA=str(data),
                           PYTHONDONTWRITEBYTECODE='1',
                           MPLCONFIGDIR=str(ignored / '.matplotlib-cache'))
        client = None
        started = time.monotonic()
        try:
            manager.start_kernel(cwd=str(sandbox), env=environment)
            client = manager.client()
            client.start_channels()
            client.wait_for_ready(timeout=30)
            for i, cell in enumerate(book['cells']):
                if cell['cell_type'] != 'code':
                    continue
                msg = client.execute(''.join(cell['source']))
                while True:
                    message = client.get_iopub_msg(timeout=90)
                    if message['parent_header'].get('msg_id') != msg:
                        continue
                    kind, value = message['msg_type'], message['content']
                    if kind == 'execute_input':
                        cell['execution_count'] = value['execution_count']
                    elif kind == 'stream':
                        cell['outputs'].append(dict(output_type=kind, name=value['name'],text=value['text']))
                    elif kind in ('display_data','execute_result'):
                        item = dict(output_type=kind,data=value['data'],metadata=value['metadata'])
                        if kind == 'execute_result': item['execution_count'] = value['execution_count']
                        cell['outputs'].append(item)
                    elif kind == 'error':
                        raise RuntimeError(f'{source.name} cell {i}: {value["ename"]}: {value["evalue"]}')
                    elif kind == 'status' and value['execution_state'] == 'idle':
                        break
            jsonschema.validate(book,schema)
            target = sandbox / 'notebooks' / source.name
            target.write_text(json.dumps(book,indent=1)+'\n',encoding='utf-8')
            receipt['notebooks'].append(dict(file=source.name,source_sha256=sha(source),
                executed_copy_sha256=sha(target),code_cells=sum(c['cell_type']=='code' for c in book['cells']),
                errors=0,status='pass',seconds=round(time.monotonic()-started,3)))
            save()
            print('PASS notebook',source.name,flush=True)
        finally:
            if client is not None: client.stop_channels()
            if manager.has_kernel: manager.shutdown_kernel(now=True)
            manager.cleanup_connection_file()
    references = [ROOT / 'results/train_signal_qc.csv']
    for reference in references:
        actual = output / reference.relative_to(ROOT / 'results')
        pd.testing.assert_frame_equal(pd.read_csv(actual),pd.read_csv(reference),
                                      check_exact=False,rtol=1e-10,atol=1e-10,check_dtype=False)
        receipt['csv_checks'].append(dict(file=str(reference.relative_to(ROOT / 'results')).replace('\\','/'),
            status='match',rows=len(pd.read_csv(actual)),reference_sha256=sha(reference),actual_sha256=sha(actual)))
    if receipt['physiology_notebook_present']:
        manifest = json.loads((ROOT / 'results/overnight/mu_reference_hashes.json').read_text())
        receipt['csv_checks'].extend(verify_physiology_csvs(output, manifest))
    save()
    import joblib
    import numpy as np
    from stroke_rehab.data import read_recording
    from stroke_rehab.spatial import CausalSpatialDecoder, replay_covariances
    from stroke_rehab.nested import input_for
    models = ignored / 'models'
    assert len(list(models.glob('*.joblib'))) == len(list(models.glob('*.json'))) == 6
    for path in sorted(models.glob('*.json')):
        meta = json.loads(path.read_text())
        source = data / f'{path.stem}_training.mat'
        assert sha(source) == meta['training_sha256']
        assert meta['fs_hz']==256 and meta['n_channels']==16 and meta['window_seconds']==[2.5,3.5]
        fitted = joblib.load(path.with_suffix('.joblib'))  # Our freshly created, trusted export only.
        rec = read_recording(source)
        cov, offsets = replay_covariances(rec,chunk_samples=meta['chunk_samples'],window=tuple(meta['window_seconds']))
        expected = fitted.predict(input_for(meta['candidate'],cov))
        decoder = CausalSpatialDecoder(fitted,candidate=meta['candidate'],window=tuple(meta['window_seconds']))
        decisions = []
        for left in range(0,len(rec.signal),meta['chunk_samples']):
            right = min(left+meta['chunk_samples'],len(rec.signal))
            onsets = rec.onsets[(rec.onsets>=left)&(rec.onsets<right)]
            event = decoder.process(rec.signal[left:right],onsets=list(map(int,onsets)))
            if event is not None: decisions.append(event)
        assert len(decisions)==80
        np.testing.assert_array_equal([d.onset_sample for d in decisions],rec.onsets)
        np.testing.assert_array_equal([d.label for d in decisions],expected)
        np.testing.assert_array_equal([d.decision_sample-d.onset_sample for d in decisions],offsets)
        assert set(expected) <= {-1,1}
        receipt['local_models'].append(dict(session=path.stem,candidate=meta['candidate'],
            model_sha256=sha(path.with_suffix('.joblib')),metadata_sha256=sha(path),decisions=80,
            max_decision_seconds=float(offsets.max()/rec.fs),stream_matches_batch=True,status='pass'))
        save()
        print('PASS local model',path.stem,flush=True)
    receipt['scope']='Execution and export correctness; no generalization score or scientific efficacy claim'
    save()

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--data-dir',type=Path,required=True)
    run(parser.parse_args().data_dir.resolve())

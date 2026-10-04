"""Reproduce the frozen team pipeline in ignored output folders."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(data_dir):
    import numpy, scipy, sklearn, pandas

    output = ROOT / 'data' / 'overnight-reproduction'
    output.mkdir(parents=True, exist_ok=True)
    receipts = ROOT / 'results' / 'overnight'
    receipts.mkdir(parents=True, exist_ok=True)
    report = {
        'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'upstream_commit': subprocess.check_output(['git', 'rev-parse', 'origin/main'], cwd=ROOT, text=True).strip(),
        'scope': 'Frozen pipeline reproduction only; no new settings selected on exposed tests',
        'runtime': {'python': sys.version.split()[0], 'numpy': numpy.__version__,
                    'scipy': scipy.__version__, 'sklearn': sklearn.__version__, 'pandas': pandas.__version__},
        'inputs': [{'file': path.name, 'sha256': digest(path)} for path in sorted(data_dir.glob('*.mat'))],
        'stages': [],
    }
    assert len(report['inputs']) == 12
    destination = receipts / 'reproduction.json'

    def save():
        destination.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')

    stages = [
        ('inventory', ['inventory', '--output', str(output / 'data_inventory.csv')], ['data_inventory.csv']),
        ('baseline', ['run', '--output-dir', str(output)], ['session_results.csv']),
        ('timing_audit', ['audit', '--output', str(output / 'timing_audit.csv')], ['timing_audit.csv']),
        ('diagnostics', ['diagnose', '--output', str(output / 'train_diagnostics.csv')], ['train_diagnostics.csv']),
        ('causal_replay', ['causal-train', '--output', str(output / 'causal_train_cv.csv')], ['causal_train_cv.csv']),
        ('nested_spatial', ['nested-train', '--output-dir', str(output)], ['causal_nested_summary.csv', 'causal_nested_folds.csv']),
        ('latency_audit', ['latency-audit', '--output', str(output / 'causal_latency_audit.csv')], ['causal_latency_audit.csv']),
        ('negative_controls', ['controls', '--output', str(output / 'training_negative_controls.csv')], ['training_negative_controls.csv']),
        ('local_calibration_models', ['calibrate-spatial', '--output-dir', str(output / 'models')], []),
    ]
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE='1',
                       MPLCONFIGDIR=str(output / '.matplotlib-cache'),
                       OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    save()
    for name, args, tables in stages:
        print(f'Starting {name}', flush=True)
        started = time.monotonic()
        command = [sys.executable, '-m', 'stroke_rehab.cli', *args, '--data-dir', str(data_dir)]
        record = {'name': name, 'status': 'running', 'command': ['python', '-m', 'stroke_rehab.cli', args[0], '--data-dir', '<local organiser data>']}
        report['stages'].append(record)
        save()
        try:
            with (output / f'{name}.log').open('w', encoding='utf-8') as stream:
                result = subprocess.run(command, cwd=ROOT, env=environment, stdout=stream,
                                        stderr=subprocess.STDOUT, timeout=5400)
            record['exit_code'] = result.returncode
            record['seconds'] = round(time.monotonic() - started, 3)
            if result.returncode:
                record['status'] = 'failed'
                save()
                raise RuntimeError(f'{name} failed; inspect ignored local log {name}.log')
            record['comparisons'] = []
            for filename in tables:
                reference = ROOT / 'results' / filename
                reproduced = output / filename
                check = {'file': filename, 'reproduced_sha256': digest(reproduced)}
                if not reference.exists():
                    check['status'] = 'no_committed_reference'
                else:
                    expected = pandas.read_csv(reference)
                    actual = pandas.read_csv(reproduced)
                    try:
                        pandas.testing.assert_frame_equal(actual, expected, check_exact=False,
                                                          rtol=1e-10, atol=1e-10, check_dtype=False)
                        check['status'] = 'match'
                    except AssertionError as error:
                        check['status'] = 'mismatch'
                        check['detail'] = str(error)[:1200].replace(str(data_dir), '<local organiser data>')
                    check['reference_sha256'] = digest(reference)
                    check['rows'] = len(actual)
                record['comparisons'].append(check)
            failures = [check for check in record['comparisons'] if check['status'] != 'match']
            record['verification_passed'] = not failures
            if failures:
                save()
                detail = ', '.join(f'{check["file"]}: {check["status"]}' for check in failures)
                raise RuntimeError(f'{name} executed successfully but numerical verification failed: {detail}')
            record['status'] = 'complete'
            save()
            print(f'Finished {name}: {record["seconds"]} seconds; '
                  f'{[(check["file"], check["status"]) for check in record["comparisons"]]}', flush=True)
        except Exception as error:
            record['status'] = 'failed'
            record['error'] = str(error)
            save()
            raise
    assert all(digest(data_dir / source['file']) == source['sha256'] for source in report['inputs'])
    report['inputs_unchanged'] = True
    report['finished_stages'] = len(stages)
    save()
    print('Frozen pipeline CSV verification passed. Review notebooks and model exports separately.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=Path, required=True)
    args = parser.parse_args()
    run(args.data_dir)

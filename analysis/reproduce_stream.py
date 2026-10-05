"""Reproduce the frozen training audit and verify covariance-stream parity."""
import argparse, csv, hashlib, json, pathlib, sys, datetime
import numpy as np
root=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root))
from stroke_rehab.forward_calibration import run_forward_calibration, training_files
from stroke_rehab.comparison import _inputs, _design, MODELS
from stroke_rehab.data import read_recording
from stroke_rehab.riemann import RecenteredTangentSpace
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--data-dir',type=pathlib.Path,required=True)
data=parser.parse_args().data_dir
out=root/'data/stream-reproduction'
pred,summary=run_forward_calibration(data,out)
reference=root/'results/forward_calibration'
def read(p):
 with p.open(newline='',encoding='utf8') as s: return list(csv.DictReader(s))
for name,count in [('forward_calibration_predictions.csv',3840),('forward_calibration_summary.csv',192)]:
 a,b=read(out/name),read(reference/name)
 assert len(a)==len(b)==count
 for x,y in zip(a,b,strict=True):
  assert x.keys()==y.keys()
  for k in x:
   try: assert np.isclose(float(x[k]),float(y[k]),rtol=0,atol=1e-12)
   except ValueError: assert x[k]==y[k]
 print('MATCH',name,count,flush=True)
checks=[]
for patient,stage,path in training_files(data):
 rec=read_recording(path)
 x=_design(MODELS[1],_inputs(rec,causal=True)[MODELS[1]],rec.fs,(2.5,3.5))
 for adapt in (True,False):
  fitted=RecenteredTangentSpace(adapt=adapt).fit(x[:60],rec.labels[:60])
  batch=fitted.predict(x[60:]); scores=fitted.decision_function(x[60:])
  fitted.reset(); singles=np.array([fitted.predict_next(row) for row in x[60:]])
  single_refs=[r.copy() for r in fitted.references_]
  fitted.reset(); chunks=np.concatenate([fitted.predict_stream(x[60:63]),fitted.predict_stream(x[63:70]),fitted.predict_stream(x[70:])])
  np.testing.assert_array_equal(batch,singles)
  np.testing.assert_array_equal(singles,chunks)
  for a,b in zip(single_refs,fitted.references_,strict=True): np.testing.assert_allclose(a,b,rtol=0,atol=1e-12)
  assert fitted.n_seen_==20
  fitted.reset(); sequential_scores=np.array([fitted.decision_next(row) for row in x[60:]])
  np.testing.assert_allclose(scores,sequential_scores,rtol=1e-10,atol=1e-12)
  check=dict(patient=patient,session=stage,adapt=adapt,trials=20,batch_single_partition_labels_match=True,final_references_match=True,decision_scores_match=True)
  if adapt:
   from stroke_rehab.riemann_stream import replay_decisions
   expected=fitted.predict(x)
   decisions=replay_decisions(rec,fitted,chunk_samples=64,window=(2.5,3.5))
   np.testing.assert_array_equal(expected,[d.label for d in decisions])
   assert len(decisions)==80
   offsets=np.array([d.decision_sample-d.onset_sample for d in decisions])
   assert np.all(offsets>=896) and np.all(offsets<960)
   check.update(raw_training_replay_labels_match=True,raw_training_trials=80,chunk_samples=64,latest_chunk_decision_s=float(offsets.max()/256))
  checks.append(check)
 print('STREAM MATCH',patient,stage,flush=True)
historical=json.loads((reference/'forward_calibration_provenance.json').read_text())
new=json.loads((out/'forward_calibration_provenance.json').read_text())
assert historical['input_sha256']==new['input_sha256']
receipt=dict(checked_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scope='Frozen historical scores reproduced; covariance API parity, not clinical validation',
 prediction_rows_matched=3840,summary_rows_matched=192,input_sha256=new['input_sha256'],historical_source_sha256=historical['source_sha256'],
 source_sha256={n:hashlib.sha256((root/'stroke_rehab'/n).read_bytes().replace(b'\r\n',b'\n')).hexdigest() for n in historical['source_sha256']},source_hash_normalization='CRLF to LF; raw recording hashes unchanged',additional_source_sha256_lf={n:hashlib.sha256((root/n).read_bytes().replace(b'\r\n',b'\n')).hexdigest() for n in ['stroke_rehab/riemann_stream.py','stroke_rehab/features.py','stroke_rehab/data.py','stroke_rehab/spatial.py','stroke_rehab/streaming.py','analysis/reproduce_stream.py']},runtime=new['runtime'],stream_checks=checks)
receipt['output_sha256_lf']={name:hashlib.sha256((reference/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest() for name in ['forward_calibration_predictions.csv','forward_calibration_summary.csv']}
(reference/'stream_parity.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf8')
print('SAVED stream_parity.json',flush=True)

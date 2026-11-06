"""Read small status/log files only; safe for the frontend, no NumPy/JAX."""
import datetime
import json
from pathlib import Path
import re

root=Path(__file__).resolve().parent
plan=json.loads((root/'plan.json').read_text())
rows=[]
for case in plan['cases']:
    folder=root/case['name']
    row={'name':case['name'],'target_steps':case['steps'],'status':'not_started'}
    if (folder/'status.json').exists(): row.update(json.loads((folder/'status.json').read_text()))
    log=folder/'optimizer.log'
    if log.exists():
        with log.open('rb') as f:
            f.seek(max(0,log.stat().st_size-12000));tail=f.read().decode(errors='replace')
        matches=re.findall(r'step (\d+)/(\d+): fidelity=([^,]+), loss=([^,]+), TP=([^,]+), optimizer=([^s]+)s',tail)
        if matches:
            s,total,fid,loss,tp,secs=matches[-1]
            row['latest_checkpoint']=dict(step=int(s),fidelity=float(fid),measurement_loss=float(loss),tp_violation=float(tp),optimizer_seconds=float(secs))
    rows.append(row)
print(json.dumps({'checked_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'runs':rows},indent=2))

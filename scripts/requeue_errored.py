"""Requeue errored (status != ok) predictions and null verdicts for driver resume.
Run ONLY while the driver is stopped, then restart the driver."""
import json
from pathlib import Path

RUNS = Path(__file__).resolve().parent.parent / 'runs/full'
ARMS = ['none', 'raw_rag', 'structured', 'claude_mem', 'mem0', 'graphiti', 'graphify']

dropped = []
for arm in ARMS:
    p = RUNS / arm / 'predictions.jsonl'
    if not p.exists():
        continue
    keep = []
    for l in p.read_text().splitlines():
        if not l.strip():
            continue
        d = json.loads(l)
        if d.get('status') != 'ok':
            dropped.append((arm, d['instance_id']))
        else:
            keep.append(l)
    p.write_text('\n'.join(keep) + ('\n' if keep else ''))

v = RUNS / 'verdicts.jsonl'
if v.exists():
    keep = [l for l in v.read_text().splitlines()
            if l.strip() and json.loads(l)['resolved'] is not None
            and (json.loads(l)['instance_id'], json.loads(l)['arm']) not in dropped]
    v.write_text('\n'.join(keep) + ('\n' if keep else ''))

print('requeued:', dropped)

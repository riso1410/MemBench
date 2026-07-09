"""Adjusted resolve rates from runs/full/*/docker_eval reports.

Strict resolved requires every PASS_TO_PASS test to pass, but some P2P tests
fail identically across arms regardless of patch (broken eval env). Adjusted:
resolved = all FAIL_TO_PASS pass AND no P2P failure outside the per-instance
env-broken set (P2P tests failing in >= 2 arms). Driver verdicts stay strict;
run this anytime for the real numbers.
"""
import json
from collections import defaultdict
from pathlib import Path

RUNS = Path(__file__).resolve().parent.parent / 'runs/full'

reports = {}
p2p_fail_counts = defaultdict(lambda: defaultdict(int))
for rp in RUNS.glob('*/docker_eval/*/*/report.json'):
    arm = rp.relative_to(RUNS).parts[0]
    r = json.loads(rp.read_text())
    iid = r['instance_id']
    reports[(arm, iid)] = r
    for t in r['PASS_TO_PASS']['failure']:
        p2p_fail_counts[iid][t] += 1

env_broken = {iid: {t for t, c in tc.items() if c >= 2}
              for iid, tc in p2p_fail_counts.items()}

strict = defaultdict(int)
adjusted = defaultdict(int)
n = defaultdict(int)
flips = []
for (arm, iid), r in sorted(reports.items()):
    n[arm] += 1
    f2p = r['FAIL_TO_PASS']
    f2p_ok = f2p['success'] and not f2p['failure']
    real_p2p = [t for t in r['PASS_TO_PASS']['failure']
                if t not in env_broken.get(iid, set())]
    adj = bool(f2p_ok and not real_p2p)
    strict[arm] += r['resolved']
    adjusted[arm] += adj
    if adj != r['resolved']:
        flips.append((arm, iid, 'strict' if r['resolved'] else 'adjusted'))

print(f'{"arm":<12} {"strict":>10} {"adjusted":>10}')
for arm in sorted(n):
    print(f'{arm:<12} {strict[arm]:>7}/{n[arm]:<3} {adjusted[arm]:>7}/{n[arm]:<3}')
by_iid = defaultdict(list)
for (a, iid), r in reports.items():
    by_iid[iid].append(bool(r['FAIL_TO_PASS']['success'] or r['FAIL_TO_PASS']['failure']))
zero_f2p = sorted(iid for iid, has in by_iid.items() if not any(has))
if zero_f2p:
    print('WARN zero-F2P in every arm (can never resolve):', zero_f2p)
for arm, iid, direction in flips:
    print(f'flip -> {direction}: {arm}/{iid}')

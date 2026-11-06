import collections
import datetime
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parent
STAGE = Path('/home/tony/qpt_quiroga_validation_20261005_2136')
REPO = Path('/home/tony/frankwolfe')
sys.path.insert(0, str(REPO))
from paper.experiments.qpt_quiroga_direct_comparison import aggregate_sidecars, collect_source_provenance
spec = importlib.util.spec_from_file_location('launcher', STAGE / 'run_validation.py')
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)
plan = json.loads((STAGE / 'validation_plan.json').read_text())
expected = {r['trial_id']: r for r in launcher.make_inventory(plan, STAGE / 'selection_manifest.json')}
hashes = [
    'c270c3457d5f479eb4a9a6952c2a54fd0168dc3453126fc4032a63e1500a440c',
    '5f28f9066a3dde395c81ad0c2ea6b384916defa1e3b4a6b770b5f7b7f15c8558',
    '1ae474842e63b9f75380a83db0b24c54d536273b7809c760c2892b8c5632b6c5',
    '0c0b9b73238434f26003fc3e6a2888f97bfda10ddb4122a98714a059d4e6efa5',
    'a4919a2e8523a10e5499b3ac1217c7a64e91c2c2e0c08779b77771c2fcbe950e',
    '2f6144512385cb83ff7fb2d9c260a8edab996db3b54d68bdcc189ab6324698bd',
    '9271e3b85c81388d95d5979472662505670cf79ae5f3383638ace65100a184dd',
    '822a3d2537d98683b3fee912f00ea05074bed871e9966adea8346cbee30ac7d0']
provenance = collect_source_provenance(REPO)
assert provenance['git']['commit'] == launcher.COMMIT and not provenance['git']['worktree_dirty']
records, outcomes, channels = [], {}, []
pair_counts = collections.Counter()
seen = set()
for channel in range(100, 108):
    folder = ROOT / 'validation-2014377' / f'channel-{channel}'
    assert launcher.file_hash(folder / 'inventory.json') == hashes[channel - 100]
    inv = json.loads((folder / 'inventory.json').read_text())
    assert inv['status'] in ('all_trials_archived', 'all_trials_archived_with_method_failures')
    assert len(inv['trials']) == 3
    assert inv['plan_sha256'] == launcher.file_hash(STAGE / 'validation_plan.json')
    assert inv['selection_manifest_sha256'] == launcher.file_hash(STAGE / 'selection_manifest.json')
    assert inv['wrapper_sha256'] == launcher.file_hash(STAGE / 'run_validation.py')
    paired_hashes, data_hashes = set(), set()
    for trial in inv['trials']:
        tid = trial['trial_id']
        assert tid not in seen
        seen.add(tid)
        for key, value in expected[tid].items():
            assert trial[key] == value, (tid, key)
        out = folder / tid
        assert launcher.file_hash(out / 'execution_status.json') == trial['status_record_sha256']
        status = json.loads((out / 'execution_status.json').read_text())
        assert status['status'] == trial['execution_status']
        expected_command = launcher.command_for(plan, expected[tid], Path(trial['output_directory']))
        assert trial['command'][1:] == status['command'][1:] == expected_command[1:]
        for name, artifact in status['artifacts'].items():
            assert launcher.file_hash(out / name) == artifact['sha256'], (tid, name)
            assert (out / name).stat().st_size == artifact['bytes']
        launcher.validate_archive(plan, expected[tid], out, status['runner_exit_code'])
        record = json.loads((out / 'pair.json').read_text())
        assert record['provenance']['source_sha256'] == provenance['source_sha256']
        records.append(record)
        comp = record['comparison']
        paired_hashes.add(comp['paired_instance_sha256'])
        data_hashes.add(comp['data_instance_sha256'])
        pair_counts[status['status']] += 1
        for method in launcher.METHODS:
            key = (method, channel, trial['sampling_index'] if method == 'frames' else 0)
            endpoint = record['methods'][method]
            outcome = {'channel': channel, 'sampling_index': key[2], 'status': endpoint['status'],
                       'gate': comp['method_gate'][method], 'completed_steps': endpoint['completed_steps'],
                       'failure_reason': endpoint['failure_reason'],
                       'factor_hash': comp['endpoint_diagnostics'][method]['final_factor_sha256']}
            assert key not in outcomes or outcomes[key] == outcome, key
            outcomes[key] = outcome
    assert len(paired_hashes) == len(data_hashes) == 1
    channels.append({'channel': channel, 'inventory_sha256': hashes[channel-100],
                     'status': inv['status'], 'finished_utc': inv['finished_utc']})
assert seen == set(expected)
aggregate_sidecars(records)
metrics = ('process_fidelity', 'rmse', 'normalized_tp_violation')
methods = {}
for method in launcher.METHODS:
    rows = [value for key, value in outcomes.items() if key[0] == method]
    assert len(rows) == (8 if method == 'printed-adafgd' else 24)
    per_channel = []
    for channel in range(100, 108):
        subset = [r for r in rows if r['channel'] == channel]
        means = {}
        for metric in metrics:
            values = [r['gate']['metric_values'][metric] for r in subset]
            worst = 0.0 if metric == 'process_fidelity' else math.inf
            means[metric] = statistics.mean(v if v is not None else worst for v in values)
        per_channel.append({'channel': channel, 'metric_means': means,
                            'full_gate_rate': statistics.mean(int(r['gate']['passed']) for r in subset)})
    complete = [r for r in rows if r['status'] == 'complete']
    methods[method] = {
        'unique_outcomes': len(rows), 'complete': len(complete), 'failed': len(rows) - len(complete),
        'failed_channels': [r['channel'] for r in rows if r['status'] != 'complete'],
        'gate_passes': sum(r['gate']['passed'] for r in rows),
        'nested_gate_rate': statistics.mean(r['full_gate_rate'] for r in per_channel),
        'component_pass_counts': {m: sum(r['gate']['component_passes'][m] for r in rows) for m in metrics},
        'failure_inclusive_channel_medians': {m: statistics.median(r['metric_means'][m] for r in per_channel) for m in metrics},
        'complete_only_metric_ranges': {m: [min(r['gate']['metric_values'][m] for r in complete),
                                              max(r['gate']['metric_values'][m] for r in complete)] for m in metrics},
        'complete_only_metric_medians': {m: statistics.median(r['gate']['metric_values'][m] for r in complete) for m in metrics},
        'channel_summary': per_channel, 'outcomes': rows}
def json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return '+infinity' if value > 0 else '-infinity'
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    return value
summary = json_safe({'checked_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'status': 'all_24_archives_verified_against_remote_inventory_hashes',
    'source_commit': launcher.COMMIT, 'selection_manifest_sha256': launcher.file_hash(STAGE / 'selection_manifest.json'),
    'pair_counts': dict(pair_counts), 'channels': channels, 'methods': methods,
    'aggregation': 'Eight equally weighted channels; average three FRAMES streams within channel. Quiroga repeats deduplicated. Failed outcomes remain in gate denominators; missing fidelity maps to 0 and missing RMSE/TP maps to +infinity for failure-inclusive channel medians. Complete-only summaries are diagnostic, not success estimands.',
    'no_validation_reranking': True})
with (ROOT / 'validation_audit.json').open('x') as stream:
    json.dump(summary, stream, indent=2, allow_nan=False)
print(json.dumps({**{k: v for k, v in summary.items() if k != 'methods'},
    'methods': {k: {a: b for a, b in v.items() if a not in ('outcomes', 'channel_summary')} for k, v in methods.items()}}, indent=2))

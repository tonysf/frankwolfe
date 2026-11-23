"""Read-only reproducibility checks between fresh starts and archived prefixes."""
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
OLD = Path('/home/tony/qpt_frames_local_followup_20261012')
FIELDS = ('measurement_loss_mean', 'measurement_loss_sum', 'rmse', 'normalized_tp_violation',
          'tp_violation', 'process_fidelity')


def compare(short, long):
    with np.load(short, allow_pickle=False) as a, np.load(long, allow_pickle=False) as b:
        for key in ('initial_factor', 'truth_factor', 'active_rows', 'active_observations'):
            assert np.array_equal(a[key], b[key]), key
        steps = a['frames_checkpoint_steps']; reference_steps = b['frames_checkpoint_steps']
        indices = np.searchsorted(reference_steps, steps)
        assert np.array_equal(reference_steps[indices], steps)
        differences = {}
        for name in FIELDS:
            x, y = a['frames_'+name], b['frames_'+name][indices]
            assert np.allclose(x, y, rtol=1e-12, atol=1e-15), name
            differences[name] = float(np.max(np.abs(x-y)))
        for name in ('step_size', 'momentum_weight', 'smoothing_parameter'):
            key = 'frames_diagnostic_'+name
            assert np.array_equal(a[key], b[key][:len(a[key])]), key
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    return {'short': str(short), 'long': str(long), 'short_sha256': digest(short),
            'long_sha256': digest(long), 'shared_checkpoint_count': len(steps),
            'physical_metric_max_abs_differences': differences, 'schedule_prefix_identical': True,
            'paired_data_and_initial_factor_identical': True}


def main():
    output = ROOT/'prefix_checks.json'
    assert not output.exists()
    checks = [compare(ROOT/'screen'/f'incumbent-c{ch}-i0-s0'/'result.npz',
                      OLD/'transfer'/f'n4-c{ch}-i0-s0'/'result.npz') for ch in (0, 1)]
    plan = json.loads((ROOT/'plan_long.json').read_text())
    for trial in plan['trials']:
        checks.append(compare(ROOT/'screen'/trial['id']/'result.npz', ROOT/'long'/trial['id']/'result.npz'))
    value = {'all_prefixes_match': True, 'checks': checks,
             'qualification': 'Fresh-start physical checkpoint and schedule agreement. Intermediate factors and realized native batches are not archived. Penalized objective omitted because final versus interior checkpoints use different beta indices.'}
    with output.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
    print(json.dumps({'all_prefixes_match': True, 'comparisons': len(checks)}))


if __name__ == '__main__': main()

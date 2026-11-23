"""Check frozen experiment boundary guards without launching optimization."""
from copy import deepcopy
import json
from pathlib import Path
import pytest
from run_followup import validate_plan

ROOT = Path(__file__).resolve().parent


@pytest.mark.parametrize('phase', ['pilot', 'robustness', 'tail', 'transfer'])
def test_frozen_plan_is_valid(phase):
    validate_plan(json.loads((ROOT / f'plan_{phase}.json').read_text()))


@pytest.mark.parametrize('key,value', [
    ('channel', 1000), ('n_qubits', 5), ('alpha0', 1e-4), ('rho_exponent', .6),
    ('initialization_seed', 17), ('sampling_seed', 12), ('tau', 4.),
    ('steps', 800000), ('sampling_replicate', 0),
])
def test_modified_robustness_trial_rejected(key, value):
    plan = json.loads((ROOT / 'plan_robustness.json').read_text())
    plan['trials'][0][key] = value
    with pytest.raises(AssertionError):
        validate_plan(plan)


def test_missing_harness_pins_rejected():
    plan = json.loads((ROOT / 'plan_robustness.json').read_text())
    plan['harness_sha256'] = {}
    with pytest.raises(AssertionError):
        validate_plan(plan)


@pytest.mark.parametrize('phase', ['pilot', 'robustness', 'tail', 'transfer'])
def test_partial_cohort_rejected(phase):
    plan = json.loads((ROOT / f'plan_{phase}.json').read_text())
    plan['trials'].pop()
    with pytest.raises(AssertionError):
        validate_plan(plan)

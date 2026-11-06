"""Time the original SCOT/POT solve on the already verified SCOT1047 arrays.

Set BLAS/OpenMP thread environment variables before starting Python. This
script neither changes solver defaults nor reconstructs the reference graph.
"""
from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import importlib.metadata
import importlib.util
import inspect
import io
import json
import math
import os
from pathlib import Path
import platform
import resource
import sys
import time
import traceback
import warnings


ROOT = Path(__file__).resolve().parent
THREAD_VARIABLES = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                    'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS',
                    'OMP_DYNAMIC', 'MKL_DYNAMIC')
OVERRIDES = dict(epsilon=.001, loss_fun='square_loss', log=True, verbose=False)


def load_reference_module():
    path = ROOT.parent / 'scot_reproduction_20261014/reference.py'
    spec = importlib.util.spec_from_file_location('_scot_cpu_frozen_reference', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def json_value(value):
    """Preserve POT log values, including nonfinite failure evidence, as JSON."""
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    if hasattr(value, 'tolist'):
        return json_value(value.tolist())
    if isinstance(value, float) and not math.isfinite(value):
        return repr(value)
    return value


def run(args):
    # Imports happen after the caller has chosen process-wide thread settings.
    import numpy as np
    import ot

    reference = load_reference_module()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    reference_path = Path(args.reference).resolve()
    if reference_path.is_dir():
        reference_path /= 'reference.json'
    start = time.perf_counter()
    result = dict(schema_version=1, status='started', experiment='scot1047_local_cpu_reference_replay',
                  generated_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                  reference_file=str(reference_path), timing={}, warnings=[],
                  environment=dict(python=platform.python_version(), python_executable=sys.executable,
                                   platform=platform.platform(), hostname=platform.node(),
                                   cpu_count=os.cpu_count(),
                                   cpu_affinity=sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None,
                                   thread_environment={name: os.environ.get(name) for name in THREAD_VARIABLES},
                                   thread_control='Environment configured by caller before Python starts; no threadpoolctl override.',
                                   effective_library_thread_counts='Not introspected; configured environment is recorded.'),
                  solver=dict(versions={name: importlib.metadata.version(name) for name in ('numpy', 'scipy', 'POT')},
                              entropic_gw_signature=str(inspect.signature(ot.gromov.entropic_gromov_wasserstein)),
                              sinkhorn_signature=str(inspect.signature(ot.bregman.sinkhorn)),
                              effective_overrides=OVERRIDES,
                              backend='NumPy CPU, float64 input arrays',
                              omitted_parameters='All omitted solver arguments retain installed POT defaults, as in original SCOT.',
                              initialization='POT default G0=None; product coupling a outer b.'),
                  interpretation='CPU timing replay on verified frozen SCOT1047 geometry; no preprocessing, tuning, tightened tolerance or stochastic comparison is included in solver time.')
    reference.write_json(out / 'replay.json', result)
    captured = []
    try:
        verified = reference.verify_reference(reference_path)
        result['reference_sha256'] = reference.sha256(reference_path)
        result['reference_status'] = verified['status']
        result['reference_solver'] = verified['solver']
        prepared = verified['prepared_instance']
        directory = reference_path.parent / prepared['relative_directory']
        result['prepared_instance'] = dict(directory=str(directory), manifest_sha256=prepared['manifest_sha256'],
                                          array_files_sha256=prepared['array_files_sha256'])
        result['source_file_sha256'] = reference.archive_sources(out, ot)
        archive = out / 'provenance/replay_reference.py'
        archive.write_bytes(Path(__file__).read_bytes())
        result['source_file_sha256']['provenance/replay_reference.py'] = reference.sha256(archive)
        configuration = io.StringIO()
        with contextlib.redirect_stdout(configuration):
            np.show_config()
        (out / 'numpy_configuration.txt').write_text(configuration.getvalue())
        result['numpy_configuration_sha256'] = reference.sha256(out / 'numpy_configuration.txt')
        before = time.perf_counter()
        A, B, a, b = (np.load(directory / name, allow_pickle=False) for name in reference.ARRAY_NAMES)
        if (A.shape, B.shape, a.shape, b.shape) != ((1047, 1047), (1047, 1047), (1047,), (1047,)):
            raise ValueError('Unexpected frozen reference array shapes.')
        if any(array.dtype != np.float64 or not np.isfinite(array).all() for array in (A, B, a, b)):
            raise ValueError('Replay requires finite original float64 arrays.')
        result['timing']['array_loading_seconds'] = time.perf_counter() - before
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter('always')
            before = time.perf_counter()
            result['initial'] = reference.evaluate(A, B, np.outer(a, b), a, b, ot)
            result['timing']['initial_diagnostics_seconds'] = time.perf_counter() - before
            with (out / 'solver_stdout.txt').open('w') as stream, contextlib.redirect_stdout(stream):
                before_cpu = time.process_time()
                before = time.perf_counter()
                P, log = ot.gromov.entropic_gromov_wasserstein(A, B, a, b, **OVERRIDES)
                result['timing']['solver_seconds'] = time.perf_counter() - before
                result['timing']['solver_process_cpu_seconds'] = time.process_time() - before_cpu
            P = np.asarray(P, dtype=np.float64)
            reference.write_json(out / 'pot_log.json', json_value(log))
            result['pot_log_sha256'] = reference.sha256(out / 'pot_log.json')
            np.save(out / 'P_raw.npy', P, allow_pickle=False)
            result['raw_plan_sha256'] = reference.sha256(out / 'P_raw.npy')
            before = time.perf_counter()
            result['raw'] = reference.evaluate(A, B, P, a, b, ot)
            logged = float(log['gw_dist'])
            fixed = reference.fixed_marginal_loss(A, B, P, a, b)
            if not math.isfinite(logged):
                raise ValueError('POT logged a nonfinite objective.')
            result['solver_output'] = dict(gw_dist=logged, independent_fixed_marginal_expression=fixed,
                                           fixed_expression_absolute_difference=abs(logged - fixed),
                                           actual_full_loss_minus_log=result['raw']['loss'] - logged)
            result['timing']['raw_diagnostics_seconds'] = time.perf_counter() - before
            before = time.perf_counter()
            Q = reference.round_transport(P, a, b)
            result['timing']['repair_seconds'] = time.perf_counter() - before
            before = time.perf_counter()
            result['rounded'] = reference.evaluate(A, B, Q, a, b, ot)
            result['repair_entrywise_l1'] = float(np.abs(Q - P).sum())
            result['timing']['rounded_diagnostics_seconds'] = time.perf_counter() - before
            np.save(out / 'P_rounded.npy', Q, allow_pickle=False)
            result['rounded_plan_sha256'] = reference.sha256(out / 'P_rounded.npy')
            # The released SCOT flag, without importing scikit-learn or
            # re-executing graph construction. evaluate() separately rejects
            # nonfinite/negative entries and audits actual marginal errors.
            basic_flag = bool(not (np.isnan(P).any() or np.any(~P.any(axis=1))
                                   or np.any(~P.any(axis=0)) or sum(sum(P)) < .95))
            result['gate'] = reference.gate(result['initial'], result['raw'], result['rounded'],
                                           author_flag=basic_flag, author_loss_agrees=abs(logged - fixed) <= 1e-11,
                                           author_loss=logged)
            result['versus_downloaded_reference'] = dict(
                raw_full_loss_difference=result['raw']['loss'] - verified['raw']['loss'],
                rounded_full_loss_difference=result['rounded']['loss'] - verified['rounded']['loss'],
                logged_loss_difference=logged - verified['author_output']['gw_dist'],
                raw_plan_sha256_matches=result['raw_plan_sha256'] == verified['raw_plan_sha256'],
                rounded_plan_sha256_matches=result['rounded_plan_sha256'] == verified['rounded_plan_sha256'])
            result['status'] = 'passed' if result['gate']['passed'] else 'failed_reproduction_gate'
    except Exception as exc:
        result.update(status='failed', error=repr(exc), traceback=traceback.format_exc())
    finally:
        result['warnings'] = [dict(category=item.category.__name__, message=str(item.message),
                                   filename=item.filename, lineno=item.lineno) for item in captured]
        result['timing']['total_elapsed_seconds'] = time.perf_counter() - start
        result['cpu_process_peak_rss_kib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        reference.write_json(out / 'replay.json', json_value(result))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    result = run(args)
    print(json.dumps(dict(status=result['status'], out=str(Path(args.out).resolve()),
                          solver_seconds=result['timing'].get('solver_seconds'),
                          rounded_loss=result.get('rounded', {}).get('loss'))), flush=True)
    return 0 if result['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())

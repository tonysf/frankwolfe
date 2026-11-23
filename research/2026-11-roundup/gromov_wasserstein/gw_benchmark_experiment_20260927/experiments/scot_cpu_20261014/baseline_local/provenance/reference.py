"""Reproduce the released SCOT 1047-cell example without solver overrides.

Run only on an allocated CPU node for the full data.  The original SCOT source
and features must match the fixed hashes below.  POT/library versions are
recorded, so this is a source/settings reproduction, not a bitwise historical
environment claim.  No self tuning, alternate epsilon, or tightened solve runs.
"""
from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import hashlib
import importlib.metadata
import inspect
import json
import math
import os
from pathlib import Path
import platform
import resource
import sys
import time
import traceback
import types
import warnings

import numpy as np

INPUT_HASHES = {
    'SNAREseq_atac_feat.npy': '80d685b6517c100a7fd612846ad38209b64ed9caf095d1630af5a1d7797bd79c',
    'SNAREseq_rna_feat.npy': 'a8a2c2bbfc036dea5b350d01dd2180b93bb3b1c0008bed49493080348da96a9b',
    'scotv1.py': 'a492f39a67ffa62d8d7cc4b5048871d6cf228bf98b53a31d0e3049ab24db8008',
}
EXPECTED_START = 0.07045975745628463
PUBLISHED_POT = 0.016519975128479896
ROUNDED_THRESHOLD = 0.025
PUBLISHED_TOLERANCE = 0.005
FEASIBILITY_TOLERANCE = 1e-12
OBJECTIVE = 'full_squared_discrepancy_no_half_no_entropy'
ARRAY_NAMES = ('A.npy', 'B.npy', 'a.npy', 'b.npy')


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def checked_source(path):
    """Compile the verified source bytes directly; never trust/write a pyc."""
    path = Path(path)
    source = path.read_bytes()
    if hashlib.sha256(source).hexdigest() != INPUT_HASHES['scotv1.py']:
        raise ValueError('The released SCOT source hash differs.')
    module = types.ModuleType('scot_reproduction_frozen_author_source')
    module.__file__ = str(path)
    exec(compile(source, str(path), 'exec'), module.__dict__)
    return module


def prepare(scot_class, X, Y, *, k=40):
    """Call the released implementation; k is variable only for tiny unit tests."""
    model = scot_class(X.copy(), Y.copy())
    model.normalize(norm='l2')
    model.init_marginals()
    model.construct_graph(k=k, mode='connectivity', metric='correlation')
    model.init_distances()
    arrays = tuple(np.asarray(x, dtype=np.float64).copy()
                   for x in (model.Cx, model.Cy, model.p, model.q))
    A, B, a, b = arrays
    for M in (A, B):
        if not np.isfinite(M).all() or (M < 0).any() or not np.array_equal(M, M.T):
            raise ValueError('SCOT produced invalid or asymmetric distances.')
    if not np.allclose(a, 1/len(a), rtol=0, atol=0) or not np.allclose(b, 1/len(b), rtol=0, atol=0):
        raise ValueError('SCOT did not produce uniform marginals.')
    return model, arrays


def full_loss(A, B, P):
    """Full four-index squared discrepancy, factored using ACTUAL marginals."""
    r, c = P.sum(axis=1), P.sum(axis=0)
    return float(r @ ((A*A) @ r) + c @ ((B*B) @ c)
                 - 2 * np.sum((A @ P @ B.T) * P))


def fixed_marginal_loss(A, B, P, a, b):
    """POT's logged expression: distinct from full_loss when P is infeasible."""
    r, c = P.sum(axis=1), P.sum(axis=0)
    return float(r @ ((A*A) @ a) + c @ ((B*B) @ b)
                 - 2 * np.sum((A @ P @ B.T) * P))


def residuals(P, a, b):
    er, ec = P.sum(axis=1)-a, P.sum(axis=0)-b
    return dict(row_l1=float(np.abs(er).sum()), col_l1=float(np.abs(ec).sum()),
                row_l2=float(np.linalg.norm(er)), col_l2=float(np.linalg.norm(ec)),
                both_l1=float(np.abs(er).sum()+np.abs(ec).sum()),
                mass=float(P.sum()), mass_error=float(abs(P.sum()-a.sum())),
                min_entry=float(P.min()), max_entry=float(P.max()))


def round_transport(P, a, b):
    """Output-only row/column downscaling followed by rank-one deficit fill."""
    if not np.isfinite(P).all() or (P < 0).any():
        raise ValueError('Cannot repair a nonfinite or negative coupling.')
    tiny = np.finfo(np.float64).tiny
    Q = P * np.minimum(1., a / np.maximum(P.sum(axis=1), tiny))[:, None]
    Q *= np.minimum(1., b / np.maximum(Q.sum(axis=0), tiny))[None, :]
    r, c = np.maximum(a-Q.sum(axis=1), 0), np.maximum(b-Q.sum(axis=0), 0)
    if r.sum() > 0:
        Q += np.outer(r, c) / r.sum()
    return Q


def evaluate(A, B, P, a, b, ot):
    if P.shape != (len(a), len(b)) or not np.isfinite(P).all() or (P < 0).any():
        raise ValueError('Invalid returned coupling; no replacement is allowed.')
    value = full_loss(A, B, P)
    # Independent library contraction receives the ACTUAL marginals explicitly.
    const, hA, hB = ot.gromov.init_matrix(A, B, P.sum(1), P.sum(0), 'square_loss')
    library_value = float(ot.gromov.gwloss(const, hA, hB, P))
    if not math.isfinite(value) or not math.isfinite(library_value):
        raise ValueError('Full objective evaluation is nonfinite.')
    difference = abs(value-library_value)
    return dict(loss=value, pot_actual_marginals_loss=library_value,
                independent_loss_absolute_difference=difference,
                independent_loss_agrees=bool(math.isfinite(value) and math.isfinite(library_value)
                                             and difference <= 1e-11),
                residuals=residuals(P, a, b))


def gate(initial, raw, rounded, *, author_flag, author_loss_agrees, author_loss):
    """Prospective reproduction gate only; never a stochastic trial loss gate."""
    checks = dict(
        losses_finite=all(math.isfinite(x['loss']) for x in (initial, raw, rounded)) and math.isfinite(author_loss),
        initial_product_matches_frozen_preprocessing=abs(initial['loss']-EXPECTED_START) <= 1e-12,
        independent_full_objectives_agree=all(x['independent_loss_agrees'] for x in (initial, raw, rounded)),
        author_log_matches_its_fixed_marginal_expression=bool(author_loss_agrees),
        author_basic_numerical_flag=bool(author_flag),
        rounded_feasible=rounded['residuals']['both_l1'] <= FEASIBILITY_TOLERANCE,
        rounded_nonnegative=rounded['residuals']['min_entry'] >= 0,
        rounded_loss_below_reproduction_threshold=rounded['loss'] <= ROUNDED_THRESHOLD,
        author_logged_loss_within_published_reference_tolerance=abs(author_loss-PUBLISHED_POT) <= PUBLISHED_TOLERANCE,
    )
    return dict(passed=all(checks.values()), checks=checks,
                rounded_loss_threshold=ROUNDED_THRESHOLD,
                published_author_logged_loss_reference=PUBLISHED_POT, published_absolute_tolerance=PUBLISHED_TOLERANCE,
                feasibility_both_l1_tolerance=FEASIBILITY_TOLERANCE,
                interpretation='Prospective reference reproduction check, not a certified optimum or a stochastic production loss gate. The published POT loss and reproduced author log both use the prescribed-marginal expression; they are compared with an explicit 0.005 tolerance. Actual-marginal raw and repaired full losses are reported separately; the repaired loss must independently be at most 0.025.')


def archive_sources(out, ot):
    directory = out/'provenance'
    directory.mkdir()
    paths = {'reference.py': Path(__file__),
             'pot_entropic_module.py': Path(inspect.getfile(ot.gromov.entropic_gromov_wasserstein)),
             'pot_sinkhorn_module.py': Path(inspect.getfile(ot.bregman.sinkhorn)),
             'pot_gw_utils_module.py': Path(inspect.getfile(ot.gromov.init_matrix))}
    hashes = {}
    for name, source in paths.items():
        dest = directory/name
        dest.write_bytes(source.read_bytes())
        hashes[str(dest.relative_to(out))] = sha256(dest)
    return hashes


def package_details(ot):
    return dict(versions={name: importlib.metadata.version(name) for name in
                          ('numpy', 'scipy', 'scikit-learn', 'POT', 'threadpoolctl')},
                entropic_gw_signature=str(inspect.signature(ot.gromov.entropic_gromov_wasserstein)),
                sinkhorn_signature=str(inspect.signature(ot.bregman.sinkhorn)),
                effective_overrides=dict(loss_fun='square_loss', epsilon=.001, log=True, verbose=False),
                other_solver_parameters='All omitted exactly as in released SCOT.find_correspondences; installed POT defaults apply.',
                initialization='Uniform product a outer b, POT default G0=None; no randomization or self-tuning.')


def run(args):
    from threadpoolctl import threadpool_info, threadpool_limits
    import ot

    out, input_dir = Path(args.out).resolve(), Path(args.input_dir).resolve()
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    result = dict(schema_version=1, status='started', dataset='SCOT_SNARE1047_human_cell_mixture',
                  config=dict(k=40, epsilon=.001, normalization='row_l2', graph_mode='connectivity',
                              graph_metric='correlation', include_self=True, shortest_path_directed=False,
                              disconnected_policy='replace infinities by finite graph diameter',
                              distance_normalization='each graph divided by its own finite diameter',
                              threads=args.threads, dtype='float64', objective=OBJECTIVE),
                  input_dir=str(input_dir), input_file_sha256={},
                  source_urls=dict(scot='https://github.com/rsinghlab/SCOT/blob/master/src/scotv1.py',
                                   tutorial='https://ott-jax.readthedocs.io/tutorials/quadratic/200_gromov_wasserstein_multiomics.html'),
                  published_tutorial=dict(POT_loss=PUBLISHED_POT, OTT_loss=.01785334522049412,
                                          POT_marginal_l1=.0013310870921931293,
                                          warning='POT Sinkhorn did not converge; values are approximate outputs, not certified optima.'),
                  environment=dict(python=platform.python_version(), platform=platform.platform(),
                                   hostname=platform.node(), slurm_job_id=os.environ.get('SLURM_JOB_ID')),
                  generated_utc=dt.datetime.now(dt.timezone.utc).isoformat(), timing={}, warnings=[])
    write_json(out/'reference.json', result)
    caught = []
    try:
        for name, expected in INPUT_HASHES.items():
            actual = sha256(input_dir/name)
            result['input_file_sha256'][name] = actual
            if actual != expected:
                raise ValueError(f'Frozen input hash mismatch: {name}')
        result['source_file_sha256'] = archive_sources(out, ot)
        author_archive = out/'provenance/scotv1.py'
        author_archive.write_bytes((input_dir/'scotv1.py').read_bytes())
        result['source_file_sha256']['provenance/scotv1.py'] = sha256(author_archive)
        result['solver'] = package_details(ot)
        with threadpool_limits(limits=args.threads), warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            result['environment']['threadpools'] = threadpool_info()
            model_source = checked_source(author_archive)
            X = np.load(input_dir/'SNAREseq_atac_feat.npy', allow_pickle=False)
            Y = np.load(input_dir/'SNAREseq_rna_feat.npy', allow_pickle=False)
            if X.shape != (1047, 19) or Y.shape != (1047, 10) or not np.isfinite(X).all() or not np.isfinite(Y).all():
                raise ValueError('Unexpected released feature shapes or nonfinite values.')
            before = time.perf_counter()
            model, (A, B, a, b) = prepare(model_source.SCOT, X, Y)
            result['timing']['preparation_seconds'] = time.perf_counter()-before
            directory = out/'prepared/scot1047'
            directory.mkdir(parents=True)
            hashes = {}
            for name, values in zip(ARRAY_NAMES, (A, B, a, b)):
                np.save(directory/name, values, allow_pickle=False)
                hashes[name] = sha256(directory/name)
            manifest = dict(schema_version=1, n=len(a), m=len(b), dataset=result['dataset'],
                            objective=OBJECTIVE, cost_storage_dtype='float64', marginal_storage_dtype='float64',
                            array_files_sha256=hashes, input_file_sha256=INPUT_HASHES,
                            preprocessing=result['config'], reference_status='pending_solver_and_validation')
            write_json(directory/'manifest.json', manifest)
            result['prepared_instance'] = dict(directory=str(directory), relative_directory='prepared/scot1047',
                                              manifest_sha256=sha256(directory/'manifest.json'), array_files_sha256=hashes)
            before = time.perf_counter()
            result['initial'] = evaluate(A, B, np.outer(a, b), a, b, ot)
            result['timing']['initial_diagnostics_seconds'] = time.perf_counter()-before
            # This is the authors' unchanged method; it supplies no inner/outer
            # iteration/tolerance overrides. Barycentric projection is unnecessary
            # for the GW objective and is not included in solver timing.
            with (out/'solver_stdout.txt').open('w') as stream, contextlib.redirect_stdout(stream):
                before = time.perf_counter()
                model.find_correspondences(e=.001, verbose=False)
                result['timing']['solver_seconds'] = time.perf_counter()-before
            P = np.asarray(model.coupling, dtype=np.float64).copy()
            np.save(out/'P_raw.npy', P, allow_pickle=False)
            result['raw_plan_sha256'] = sha256(out/'P_raw.npy')
            before = time.perf_counter()
            result['raw'] = evaluate(A, B, P, a, b, ot)
            reported = float(model.gwdist)
            fixed = fixed_marginal_loss(A, B, P, a, b)
            if not math.isfinite(reported):
                raise ValueError('Author solver reported a nonfinite loss.')
            result['author_output'] = dict(gw_dist=reported, basic_numerical_flag=bool(model.flag),
                                          independent_fixed_marginal_expression=fixed,
                                          fixed_expression_absolute_difference=abs(reported-fixed),
                                          actual_full_loss_minus_author_log=result['raw']['loss']-reported)
            result['timing']['raw_diagnostics_seconds'] = time.perf_counter()-before
            before = time.perf_counter()
            Q = round_transport(P, a, b)
            result['timing']['repair_seconds'] = time.perf_counter()-before
            before = time.perf_counter()
            result['rounded'] = evaluate(A, B, Q, a, b, ot)
            result['repair_entrywise_l1'] = float(np.abs(Q-P).sum())
            result['timing']['rounded_diagnostics_seconds'] = time.perf_counter()-before
            np.save(out/'P_rounded.npy', Q, allow_pickle=False)
            result['rounded_plan_sha256'] = sha256(out/'P_rounded.npy')
            result['gate'] = gate(result['initial'], result['raw'], result['rounded'],
                                  author_flag=model.flag, author_loss_agrees=abs(reported-fixed) <= 1e-11,
                                  author_loss=reported)
            result['status'] = 'passed' if result['gate']['passed'] else 'failed_reproduction_gate'
            manifest['reference_status'] = result['status']
            write_json(directory/'manifest.json', manifest)
            result['prepared_instance']['manifest_sha256'] = sha256(directory/'manifest.json')
    except Exception as exc:
        result.update(status='failed', error=repr(exc), traceback=traceback.format_exc())
    finally:
        result['warnings'] = [dict(category=w.category.__name__, message=str(w.message),
                                   filename=w.filename, lineno=w.lineno) for w in caught]
        result['timing']['total_elapsed_seconds'] = time.perf_counter()-started
        result['cpu_process_peak_rss_kib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        result['interpretation'] = 'Original SCOT/POT settings, installed library defaults recorded; raw and output-only repaired GW costs are independently evaluated in float64. No tightened alternate solve or optimizer selection.'
        write_json(out/'reference.json', result)
    return result


def verify_reference(path):
    """Read-only portable gate/hash verification, before a comparison is run."""
    path = Path(path)
    root = path if path.is_dir() else path.parent
    record = json.loads((root/'reference.json').read_text())
    if record.get('status') != 'passed' or not record.get('gate', {}).get('passed'):
        raise ValueError('SCOT reference reproduction did not pass.')
    if record.get('input_file_sha256') != INPUT_HASHES:
        raise ValueError('Reference does not bind the released features/source.')
    recalculated_gate = gate(record['initial'], record['raw'], record['rounded'],
                             author_flag=record['author_output']['basic_numerical_flag'],
                             author_loss_agrees=record['author_output']['fixed_expression_absolute_difference'] <= 1e-11,
                             author_loss=record['author_output']['gw_dist'])
    if record['gate'] != recalculated_gate or not recalculated_gate['passed']:
        raise ValueError('Reference gate or threshold changed.')
    prepared = record['prepared_instance']
    if prepared['relative_directory'] != 'prepared/scot1047':
        raise ValueError('Unexpected canonical instance location.')
    directory = root/prepared['relative_directory']
    if sha256(directory/'manifest.json') != prepared['manifest_sha256']:
        raise ValueError('Prepared manifest hash mismatch.')
    manifest = json.loads((directory/'manifest.json').read_text())
    hashes = {name: sha256(directory/name) for name in ARRAY_NAMES}
    if hashes != prepared['array_files_sha256'] or hashes != manifest['array_files_sha256']:
        raise ValueError('Prepared array file hash mismatch.')
    if (manifest['n'], manifest['m']) != (1047, 1047) or manifest['objective'] != OBJECTIVE or manifest.get('reference_status') != 'passed':
        raise ValueError('Prepared instance contract mismatch.')
    for name, key in [('P_raw.npy', 'raw_plan_sha256'), ('P_rounded.npy', 'rounded_plan_sha256')]:
        if sha256(root/name) != record[key]:
            raise ValueError('Saved reference plan hash mismatch.')
    for name, expected in record['source_file_sha256'].items():
        relative = Path(name)
        if relative.is_absolute() or '..' in relative.parts or sha256(root/relative) != expected:
            raise ValueError('Archived source hash mismatch.')
    if record['source_file_sha256'].get('provenance/scotv1.py') != INPUT_HASHES['scotv1.py']:
        raise ValueError('Archived author source mismatch.')
    return record


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-dir', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--threads', type=int, default=1)
    return p


def main():
    args = parser().parse_args()
    if args.threads < 1:
        raise ValueError('threads must be positive.')
    result = run(args)
    print(json.dumps(dict(status=result['status'], out=str(Path(args.out).resolve()), gate=result.get('gate'))))
    return 0 if result['status'] == 'passed' else 1


if __name__ == '__main__':
    sys.exit(main())

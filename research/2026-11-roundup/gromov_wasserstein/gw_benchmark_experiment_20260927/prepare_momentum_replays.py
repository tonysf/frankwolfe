"""Freeze companion-only momentum diagnostic replay plans; never run optimizers.

Tuning and pilot files are bound now from verified local archives. Final trials
are selected from the frozen scientific manifests (not discovered by successes);
actual final artifacts must be terminal and hash-bound before their replay starts.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
from functools import lru_cache
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent
METHODS = {'smoothing-fw', 'smoothing-fw-exact', 'smoothing-row-projection'}
SOURCES = ('run_benchmark.py', 'smoothing_fw.py', 'gw_core.py', 'theory_config.json')
TERMINAL = {'completed', 'completed_with_warnings', 'completed_with_inner_failures', 'failed'}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def canonical_absolute(value):
    path = Path(value)
    if not path.is_absolute() or str(path) != str(value) or '..' in path.parts:
        raise ValueError(f'Noncanonical absolute path: {value!r}')
    return path


def safe_relative(value):
    path = Path(value)
    if path.is_absolute() or str(path) != value or not path.parts or '..' in path.parts:
        raise ValueError(f'Unsafe relative path: {value!r}')
    if path.parts[0].startswith('_'):
        raise ValueError('Reserved control path.')
    return path


def overlaps(a, b):
    a, b = Path(a), Path(b)
    return a == b or a in b.parents or b in a.parents


def artifact_hashes(run_dir):
    run_dir = Path(run_dir)
    failed = read_json(run_dir / 'metadata.json').get('status') == 'failed'
    names = ('metadata.json', 'trace.csv', 'final.npz')
    return {name: (None if failed and name != 'metadata.json' and not (run_dir / name).exists()
                   else digest(run_dir / name)) for name in names}


def snapshot_hashes(run_dir):
    return {name: digest(Path(run_dir) / 'source_snapshot' / name) for name in SOURCES}


@lru_cache(maxsize=None)
def revision_hashes(revision, study_root=ROOT):
    result = {}
    for name in SOURCES:
        tracked = 'configs/theory_config.json' if name == 'theory_config.json' else name
        content = subprocess.check_output(['git', 'show', f'{revision}:{tracked}'], cwd=study_root)
        result[name] = hashlib.sha256(content).hexdigest()
    return result


def bind_archive(run_dir, *, run_spec=None, manifest_sha256=None, study_root=ROOT):
    meta = read_json(Path(run_dir) / 'metadata.json')
    if meta.get('status') not in TERMINAL or meta.get('source_dirty') is not False:
        raise ValueError(f'Original must be terminal with clean archived source: {run_dir}')
    sources = snapshot_hashes(run_dir)
    for name in SOURCES[:3]:
        if meta['source_file_sha256'][name] != sources[name]:
            raise ValueError(f'Original source snapshot hash mismatch: {run_dir}/{name}')
    if sources != revision_hashes(meta['source_commit'], Path(study_root)):
        raise ValueError(f'Original snapshot differs from its recorded revision: {run_dir}')
    if run_spec is not None:
        rec = read_json(Path(run_dir) / 'campaign_receipt.json')
        if (rec.get('run_spec') != run_spec or rec.get('manifest_sha256') != manifest_sha256
                or rec.get('status') != meta['status'] or rec.get('infrastructure_error') is not None):
            raise ValueError(f'Original campaign receipt mismatch: {run_dir}')
    return meta, sources, artifact_hashes(run_dir)


def validate_plan(plan):
    if plan.get('schema_version') != 1 or plan.get('purpose') != 'missing_momentum_diagnostics_only':
        raise ValueError('Unsupported replay manifest.')
    out_root = canonical_absolute(plan['out_root'])
    data_root = canonical_absolute(plan['data_root'])
    original_roots = [canonical_absolute(p) for p in plan['original_roots']]
    if any(overlaps(out_root, p) for p in [data_root, *original_roots]):
        raise ValueError('Companion output must be disjoint from originals and data.')
    ids, originals = set(), set()
    for trial in plan['runs']:
        ident = str(safe_relative(trial['run_id']))
        original = canonical_absolute(trial['original_run_dir'])
        if ident in ids or str(original) in originals:
            raise ValueError('Duplicate replay ID or original trajectory.')
        if any(overlaps(ident, other) for other in ids):
            raise ValueError('Nested replay IDs would overlap companion directories.')
        ids.add(ident); originals.add(str(original))
        if not any(root in original.parents for root in original_roots):
            raise ValueError('Original run is outside the bound original roots.')
        if trial['method'] not in METHODS:
            raise ValueError('Unrelated method in momentum diagnostic replay.')
        if Path(trial['instance']).name != trial['instance']:
            raise ValueError('Instance must be a filename.')
        if set(trial['source_snapshot_sha256']) != set(SOURCES):
            raise ValueError('Incomplete original source binding.')
        if trial['artifact_binding'] not in {'frozen_at_preparation', 'freeze_terminal_before_replay'}:
            raise ValueError('Unknown original artifact binding policy.')
        if trial['artifact_binding'] == 'frozen_at_preparation' and set(trial['original_artifact_sha256']) != {'metadata.json', 'trace.csv', 'final.npz'}:
            raise ValueError('Incomplete frozen original artifact binding.')
    if plan['run_count'] != len(ids):
        raise ValueError('Declared replay count mismatch.')
    return plan


def build_plans(study_root, companion_root):
    study_root, companion_root = Path(study_root).resolve(), canonical_absolute(companion_root)
    tuning_path = study_root / 'configs/stage_c_manifests/tuning.json'
    dispatch = read_json(study_root / 'configs/stage_c_final_dispatch.json')
    pilot_cfg_path = study_root / 'configs/stage_c_pilot.json'
    pilot_cfg = read_json(pilot_cfg_path)
    plans = {}
    all_originals = set()
    data_roots = set()

    def make_plan(phase, runs, inputs, dependency):
        roots = sorted({r.pop('_original_root') for r in runs})
        value = dict(schema_version=1, purpose='missing_momentum_diagnostics_only', phase=phase,
                     original_roots=roots, out_root=str(companion_root / phase),
                     data_root=next(iter(data_roots)), run_count=len(runs), runs=runs,
                     scientific_inputs=inputs, dependency_policy=dependency,
                     selection_policy='All prescribed smoothing trajectories; no outcome filtering or retuning.',
                     replay_policy='Use each trial archived source and original completed update count; no new optimizer budget.',
                     timing_policy='Companion replay timing is separate; never replace original wall-time measurements.',
                     generator_sha256=digest(__file__))
        plans[phase] = validate_plan(value)

    tuning_sha = digest(tuning_path)
    tuning_runs = []
    for spec in read_json(tuning_path)['runs']:
        if spec['method'] not in METHODS:
            continue
        local = study_root / 'ruche_results/tuning_aaeb5fa' / safe_relative(spec['run_id'])
        meta, sources, artifacts = bind_archive(local, run_spec=spec, manifest_sha256=tuning_sha, study_root=study_root)
        original = canonical_absolute(meta['config']['out'])
        data_roots.add(str(canonical_absolute(meta['config']['instance']).parent))
        original_root = original
        for _ in Path(spec['run_id']).parts:
            original_root = original_root.parent
        tuning_runs.append(dict(run_id=spec['run_id'], original_phase='tuning', method=spec['method'],
            original_run_dir=str(original), _original_root=str(original_root),
            original_manifest_sha256=tuning_sha, original_run_spec=spec,
            original_config=meta['config'], original_source_commit=meta['source_commit'],
            source_snapshot_sha256=sources, original_artifact_sha256=artifacts,
            artifact_binding='frozen_at_preparation', instance=spec['instance'],
            instance_sha256=spec['instance_sha256']))
    if len(data_roots) != 1:
        raise ValueError('Original data roots disagree.')
    make_plan('tuning', tuning_runs, {str(tuning_path.relative_to(study_root)): tuning_sha},
              'Original tuning collection is terminal and hash-verified; no dependency on final array.')

    pilot_runs = []
    for filename in pilot_cfg['instances']:
        for method in ('smoothing-fw', 'smoothing-fw-exact'):
            local = study_root / 'ruche_results/1961459' / Path(filename).stem / method
            meta, sources, artifacts = bind_archive(local, study_root=study_root)
            if (meta['config']['method'] != 'smoothing-fw'
                    or meta['config']['oracle'] != ('exact' if method.endswith('-exact') else 'sketch')
                    or Path(meta['config']['instance']).name != filename):
                raise ValueError('Pilot archive does not match the prespecified dataset/method.')
            original = canonical_absolute(meta['config']['out'])
            data_roots.add(str(canonical_absolute(meta['config']['instance']).parent))
            pilot_runs.append(dict(run_id=f'pilot/{Path(filename).stem}/{method}', original_phase='pilot',
                method=method, original_run_dir=str(original), _original_root=str(original.parent.parent),
                original_manifest_sha256=None, original_run_spec=None, original_config=meta['config'],
                original_source_commit=meta['source_commit'], source_snapshot_sha256=sources,
                original_artifact_sha256=artifacts, artifact_binding='frozen_at_preparation',
                instance=filename, instance_sha256=meta['instance_sha256']))
    make_plan('pilot', pilot_runs, {str(pilot_cfg_path.relative_to(study_root)): digest(pilot_cfg_path)},
              'Original pilot archive is terminal and hash-verified; no dependency on final array.')

    final_runs, final_inputs = [], {}
    for phase in ('final_product', 'final_randomized'):
        path = study_root / f'configs/stage_c_final_manifests/{phase}.json'
        sha = digest(path)
        roots = {task['out_root'] for task in dispatch['tasks'] if task['manifest_sha256'] == sha}
        if len(roots) != 1:
            raise ValueError(f'Frozen dispatch must bind exactly one root for {phase}.')
        original_root = canonical_absolute(roots.pop())
        sources = revision_hashes(dispatch['source_commit'], study_root)
        final_inputs[str(path.relative_to(study_root))] = sha
        for spec in read_json(path)['runs']:
            if spec['method'] not in METHODS:
                continue
            final_runs.append(dict(run_id=spec['run_id'], original_phase=phase, method=spec['method'],
                original_run_dir=str(original_root / safe_relative(spec['run_id'])),
                _original_root=str(original_root), original_manifest_sha256=sha, original_run_spec=spec,
                original_config=None, original_source_commit=dispatch['source_commit'],
                source_snapshot_sha256=sources, original_artifact_sha256=None,
                artifact_binding='freeze_terminal_before_replay', instance=spec['instance'],
                instance_sha256=spec['instance_sha256']))
    final_inputs['configs/stage_c_final_dispatch.json'] = digest(study_root / 'configs/stage_c_final_dispatch.json')
    make_plan('finals', final_runs, final_inputs,
              'Start only after the whole original final array 1961580 terminates; every assigned original must be terminal before replay.')
    if {key: len(plan['runs']) for key, plan in plans.items()} != {'tuning': 504, 'pilot': 6, 'finals': 286}:
        raise ValueError('Prespecified replay coverage differs from 504 tuning + 6 pilot + 286 finals.')
    if len(data_roots) != 1:
        raise ValueError('Pilot/tuning data roots disagree.')
    input_hashes = {name: digest(study_root / 'data' / name)
                    for name in {r['instance'] for p in plans.values() for r in p['runs']}}
    for plan in plans.values():
        for trial in plan['runs']:
            if trial['original_run_dir'] in all_originals:
                raise ValueError('Original trajectory occurs in more than one replay phase.')
            all_originals.add(trial['original_run_dir'])
            if input_hashes[trial['instance']] != trial['instance_sha256']:
                raise ValueError(f'Canonical instance hash mismatch: {trial["instance"]}')
        plan['method_run_counts'] = dict(sorted(Counter(r['method'] for r in plan['runs']).items()))
    for a in plans.values():
        for b in plans.values():
            if any(overlaps(a['out_root'], root) for root in b['original_roots']):
                raise ValueError('Companion root overlaps an original from another phase.')
    return plans


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study-root', default=str(ROOT))
    p.add_argument('--companion-root', required=True, help='Canonical absolute separate output parent on execution host.')
    p.add_argument('--out', required=True, help='New local manifest directory; never overwrite.')
    args = p.parse_args()
    plans = build_plans(args.study_root, args.companion_root)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    for phase, plan in plans.items():
        path = out / f'{phase}.json'
        path.write_text(json.dumps(plan, indent=2, sort_keys=True, allow_nan=False) + '\n')
        print(json.dumps(dict(path=str(path.resolve()), sha256=digest(path), run_count=plan['run_count'])))


if __name__ == '__main__':
    main()

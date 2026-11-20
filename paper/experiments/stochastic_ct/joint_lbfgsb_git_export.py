"""Archive joint L-BFGS-B evidence with the existing guarded Git transaction.

An isolated instance of the previous exporter supplies its tested path guards,
commit checks and push-retry behavior. Its archive destination is configured for
this successor; the previous exporter and previous archives are never edited.
"""

import argparse
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

_source = Path(__file__).with_name('nonnegative_git_export.py')
_spec = importlib.util.spec_from_file_location('joint_lbfgsb_git_transaction', _source)
backend = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(backend)
ARCHIVE = 'results/stochastic_matrix_factorization/ct_joint_lbfgsb'
backend.ARCHIVE = ARCHIVE
BUNDLE = backend.BUNDLE
REPORT_FILES = backend.REPORT_FILES
allowed_path = backend.allowed_path
read, write, digest = backend.read, backend.write, backend.digest
require, record, git = backend.require, backend.record, backend.git
atomic_copy, now = backend.atomic_copy, backend.now


def scientific_inputs(root):
    audit = read(root / 'report_update.json')
    require(audit['status'] == 'completed_published' and audit['physical_runs'] == 532
            and audit['figures'] == 2 and audit['tables'] == 1, 'Joint comparison is not validated and published')
    summary = Path(audit['comparison_summary'])
    require(digest(summary) == audit['comparison_summary_sha256'], 'Published joint summary changed')
    paths = set()
    for phase, count in (('development', 12), ('evaluation', 28)):
        plan = read(root / f'{phase}_plan.json')
        validation = read(root / ('development_validation.json' if phase == 'development' else 'validation_summary.json'))
        require(validation['status'] == 'completed_validated', 'Campaign is not validated')
        cases = {c['run_id']: c for c in validation['cases']}
        require(len(cases) == count == len(plan['runs']) and set(cases) == {c['run_id'] for c in plan['runs']}, 'Joint export cohort is incomplete')
        for cfg in plan['runs']:
            case = cases[cfg['run_id']]
            require(cfg['method'] == case['method'] == 'joint_lbfgsb', 'Unexpected exported method')
            folder = root / 'runs' / cfg['run_id']
            require(Path(case['source_directory']).resolve() == folder.resolve(), 'Run source escapes campaign')
            for name in ('metadata.json', 'result.json', 'trace.json'):
                path = folder / name
                require(digest(path) == case[name.replace('.json', '_sha256')], 'Validated evidence changed: ' + str(path))
                paths.add(path)
    paths |= {p for p in root.glob('*.json') if not p.name.startswith('git_export') and p.name not in ('progress.json', 'staged_report.json')}
    paths.add(root / 'git_export_helper.json')
    paths |= {p for p in (root / 'frozen_sources').iterdir() if p.is_file() and p.suffix in ('.py', '.md')}
    if (root / 'README.md').is_file():
        paths.add(root / 'README.md')
    stage = Path(audit['stage'])
    require(stage.parent.resolve() == root.resolve() and stage.name.startswith('report_stage_'), 'Unexpected report stage')
    paths |= set(stage.glob('*.json'))
    for name, expected in audit['statistics'].items():
        require(digest(stage / name) == expected, 'Published trajectory statistics changed')
    return audit, paths


def refresh_bundle(root, bundle, audit, original):
    # The old renderer adds its own nonnegative link relocation. Remove that
    # one from the incoming list first, so it is not duplicated on this update.
    previous = json.loads(json.dumps(original))
    previous['appendix_link_relocations'] = [r for r in previous['appendix_link_relocations'] if r['from'] != '](../../nonnegative_baselines/']
    inputs = backend.refresh_bundle(root, bundle, audit, previous)
    manifest = read(bundle / 'manifest.json')
    manifest['appendix_link_relocations'].append({'from': '](../../joint_lbfgsb/', 'to': '](../../../' + ARCHIVE + '/'})
    appendix = bundle / 'provenance/CT_REPRODUCIBILITY.source.md'
    text = appendix.read_text()
    for relocation in manifest['appendix_link_relocations']:
        text = text.replace(relocation['from'], relocation['to'])
    target = bundle / 'CT_REPRODUCIBILITY.md'
    target.write_text(text)
    manifest['files']['CT_REPRODUCIBILITY.md'].update(bytes=target.stat().st_size, sha256=digest(target))
    snapshot = manifest['snapshot']
    snapshot.update(physical_trajectories=532, added_nonnegative_trajectories=84,
        added_joint_lbfgsb_trajectories=28,
        domains='Six original spectral-constrained arms; three nonnegative-only baselines.')
    manifest['provenance']['publication_audit']['archive_study'] = 'ct_joint_lbfgsb'
    manifest['provenance']['predecessor_document_revisions'] = original.get('document_revisions', [])
    manifest['provenance']['scientific_content'] = 'Current report files and source appendix preserve their published Elite bytes. Prior numerical evidence and the gradient-estimator explanation are retained. Only public appendix links are relocated.'
    manifest['document_revisions'] = []
    require('Stochastic gradient estimator.' in (bundle / 'report/ct_experiments.tex').read_text(), 'Gradient-estimator explanation was lost')
    remainder = (bundle / 'README.md').read_text().split('## Validate', 1)[1]
    intro = ('# Stochastic CT paper section\n\n'
        'This bundle compares SFRAMES, SPRING–SARAH, ProxSARAH and PALM with nonnegative-only '
        'alternating NNLS, multiplicative updates and joint L-BFGS-B on 2DeteCT. The latter three '
        'methods have no spectral constraints. The two mean ±1 SD figures and endpoint table cover '
        '532 physical trajectories at a 69,120-view cap per direction. All joint L-BFGS-B '
        'line-search work is charged; historical curvature work remains separate.\n\n'
        'The [PDF](report/sframes_ct_section.pdf), LaTeX, figures and '
        '[public appendix](CT_REPRODUCIBILITY.md) are bundled. '
        '[Joint L-BFGS-B evidence](../../../' + ARCHIVE + '/archive_manifest.json), '
        '[NNLS/MU evidence](../../../results/stochastic_matrix_factorization/ct_nonnegative/archive_manifest.json), '
        'and [original evidence](../../../results/stochastic_matrix_factorization/ct/archive_manifest.json) '
        'retain frozen sources, settings and trajectories. The [manifest](manifest.json) records '
        'hashes and predecessor provenance. The gradient-estimator explanation is preserved.\n\n## Validate')
    readme = bundle / 'README.md'
    readme.write_text(intro + remainder)
    manifest['files']['README.md'].update(bytes=readme.stat().st_size, sha256=digest(readme))
    write(bundle / 'manifest.json', manifest)
    return inputs


def build_stage(root, repo):
    root, repo = Path(root).resolve(), Path(repo).resolve()
    helper = read(root / 'git_export_helper.json')
    head = helper['expected_head']
    plan_path = root / 'git_export_plan.json'
    if plan_path.exists():
        plan = read(plan_path)
        require(plan['expected_head'] == head, 'Export belongs to another Git base')
        backend.validate_stage_files(plan)
        for path, expected in plan['input_files'].items():
            require(digest(path) == expected, 'Export input changed after staging')
        return plan
    audit, sources = scientific_inputs(root)
    stage = root / 'git_export_stage'
    require(not stage.exists(), 'Partial export stage retained; review before retry')
    files = stage / 'files'
    archive = files / ARCHIVE
    archive.mkdir(parents=True)
    inputs, records = {}, []
    for source in sorted(sources):
        relative = str(source.relative_to(root))
        target = archive / relative
        atomic_copy(source, target)
        sha = digest(source)
        require(digest(target) == sha, 'Archive copy changed bytes')
        inputs[str(source)] = sha
        records.append(dict(path=relative, size=target.stat().st_size, sha256=sha))
    doc = archive / 'ARCHIVE.md'
    doc.write_text('# Joint L-BFGS-B CT evidence\n\n'
        'This archive preserves 12 development runs selecting L-BFGS-B memory and 28 deterministic '
        'evaluation runs without spectral constraints, including every accepted joint update and '
        'terminal charged-work state. All solver forward/adjoint calls, including rejected line-search '
        'trials, are charged. Previous comparisons remain in the separate ct and ct_nonnegative archives.\n\n'
        'Files retain runtime-relative paths and original bytes. Embedded absolute paths are provenance. '
        'The manifest records sizes and SHA-256 hashes. Raw CT and calibration data, optimizer checkpoints, '
        'environments, fixtures, caches and transient logs remain outside Git.\n\n'
        'See the [paper bundle](../../../paper/experiments/stochastic_ct/) for the report and public appendix.\n')
    records.append(dict(path='ARCHIVE.md', size=doc.stat().st_size, sha256=digest(doc), origin='generated_archive_documentation'))
    write(archive / 'archive_manifest.json', dict(source_root=str(root), generated_utc=now(), files=records,
        compressed_archives=[], exclusions=['Raw/preprocessed data and calibration arrays; optimizer checkpoints.',
        'Environments, fixtures, caches, locks and transient logs. Previous archives remain unchanged.']))
    original = backend.bundle_from_commit(root, repo, files, head)
    inputs.update(refresh_bundle(root, files / BUNDLE, audit, original))
    validator = backend.load_module('joint_export_bundle', files / BUNDLE / 'bundle.py')
    checked = validator.validate_bundle(files / BUNDLE, require_archive=False)
    for filename in ('README.md', 'CT_REPRODUCIBILITY.md'):
        for _, relative in validator.local_markdown_links(files / BUNDLE / filename):
            destination = ((files / BUNDLE) / relative).resolve()
            relative_to_stage = destination.relative_to(files.resolve())
            require(destination.exists() or (repo / relative_to_stage).exists(), 'Missing archived link: ' + relative)
    changes = {}
    for path in files.rglob('*'):
        if not path.is_file():
            continue
        name = str(path.relative_to(files))
        old = git(repo, 'show', head + ':' + name, binary=True, check=False)
        if old.returncode == 0 and old.stdout == path.read_bytes():
            continue
        require(backend.allowed_path(name), 'Unexpected source/archive mutation: ' + name)
        changes[name] = record(path)
    plan = dict(stage_files=str(files), files=changes, input_files=inputs, expected_head=head,
        expected_branch=helper['expected_branch'], bundle_validation=checked, archive_files=len(records))
    backend.validate_stage_files(plan)
    for path, expected in inputs.items():
        require(digest(path) == expected, 'Scientific source changed during staging')
    write(plan_path, plan)
    return plan


# Configure only this private module instance; the old exporter keeps its own globals.
backend.build_stage = build_stage
publish_stage = backend.publish_stage
export = backend.export


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--publish', action='store_true')
    args = parser.parse_args()
    with (args.root / 'git_export.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = export(args.root, args.repo, args.publish)
    print(json.dumps(result, indent=2))
    if result['status'] not in ('completed_pushed', 'staged_validated'):
        raise SystemExit(2)


if __name__ == '__main__':
    main()

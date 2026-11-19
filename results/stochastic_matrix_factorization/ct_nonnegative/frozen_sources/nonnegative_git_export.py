"""Archive validated nonnegative CT results and publish a scoped Git update.

The scientific export is built and checked before touching the checkout. Git
publication requires the frozen branch/HEAD and a clean checkout. Failures keep
the stage and an explicit pending receipt; no resets, force pushes or merges are
performed. A retry after a failed push reuses the recorded commit.
"""

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys


BUNDLE = 'paper/experiments/stochastic_ct'
ARCHIVE = 'results/stochastic_matrix_factorization/ct_nonnegative'
REPORT_FILES = ('ct_experiments.tex', 'sframes_ct_section.tex', 'ct_reproducibility_link.tex',
                'sframes_ct_section.pdf', 'trajectories_360.pdf', 'trajectories_360.png',
                'trajectories_1080.pdf', 'trajectories_1080.png')
BUNDLE_MUTATIONS = {'manifest.json', 'README.md', 'CT_REPRODUCIBILITY.md',
                    'provenance/CT_REPRODUCIBILITY.source.md',
                    *('report/' + name for name in REPORT_FILES)}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def git(repo, *args, binary=False, check=True):
    result = subprocess.run(['git', '-C', str(repo), *args], capture_output=True,
                            text=not binary, check=False)
    if check and result.returncode:
        message = result.stderr.decode(errors='replace') if binary else result.stderr
        raise RuntimeError('git ' + args[0] + ' failed: ' + message.strip())
    return result.stdout if check else result


def safe_path(name):
    path = PurePosixPath(name)
    require(not path.is_absolute() and path.parts and '..' not in path.parts,
            'Export path must stay inside the repository')
    return str(path)


def allowed_path(name):
    name = safe_path(name)
    return name.startswith(ARCHIVE + '/') or (
        name.startswith(BUNDLE + '/') and name[len(BUNDLE)+1:] in BUNDLE_MUTATIONS)


def record(path):
    return dict(size=Path(path).stat().st_size, sha256=digest(path))


def atomic_copy(source, destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + '.export-tmp')
    temporary.write_bytes(Path(source).read_bytes())
    temporary.replace(destination)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    previous = sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode = True
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
    return module


def scientific_inputs(root):
    """Reject incomplete/tampered run cohorts before constructing any export."""
    audit = read(root / 'report_update.json')
    require(audit['status'] == 'completed_published' and audit['physical_runs'] == 504
            and audit['figures'] == 2 and audit['tables'] == 1, 'Report is not the validated final comparison')
    summary = Path(audit['comparison_summary'])
    require(digest(summary) == audit['comparison_summary_sha256'], 'Published numerical summary changed')
    paths = set()
    for phase, count in (('development', 12), ('evaluation', 56)):
        plan = read(root / f'{phase}_plan.json')
        validation = read(root / ('development_validation.json' if phase == 'development' else 'validation_summary.json'))
        require(validation['status'] == 'completed_validated', 'Campaign validation is incomplete')
        cases = {case['run_id']: case for case in validation['cases']}
        require(len(cases) == count == len(plan['runs'])
                and set(cases) == {cfg['run_id'] for cfg in plan['runs']}, 'Export cohort is incomplete')
        for cfg in plan['runs']:
            case = cases[cfg['run_id']]
            folder = root / 'runs' / cfg['run_id']
            require(Path(case['source_directory']).resolve() == folder.resolve(), 'Run source leaves the campaign')
            for name, field in (('metadata.json', 'metadata_sha256'), ('result.json', 'result_sha256'), ('trace.json', 'trace_sha256')):
                path = folder / name
                require(digest(path) == case[field], 'Validated run evidence changed: ' + str(path))
                paths.add(path)
    for path in root.glob('*.json'):
        if not path.name.startswith('git_export') and path.name not in ('progress.json', 'staged_report.json'):
            paths.add(path)
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


def bundle_from_commit(root, repo, files, head):
    """Use committed predecessor bytes, never unpublished checkout edits."""
    names = git(repo, 'ls-tree', '-rz', '--name-only', head, '--', BUNDLE, binary=True).split(b'\0')
    backup = root / 'git_export_previous_bundle'
    for encoded in names:
        if not encoded:
            continue
        name = safe_path(encoded.decode())
        raw = git(repo, 'show', head + ':' + name, binary=True)
        relative = name[len(BUNDLE)+1:]
        for target in (files / name, backup / relative):
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                require(target.read_bytes() == raw, 'Previous bundle backup changed')
            else:
                target.write_bytes(raw)
    return read(files / BUNDLE / 'manifest.json')


def refresh_bundle(root, bundle, audit, original):
    source_pdf = Path(audit['report_pdf'])
    report = source_pdf.parent
    appendix = report.parent / 'CT_REPRODUCIBILITY.md'
    source_root = root.parents[2]
    inputs = {}
    manifest = dict(original)
    manifest['files'] = dict(original['files'])

    def capture(source, relative, raw=None, copy_mode='byte-identical'):
        original_bytes = source.read_bytes()
        data = original_bytes if raw is None else raw
        target = bundle / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        original_sha = hashlib.sha256(original_bytes).hexdigest()
        inputs[str(source)] = original_sha
        manifest['files'][relative] = dict(bytes=len(data), sha256=hashlib.sha256(data).hexdigest(),
            copy_mode=copy_mode, original_sha256=original_sha,
            source_relative_path=str(source.relative_to(source_root)))

    for name in REPORT_FILES:
        require(digest(report / name) == audit['published_sha256'][name], 'Published report asset changed')
        capture(report / name, 'report/' + name)
    require(digest(appendix) == audit['published_sha256']['CT_REPRODUCIBILITY.md'], 'Published appendix changed')
    capture(appendix, 'provenance/CT_REPRODUCIBILITY.source.md')
    replacements = list(original['appendix_link_relocations'])
    replacements.append(dict(**{'from': '](../../nonnegative_baselines/',
        'to': '](../../../results/stochastic_matrix_factorization/ct_nonnegative/'}))
    text = appendix.read_text()
    for replacement in replacements:
        text = text.replace(replacement['from'], replacement['to'])
    capture(appendix, 'CT_REPRODUCIBILITY.md', text.encode(), 'relative-links-only')
    manifest['appendix_link_relocations'] = replacements
    summary = read(audit['comparison_summary'])
    manifest['snapshot'] = dict(date=audit['published_utc'][:10], published_utc=audit['published_utc'],
        comparison='equal directional solver-view budget', solver_view_cap_per_direction=69120,
        physical_trajectories=504, added_nonnegative_trajectories=56, continued_trajectories=168,
        native_iteration_observations=sum(c['observations'] for c in summary['cases']),
        pages=audit['pages'], figures=2, tables=1,
        domains='Six original spectral-constrained arms; two additional nonnegative-only baselines.',
        curvature_accounting='Historical curvature-estimation work remains excluded and separately recorded.')
    manifest['bibliography_keys'] = sorted(re.findall(r'\\bibitem\{([^}]+)\}', (bundle / 'report/sframes_ct_section.tex').read_text()))
    manifest['provenance'] = dict(original['provenance'], publication_audit=dict(
        archive_study='ct_nonnegative', archive_relative_path='report_update.json', sha256=digest(root / 'report_update.json')),
        predecessor_snapshot=original['snapshot'], predecessor_pdf_sha256=original['files']['report/sframes_ct_section.pdf']['sha256'])
    remainder = (bundle / 'README.md').read_text().split('## Validate', 1)[1]
    intro = ('# Stochastic CT paper section\n\n'
        'This bundle contains the published comparison of SFRAMES, SPRING–SARAH, ProxSARAH and PALM '
        'with nonnegative-only alternating NNLS and multiplicative updates on 2DeteCT. The two new '
        'baselines have no spectral constraints. The same two mean ±1 SD figures and endpoint table '
        'cover 504 physical trajectories at a 69,120-view cap per direction. Every operator call '
        'inside the new solvers is charged; historical curvature estimates remain recorded separately.\n\n'
        'The exact [PDF](report/sframes_ct_section.pdf), LaTeX, figures, and '
        '[public appendix](CT_REPRODUCIBILITY.md) are bundled. The '
        '[new evidence](../../../results/stochastic_matrix_factorization/ct_nonnegative/archive_manifest.json) '
        'and [original evidence](../../../results/stochastic_matrix_factorization/ct/archive_manifest.json) '
        'retain frozen sources, configurations and trajectories. The [manifest](manifest.json) '
        'records original hashes and the predecessor snapshot. Only appendix links are relocated; '
        'its [original bytes](provenance/CT_REPRODUCIBILITY.source.md) are retained.\n\n## Validate')
    (bundle / 'README.md').write_text(intro + remainder)
    manifest['files']['README.md'] = dict(bytes=(bundle / 'README.md').stat().st_size,
        sha256=digest(bundle / 'README.md'), copy_mode='archive-support-file')
    write(bundle / 'manifest.json', manifest)
    return inputs


def validate_stage_files(plan):
    stage = Path(plan['stage_files'])
    require(plan['files'], 'Empty export')
    for name, expected in plan['files'].items():
        require(allowed_path(name), 'Unapproved Git path: ' + name)
        target = stage / name
        require(target.is_file() and not target.is_symlink() and record(target) == expected,
                'Staged export changed: ' + name)


def build_stage(root, repo):
    root, repo = Path(root).resolve(), Path(repo).resolve()
    helper = read(root / 'git_export_helper.json')
    head = helper['expected_head']
    plan_path = root / 'git_export_plan.json'
    if plan_path.exists():
        plan = read(plan_path)
        require(plan['expected_head'] == head, 'Stage belongs to another Git base')
        validate_stage_files(plan)
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
    archive_doc = ('# Nonnegative-only CT baseline evidence\n\n'
        'This successor archive preserves the completed alternating NNLS and multiplicative-update '
        'campaign without spectral constraints. It contains 12 development runs and 56 deterministic '
        'evaluation runs, including every recorded native and terminal state. All solver forward/adjoint '
        'work is charged, including inner trials and the cached multiplicative-update right-hand side. '
        'The existing SFRAMES and other original results are retained in the separate `ct` archive.\n\n'
        'Files retain their original bytes and runtime-relative paths. Absolute paths inside them are '
        'historical provenance. `archive_manifest.json` records sizes and SHA-256 hashes. Raw CT data, '
        'calibration arrays, checkpoints, environments, caches and transient logs remain outside Git. '
        'This archive supports result inspection and plotting; restarting optimization needs the '
        'external datasets and checkpoints. The public paper bundle is at '
        '[paper/experiments/stochastic_ct](../../../paper/experiments/stochastic_ct/).\n')
    (archive / 'ARCHIVE.md').write_text(archive_doc)
    records.append(dict(path='ARCHIVE.md', size=(archive / 'ARCHIVE.md').stat().st_size,
                        sha256=digest(archive / 'ARCHIVE.md'), origin='generated_archive_documentation'))
    write(archive / 'archive_manifest.json', dict(source_root=str(root), generated_utc=now(), files=records,
        compressed_archives=[], exclusions=['Raw/preprocessed CT and calibration arrays; optimizer checkpoints.',
        'Environments, third-party dependencies, fixture output, caches, locks and transient logs.',
        'The prior ct evidence archive is retained separately and never overwritten.']))
    original = bundle_from_commit(root, repo, files, head)
    inputs.update(refresh_bundle(root, files / BUNDLE, audit, original))
    validator = load_module('nonnegative_export_bundle', files / BUNDLE / 'bundle.py')
    checked = validator.validate_bundle(files / BUNDLE, require_archive=False)
    for filename in ('README.md', 'CT_REPRODUCIBILITY.md'):
        for _, relative in validator.local_markdown_links(files / BUNDLE / filename):
            destination = ((files / BUNDLE) / relative).resolve()
            relative_to_stage = destination.relative_to(files.resolve())
            require(destination.exists() or (repo / relative_to_stage).exists(),
                    'Missing relocated appendix evidence: ' + relative)
    changes = {}
    for path in files.rglob('*'):
        if not path.is_file():
            continue
        name = str(path.relative_to(files))
        old = git(repo, 'show', head + ':' + name, binary=True, check=False)
        if old.returncode == 0 and old.stdout == path.read_bytes():
            continue
        require(allowed_path(name), 'Unexpected report/source mutation: ' + name)
        changes[name] = record(path)
    plan = dict(stage_files=str(files), files=changes, input_files=inputs,
                expected_head=head, expected_branch=helper['expected_branch'],
                bundle_validation=checked, archive_files=len(records))
    validate_stage_files(plan)
    for path, expected in inputs.items():
        require(digest(path) == expected, 'Scientific input changed while staging')
    write(plan_path, plan)
    return plan


def remote_head(repo, helper):
    output = git(repo, 'ls-remote', '--heads', helper['remote'], 'refs/heads/' + helper['remote_branch']).strip()
    require(output and len(output.splitlines()) == 1, 'Expected remote branch is missing or ambiguous')
    return output.split()[0]


def receipt(root, status, **values):
    value = dict(status=status, updated_utc=now(), **values)
    write(root / 'git_export.json', value)
    return value


def check_export_commit(repo, commit, plan, parent):
    require(git(repo, 'rev-parse', commit + '^').strip() == parent, 'Export commit has an unexpected parent')
    names = set(git(repo, 'diff-tree', '--no-commit-id', '--name-only', '-r', commit).splitlines())
    require(names == set(plan['files']), 'Export commit contains unexpected changes')
    for name, expected in plan['files'].items():
        raw = git(repo, 'show', commit + ':' + name, binary=True)
        require(len(raw) == expected['size'] and hashlib.sha256(raw).hexdigest() == expected['sha256'],
                'Committed evidence differs from the validated stage')


def publish_stage(root, repo, plan, helper):
    root, repo = Path(root).resolve(), Path(repo).resolve()
    previous = read(root / 'git_export.json') if (root / 'git_export.json').exists() else {}
    commit = previous.get('commit')
    try:
        validate_stage_files(plan)
        for path, expected in plan.get('input_files', {}).items():
            require(digest(path) == expected, 'Scientific input changed before Git publication')
        require(git(repo, 'rev-parse', '--show-toplevel').strip() == str(repo), 'Unexpected repository root')
        require(helper['expected_branch'] == plan['expected_branch'] == 'stochastic-frames', 'Wrong authorized branch')
        require(git(repo, 'branch', '--show-current').strip() == helper['expected_branch'], 'Target branch changed')
        require(helper['expected_head'] == plan['expected_head'], 'Expected Git base changed')
        require(git(repo, 'remote', 'get-url', helper['remote']).strip() == helper['remote_url'], 'Remote destination changed')
        require(not git(repo, 'status', '--porcelain=v1', '--untracked-files=all'), 'Concurrent or uncommitted checkout changes')
        head = git(repo, 'rev-parse', 'HEAD').strip()
        if commit:
            require(head == commit, 'Pending export commit is no longer the checkout tip')
            check_export_commit(repo, commit, plan, helper['expected_head'])
            for name, expected in plan['files'].items():
                require(record(repo / name) == expected, 'Pending committed export changed')
        else:
            require(head == helper['expected_head'], 'Checkout HEAD advanced during the campaign')
        remote = remote_head(repo, helper)
        if commit and remote == commit:
            return receipt(root, 'completed_pushed', commit=commit, expected_head=helper['expected_head'],
                           branch=helper['expected_branch'], remote=helper['remote'], files=len(plan['files']))
        require(remote == helper['expected_head'], 'Remote branch advanced; a reviewed integration is required')
        if not commit:
            stage = Path(plan['stage_files'])
            for name in sorted(plan['files']):
                target = repo / name
                old = git(repo, 'show', head + ':' + name, binary=True, check=False)
                if old.returncode == 0:
                    require(target.is_file() and target.read_bytes() == old.stdout,
                            'Checkout file changed before installation: ' + name)
                else:
                    require(not target.exists(), 'Untracked destination appeared: ' + name)
                atomic_copy(stage / name, target)
            git(repo, 'add', '-f', '--', *sorted(plan['files']))
            changed = git(repo, 'status', '--porcelain=v1', '-z', '--untracked-files=all').split('\0')
            require({line[3:] for line in changed if line} == set(plan['files']), 'Concurrent edits appeared during installation')
            require(set(git(repo, 'diff', '--cached', '--name-only').splitlines()) == set(plan['files']),
                    'Git index contains unrelated changes')
            require(git(repo, 'rev-parse', 'HEAD').strip() == head, 'HEAD changed before commit')
            git(repo, 'commit', '-m', 'Add nonnegative CT baselines and update the experiment report')
            commit = git(repo, 'rev-parse', 'HEAD').strip()
            check_export_commit(repo, commit, plan, head)
            receipt(root, 'pending_push', commit=commit, expected_head=head, branch=helper['expected_branch'])
        require(git(repo, 'rev-parse', 'HEAD').strip() == commit
                and not git(repo, 'status', '--porcelain=v1', '--untracked-files=all'),
                'Concurrent checkout changes before push')
        git(repo, 'push', helper['remote'], commit + ':refs/heads/' + helper['remote_branch'])
        require(remote_head(repo, helper) == commit, 'Remote did not confirm the exported commit')
        return receipt(root, 'completed_pushed', commit=commit, expected_head=helper['expected_head'],
                       branch=helper['expected_branch'], remote=helper['remote'], files=len(plan['files']))
    except Exception as error:
        return receipt(root, 'pending_push' if commit else 'pending_git_review', error=str(error),
                       commit=commit, expected_head=helper.get('expected_head'), stage_files=plan.get('stage_files'))


def export(root, repo, publish=False):
    root, repo = Path(root).resolve(), Path(repo).resolve()
    helper = read(root / 'git_export_helper.json')
    try:
        for path, expected in helper.get('dependencies', {}).items():
            require(digest(path) == expected, 'Git export dependency changed: ' + path)
        plan = build_stage(root, repo)
        if publish:
            return publish_stage(root, repo, plan, helper)
        return receipt(root, 'staged_validated', expected_head=plan['expected_head'], files=len(plan['files']),
                       stage_files=plan['stage_files'])
    except Exception as error:
        previous = read(root / 'git_export.json') if (root / 'git_export.json').exists() else {}
        return receipt(root, 'pending_push' if previous.get('commit') else 'pending_git_review',
                       error=str(error), commit=previous.get('commit'), stage_files=str(root / 'git_export_stage/files'))


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

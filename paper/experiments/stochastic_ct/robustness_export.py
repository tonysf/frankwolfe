"""Guarded archival/publication of the expanded CT campaign.

Reuse the existing reviewed Git transaction, with a private backend instance.
Dense traces are losslessly compressed in bounded-size parts; data and restart
checkpoints remain outside Git. All older result archives are preserved.
"""
import gzip
import importlib.util
from pathlib import Path
import re
import shutil
import tarfile

try:
    from .robustness_report import read, write, digest
except ImportError:
    from robustness_report import read, write, digest

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('_robustness_export_backend', ROOT/'nonnegative_git_export.py')
backend = importlib.util.module_from_spec(spec)
spec.loader.exec_module(backend)
ARCHIVE = 'results/stochastic_matrix_factorization/ct_robustness_20261122'
BUNDLE = backend.BUNDLE
backend.ARCHIVE = ARCHIVE
backend.BUNDLE_MUTATIONS |= {'ROBUSTNESS_RESULTS.md', 'ROBUSTNESS_EVALUATION.md'}
original_git = backend.git


def named_git(repo, *args, **kwargs):
    if args == ('commit', '-m', 'Add nonnegative CT baselines and update the experiment report'):
        args = ('commit', '-m', 'Publish the expanded SFRAMES mechanism and robustness evaluation')
    return original_git(repo, *args, **kwargs)


backend.git = named_git


def pack_runs(root, archive, limit=32*1024*1024):
    """Pack at most 32 MiB of source payload, with a 50 MiB compressed ceiling."""
    members = sorted(p for p in (root/'runs').glob('*/*.json')
                     if p.name in ('metadata.json', 'trace.json', 'result.json', 'candidate_failure.json'))
    batches, current, size = [], [], 0
    for path in members:
        if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
            raise ValueError('Archive members must be regular files inside the campaign')
        if path.stat().st_size > limit:
            raise ValueError('A single dense trace exceeds the archive part limit; review splitting before publication')
        if current and size + path.stat().st_size > limit:
            batches.append(current)
            current, size = [], 0
        current.append(path)
        size += path.stat().st_size
    if current:
        batches.append(current)
    records = []
    for i, batch in enumerate(batches):
        target = archive/f'runs-{i:04d}.tar.gz'
        descriptions = []
        with target.open('wb') as raw, gzip.GzipFile(fileobj=raw, mode='wb', mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode='w|') as tar:
                for path in batch:
                    name = str(path.relative_to(root))
                    info = tar.gettarinfo(str(path), arcname=name)
                    info.uid = info.gid = info.mtime = 0
                    info.uname = info.gname = ''
                    with path.open('rb') as stream:
                        tar.addfile(info, stream)
                    descriptions.append(dict(path=name, size=path.stat().st_size, sha256=digest(path)))
        if target.stat().st_size >= 50*1024*1024:
            raise ValueError('An archive part exceeds the publication size limit')
        records.append(dict(path=target.name, size=target.stat().st_size, sha256=digest(target), members=descriptions))
    return records


def scientific_gate(root, audit):
    completion = read(root/'completion.json')
    summary = read(root/'report_summary.json')
    if (completion.get('status') != 'completed' or summary.get('stage') != 'completed'
            or summary.get('pending_runs') != []
            or audit.get('status') != 'completed_published'
            or audit['summary_sha256'] != digest(root/'report_summary.json')
            or audit['protocol_sha256'] != digest(root/'protocol.json')
            or completion['protocol_sha256'] != digest(root/'protocol.json')
            or completion['evaluation_runs'] != audit['evaluation_runs']):
        raise ValueError('Incomplete or inconsistent scientific publication')
    try:
        from .robustness_report import load_configs
    except ImportError:
        from robustness_report import load_configs
    configs = load_configs(root)
    if len(configs) != summary['planned_physical_runs']:
        raise ValueError('Publication plan differs from the summary')
    phases = [cfg['phase'] for cfg in configs.values()]
    if any(phase not in ('development', 'evaluation') for phase in phases):
        raise ValueError('Unknown phase in the publication plan')
    failures = summary['failed_runs']
    failed_ids = [item['run_id'] for item in failures]
    if len(set(failed_ids)) != len(failed_ids) or any(key not in configs for key in failed_ids):
        raise ValueError('Publication failure inventory differs from the plan')
    if any(configs[key]['phase'] != 'development' for key in failed_ids):
        raise ValueError('An evaluation failure cannot be published automatically')
    if (summary['completed_physical_runs'] + len(failures) != len(configs)
            or audit['physical_runs'] != summary['completed_physical_runs']
            or audit['development_runs'] != phases.count('development')
            or audit['evaluation_runs'] != phases.count('evaluation')
            or audit['failed_development_runs'] != len(failures)):
        raise ValueError('Scientific publication counts are inconsistent')
    # Inspect persisted outcomes independently of the report's aggregate counts.
    # Dense trace contents were validated before rendering and are hash-pinned
    # during staging; checking their presence here does not parse them again.
    observed_failures = set()
    for key, cfg in configs.items():
        if Path(key).name != key or key in ('.', '..'):
            raise ValueError('Invalid run identifier in the publication plan')
        folder = root/'runs'/key
        failure_path = folder/'candidate_failure.json'
        if failure_path.exists():
            if cfg['phase'] != 'development' or read(failure_path).get('config') != cfg:
                raise ValueError('Invalid retained development failure: '+key)
            status = 'numerical_failure'
        else:
            if not all((folder/name).is_file() for name in ('result.json', 'metadata.json', 'trace.json')):
                raise ValueError('Missing completed run evidence: '+key)
            result, meta = read(folder/'result.json'), read(folder/'metadata.json')
            if meta.get('config') != cfg or result.get('metadata') != meta:
                raise ValueError('Run outcome configuration differs from the plan: '+key)
            status = result.get('status')
        if status not in ('completed', 'numerical_failure') or (cfg['phase'] == 'evaluation' and status != 'completed'):
            raise ValueError('Incomplete or failed evaluation outcome: '+key)
        if status == 'numerical_failure':
            observed_failures.add(key)
    if observed_failures != set(failed_ids) or any(item.get('status') != 'numerical_failure' for item in failures):
        raise ValueError('Persisted run outcomes differ from the publication summary')
    stage = Path(audit['stage'])
    if stage.parent.resolve() != root.resolve() or not stage.name.startswith('report_stage_'):
        raise ValueError('Report stage is outside the campaign')
    for name, expected in audit['published_sha256'].items():
        if name not in backend.REPORT_FILES or digest(stage/'report'/name) != expected:
            raise ValueError('Rendered report changed after validation: '+name)
    if set(audit['published_sha256']) != set(backend.REPORT_FILES):
        raise ValueError('Missing validated report asset')
    for name, expected in audit['statistics'].items():
        if Path(name).name != name or digest(stage/'report'/name) != expected:
            raise ValueError('Rendered trajectory statistics changed')
    supplements = ('CT_REPRODUCIBILITY.md', 'ROBUSTNESS_RESULTS.md', 'ROBUSTNESS_EVALUATION.md')
    if set(audit.get('supplement_sha256', {})) != set(supplements):
        raise ValueError('Missing validated reproducibility supplement')
    for name, expected in audit['supplement_sha256'].items():
        if digest(stage/name) != expected:
            raise ValueError('Reproducibility supplement changed after validation')


def build_stage(root, publication, audit):
    scientific_gate(root, audit)
    repo = Path(publication['repo'])
    plan_path = root/'git_export_plan.json'
    if plan_path.exists():
        plan = read(plan_path)
        backend.validate_stage_files(plan)
        return plan
    output = root/'git_export_stage'
    if output.exists():
        raise ValueError('Partial export retained for review: ' + str(output))
    files = output/'files'
    archive = files/ARCHIVE
    archive.mkdir(parents=True)
    evidence = [p for p in root.glob('*.json') if not p.name.startswith(('git_export','progress','staged_report'))]
    for directory in ('plans', 'stages', 'frozen_sources'):
        evidence += [p for p in (root/directory).rglob('*') if p.is_file() and p.suffix in ('.py','.md','.json')]
    inputs, records = {}, []
    for path in sorted(set(evidence)):
        name = str(path.relative_to(root))
        target = archive/name
        backend.atomic_copy(path, target)
        inputs[str(path)] = digest(path)
        records.append(dict(path=name, **backend.record(target)))
    write(archive/'report_update.json', audit)
    records.append(dict(path='report_update.json', **backend.record(archive/'report_update.json')))
    compressed = pack_runs(root, archive)
    # Pin uncompressed scientific evidence until the Git transaction completes.
    for part in compressed:
        for record in part['members']:
            inputs[str(root/record['path'])] = record['sha256']
    write(archive/'archive_manifest.json', dict(source_root=str(root), files=records,
        compressed_archives=compressed, exclusions=['Raw CT/calibration arrays, optimizer checkpoints, environments, logs and caches.'],
        previous_archives_preserved=True))
    original = backend.bundle_from_commit(root, repo, files, publication['expected_head'])
    bundle = files/BUNDLE
    stage = Path(audit['stage'])
    for name in backend.REPORT_FILES:
        backend.atomic_copy(stage/'report'/name, bundle/'report'/name)
        inputs[str(stage/'report'/name)] = digest(stage/'report'/name)
    for name in ('CT_REPRODUCIBILITY.md','ROBUSTNESS_RESULTS.md','ROBUSTNESS_EVALUATION.md'):
        backend.atomic_copy(stage/name, bundle/name)
        inputs[str(stage/name)] = digest(stage/name)
    for name in audit['statistics']:
        inputs[str(stage/'report'/name)] = digest(stage/'report'/name)
        target = archive/'report'/name
        backend.atomic_copy(stage/'report'/name, target)
        records.append(dict(path=str(target.relative_to(archive)), **backend.record(target)))
    # Statistics are added after the source bundle is staged; include them in
    # the final archive inventory as well as pinning their runtime bytes.
    archive_manifest = read(archive/'archive_manifest.json')
    archive_manifest['files'] = records
    write(archive/'archive_manifest.json', archive_manifest)
    source_appendix = (bundle/'CT_REPRODUCIBILITY.md').read_text()
    for relocation in reversed(original['appendix_link_relocations']):
        source_appendix = source_appendix.replace(relocation['to'], relocation['from'])
    (bundle/'provenance/CT_REPRODUCIBILITY.source.md').write_text(source_appendix)
    (bundle/'README.md').write_text('# Stochastic CT paper section\n\n'
        'This bundle reports the expanded SFRAMES mechanism and robustness evaluation: four training-view counts, '
        'five sampling repetitions, proper deterministic FraMES, batch and averaging ablations, joint tuning, '
        'rank/radius/initialization studies, and six fresh confirmation slices within familiar groups. '
        'The two trajectory figures show group-balanced means and one descriptive standard deviation.\n\n'
        'See the [PDF](report/sframes_ct_section.pdf), [protocol](ROBUSTNESS_EVALUATION.md), '
        '[complete results](ROBUSTNESS_RESULTS.md), and [reproducibility details](CT_REPRODUCIBILITY.md). '
        'Dense traces and source hashes are preserved in the [result archive](../../../'+ARCHIVE+'/archive_manifest.json). '
        'Previous result archives remain unchanged.\n\n'
        'Validate this bundle with `python3 paper/experiments/stochastic_ct/bundle.py --check --require-archive`.\n')
    manifest = original
    manifest['provenance']['predecessor_snapshot'] = dict(original['snapshot'])
    manifest['provenance']['predecessor_pdf_sha256'] = original['files']['report/sframes_ct_section.pdf']['sha256']
    manifest['provenance']['predecessor_document_revisions'] = original.get('document_revisions', [])
    manifest['provenance']['publication_audit'] = dict(archive_study='ct_robustness_20261122',
        archive_relative_path='report_update.json', sha256=digest(archive/'report_update.json'))
    manifest['provenance']['scientific_content'] = 'Expanded campaign evidence and report; previous archives remain unchanged. Public appendix links are relocated.'
    manifest['snapshot'] = dict(date=audit['generated_utc'][:10], published_utc=audit['generated_utc'],
        comparison='equal directional solver-view budget', solver_view_cap_per_direction=69120,
        physical_trajectories=audit['evaluation_runs'], development_runs=audit['development_runs'],
        pages=audit['pages'], figures=2, tables=1,
        domains='Spectral-constrained methods and a nonnegative-only L-BFGS-B application reference.',
        curvature_accounting='Gradient/solver caps; curvature and diagnostic costs recorded separately.')
    manifest['document_revisions'] = []
    manifest['bibliography_keys'] = re.findall(r'\\bibitem\{([^}]+)\}',
        (bundle/'report/sframes_ct_section.tex').read_text())
    for name in ('ROBUSTNESS_RESULTS.md','ROBUSTNESS_EVALUATION.md'):
        manifest['files'][name] = dict(copy_mode='byte-identical', source_relative_path=str((Path(publication['runtime_report']).parent/name).relative_to(root.parents[2])))
    for name, record in manifest['files'].items():
        path = bundle/name
        record.update(bytes=path.stat().st_size, sha256=digest(path))
        if record.get('copy_mode') == 'byte-identical':
            record['original_sha256'] = record['sha256']
    manifest['files']['CT_REPRODUCIBILITY.md']['original_sha256'] = digest(bundle/'provenance/CT_REPRODUCIBILITY.source.md')
    write(bundle/'manifest.json', manifest)
    validator = backend.load_module('_robustness_bundle_validation', bundle/'bundle.py')
    checked = validator.validate_bundle(bundle, require_archive=False)
    changes = {}
    for path in files.rglob('*'):
        if not path.is_file():
            continue
        name = str(path.relative_to(files))
        old = original_git(repo, 'show', publication['expected_head']+':'+name, binary=True, check=False)
        if old.returncode == 0 and old.stdout == path.read_bytes():
            continue
        if not backend.allowed_path(name):
            raise ValueError('Export would mutate an unauthorized path: '+name)
        changes[name] = backend.record(path)
    for name in ('report_summary.json','selection.json','protocol.json','completion.json','staged_report.json'):
        inputs[str(root/name)] = digest(root/name)
    plan = dict(stage_files=str(files), files=changes, input_files=inputs,
        expected_head=publication['expected_head'], expected_branch=publication['expected_branch'],
        bundle_validation=checked, archive_files=len(records), compressed_trace_parts=len(compressed))
    backend.validate_stage_files(plan)
    write(plan_path, plan)
    return plan


def export(root, staged=None):
    root = Path(root).resolve()
    publication = read(root/'publication_config.json')
    audit = staged or read(root/'staged_report.json')
    if audit['summary_sha256'] != digest(root/'report_summary.json'):
        raise ValueError('Report summary changed after rendering')
    repo = Path(publication['repo'])
    plan = build_stage(root, publication, audit)
    # Avoid touching the runtime report if a newer document was published.
    already = (root/'report_update.json').exists()
    runtime = Path(publication['runtime_report'])
    editorial = runtime/'editorial_revision.json'
    if not already:
        if editorial.exists() and str(editorial) not in publication['runtime_assets']:
            raise ValueError('Existing runtime editorial revision was not frozen for publication')
        for name, expected in publication['runtime_assets'].items():
            if digest(name) != expected:
                raise ValueError('Runtime publication changed during this campaign: '+name)
    receipt = backend.publish_stage(root, repo, plan, publication)
    if receipt['status'] != 'completed_pushed':
        raise ValueError('Git publication did not complete')
    if not already:
        backup = root/'report_previous'
        backup.mkdir(exist_ok=True)
        if editorial.exists():
            backend.atomic_copy(editorial, backup/editorial.name)
        for name in backend.REPORT_FILES:
            backend.atomic_copy(runtime/name, backup/name)
            backend.atomic_copy(repo/BUNDLE/'report'/name, runtime/name)
        for name in ('ROBUSTNESS_RESULTS.md','ROBUSTNESS_EVALUATION.md'):
            backend.atomic_copy(repo/BUNDLE/name, runtime.parent/name)
        backend.atomic_copy(repo/BUNDLE/'provenance/CT_REPRODUCIBILITY.source.md', runtime.parent/'CT_REPRODUCIBILITY.md')
        if editorial.exists():
            write(editorial, dict(date=audit['generated_utc'][:10], updated_utc=audit['generated_utc'],
                description='Publish the expanded SFRAMES mechanism and robustness evaluation.',
                previous_pdf_sha256=digest(backup/'sframes_ct_section.pdf'),
                pdf_sha256=audit['published_sha256']['sframes_ct_section.pdf'],
                previous_tex_sha256=digest(backup/'ct_experiments.tex'),
                tex_sha256=audit['published_sha256']['ct_experiments.tex'],
                numerical_results_unchanged=False, pages=audit['pages'], figures=2, tables=1,
                previous_document=str(backup), stage=audit['stage'],
                publication_audit=dict(git_commit=receipt['commit'],
                    summary_sha256=audit['summary_sha256'], protocol_sha256=audit['protocol_sha256'])))
        write(root/'report_update.json', audit | dict(git_commit=receipt['commit']))
    return receipt

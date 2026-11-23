"""Publish a revised paper after an immutable CT campaign has completed.

The running campaign and its protocol are never edited. A separate publication
workspace preserves original evidence and labels its new protocol as an
editorial overlay, not a new numerical execution. Systemd may invoke this
helper on either success or failure; incomplete numerical campaigns are skipped.
"""

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys


RUN_FILES = ('metadata.json', 'result.json', 'trace.json', 'candidate_failure.json')


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


def require(condition, message):
    if not condition:
        raise ValueError(message)


def safe_relative(name):
    value = Path(name)
    require(not value.is_absolute() and '..' not in value.parts, 'Unsafe publication path')
    return value


def immutable_copy(source, destination, *, hardlink=False):
    source, destination = Path(source), Path(destination)
    require(source.is_file() and not source.is_symlink(), 'Expected regular source evidence')
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        require(not destination.is_symlink() and digest(destination) == digest(source),
                'Previously staged evidence changed: ' + str(destination))
        return
    if hardlink:
        try:
            os.link(source, destination)
            return
        except OSError:
            pass
    shutil.copyfile(source, destination)


def immutable_json(path, value):
    if Path(path).exists():
        require(read(path) == value, 'Previously frozen publication record changed: ' + str(path))
    else:
        write(path, value)


def module(name, source):
    spec = importlib.util.spec_from_file_location(name, source)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def validate_config(config):
    original = Path(config['original_root']).resolve()
    output = Path(config['output_root']).resolve()
    require(original != output and original.parent == output.parent,
            'Publication workspace must be a distinct campaign sibling')
    sources = config['source_files']
    require({'compact_publication_handoff.py', 'compact_paper.py'} <= set(sources),
            'Missing frozen publication implementation')
    for name, record in sources.items():
        safe_relative(name)
        require(digest(record['path']) == record['sha256'], 'Publication source changed: ' + name)
    require(digest(__file__) == sources['compact_publication_handoff.py']['sha256'],
            'Active publication helper was not frozen')
    predecessor = Path(config['predecessor_bundle'])
    for name, expected in config['publication']['predecessor_files'].items():
        require(digest(predecessor/safe_relative(name)) == expected,
                'Compact predecessor changed: ' + name)
    require(digest(original/'protocol.json') == config['expected_original_protocol_sha256'],
            'Original numerical protocol changed')
    return original, output


def validate_original(root):
    """Use the original audited validator on its original numerical evidence."""
    root = Path(root)
    protocol = read(root/'protocol.json')
    source = root/'frozen_sources/robustness_campaign.py'
    require(digest(source) == protocol['frozen_sources']['robustness_campaign.py'],
            'Original controller changed')
    controller = module('_ct_original_campaign_handoff', source)
    controller.verify_prepared(root)
    completion = read(root/'completion.json')
    require(completion['status'] == 'completed'
            and completion['protocol_sha256'] == digest(root/'protocol.json')
            and completion['selection_sha256'] == digest(root/'selection.json')
            and completion['evaluation_plan_sha256'] == digest(root/'evaluation_plan.json'),
            'Numerical completion evidence changed')
    for name, expected in completion['stage_files'].items():
        require(digest(root/safe_relative(name)) == expected, 'Completed stage receipt changed')
    cases = {}
    failures = 0
    for path in sorted((root/'plans').glob('*.json')):
        plan = read(path)
        for cfg in plan['runs']:
            rid = cfg['run_id']
            require(Path(rid).name == rid and rid not in ('.', '..'), 'Unsafe run identifier')
            if rid in cases:
                require(cases[rid] == cfg, 'Conflicting original run configuration')
                continue
            require(cfg['phase'] in ('development', 'evaluation'), 'Unknown experiment phase')
            checked = controller.validate_result(cfg, plan['assignments'][rid], protocol, root)
            require(checked['status'] == 'completed' or cfg['phase'] == 'development',
                    'An evaluation failed')
            failures += checked['status'] != 'completed'
            cases[rid] = cfg
    evaluation = sum(cfg['phase'] == 'evaluation' for cfg in cases.values())
    require(evaluation == completion['evaluation_runs'], 'Evaluation plan/count mismatch')
    return dict(status='validated_original_numerical_evidence', original_root=str(root),
        protocol_sha256=digest(root/'protocol.json'), completion_sha256=digest(root/'completion.json'),
        planned_runs=len(cases), evaluation_runs=evaluation,
        development_runs=len(cases)-evaluation, failed_development_runs=failures,
        numerical_results_unchanged=True)


def prepare_overlay(config_path, config, validation):
    original, output = Path(config['original_root']), Path(config['output_root'])
    output.mkdir(parents=True, exist_ok=True)
    # Preserve all frozen numerical files and their actual executed sources.
    excluded = {'protocol.json', 'completion.json', 'publication_config.json',
                'progress.json', 'report_summary.json', 'staged_report.json',
                'report_update.json', 'handoff_status.json'}
    for source in sorted(original.glob('*.json')):
        if source.name not in excluded and not source.name.startswith('git_export'):
            immutable_copy(source, output/source.name)
    for name in ('protocol.json', 'completion.json', 'publication_config.json'):
        immutable_copy(original/name, output/('original_'+name))
    for directory in ('plans', 'stages', 'frozen_sources'):
        for source in sorted((original/directory).rglob('*')):
            if source.is_file() and source.suffix in ('.json', '.py', '.md'):
                immutable_copy(source, output/source.relative_to(original))
    run_hashes = {}
    for source in sorted((original/'runs').glob('*/*.json')):
        if source.name in RUN_FILES:
            relative = source.relative_to(original)
            require(source.resolve().is_relative_to(original.resolve()), 'Run evidence leaves original campaign')
            immutable_copy(source, output/relative, hardlink=True)
            run_hashes[str(relative)] = digest(source)
    immutable_copy(config_path, output/'publication_revision.json')
    for name, source in config['source_files'].items():
        immutable_copy(source['path'], output/'frozen_sources/publication'/safe_relative(name))
    predecessor = Path(config['predecessor_bundle'])
    for name in config['publication']['predecessor_files']:
        immutable_copy(predecessor/name, output/'predecessor_bundle'/safe_relative(name))
    immutable_json(output/'publication_config.json', config['publication'])
    immutable_json(output/'original_validation.json', validation)
    immutable_json(output/'numerical_evidence_manifest.json', run_hashes)
    # Keep the actual original run metadata intact, including its protocol hash.
    # The new protocol describes only the publication operation and explicitly
    # identifies the original numerical protocol that those run records certify.
    original_protocol = read(original/'protocol.json')
    source_hashes = dict(original_protocol['frozen_sources'])
    source_hashes.update({'publication/'+name: record['sha256']
                         for name, record in config['source_files'].items()})
    frozen_files = dict(original_protocol['frozen_files'])
    for name in ('publication_config.json', 'publication_revision.json', 'original_validation.json',
                 'numerical_evidence_manifest.json', 'original_protocol.json',
                 'original_completion.json', 'original_publication_config.json'):
        frozen_files[name] = digest(output/name)
    overlay_protocol = dict(original_protocol, kind='publication_overlay', publication_overlay=True,
        original_campaign_root=str(original.resolve()),
        original_protocol_sha256=digest(original/'protocol.json'),
        original_completion_sha256=digest(original/'completion.json'),
        numerical_results_unchanged=True, new_numerical_execution=False,
        frozen_sources=source_hashes, frozen_files=frozen_files,
        interpretation='Editorial publication only. Unmodified run metadata retains the original numerical protocol hash.')
    immutable_json(output/'protocol.json', overlay_protocol)
    original_completion = read(original/'completion.json')
    immutable_json(output/'completion.json', dict(original_completion,
        kind='publication_overlay', publication_overlay=True, protocol_sha256=digest(output/'protocol.json'),
        original_completion_sha256=digest(original/'completion.json'),
        original_protocol_sha256=digest(original/'protocol.json'),
        numerical_results_unchanged=True, new_numerical_execution=False))
    return output


def execute(config_path):
    config_path = Path(config_path).resolve()
    config = read(config_path)
    original, output = validate_config(config)
    output.mkdir(parents=True, exist_ok=True)
    with (output/'handoff.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (output/'report_update.json').exists():
            require(read(output/'report_update.json').get('status') == 'completed_published',
                    'Unexpected prior publication status')
            return read(output/'report_update.json')
        if not (original/'completion.json').exists() or read(original/'completion.json').get('status') != 'completed':
            result = dict(status='waiting_for_numerical_completion', original_root=str(original))
            write(output/'handoff_status.json', result)
            return result
        try:
            with (original/'campaign.lock').open('a') as numerical_lock:
                fcntl.flock(numerical_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                validation = validate_original(original)
                prepare_overlay(config_path, config, validation)
                source = Path(config['source_files']['compact_paper.py']['path'])
                sys.path.insert(0, str(source.parent))
                renderer = module('_ct_compact_publication', source)
                result = renderer.publish_expanded(output)
                require((output/'report_update.json').is_file()
                        and read(output/'report_update.json').get('status') == 'completed_published',
                        'Compact renderer did not complete publication')
                write(original/'publication_successor.json', dict(status='completed_published',
                    publication_root=str(output), report_update=str(output/'report_update.json'),
                    report_update_sha256=digest(output/'report_update.json'),
                    original_protocol_sha256=digest(original/'protocol.json'),
                    publication_overlay=True, numerical_results_unchanged=True))
            write(output/'handoff_status.json', dict(status='completed_published',
                utc=datetime.now(timezone.utc).isoformat(), original_root=str(original),
                publication_root=str(output)))
            return result
        except Exception as error:
            write(output/'handoff_status.json', dict(status='publication_pending', error=repr(error),
                utc=datetime.now(timezone.utc).isoformat(), original_root=str(original)))
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(execute(args.config), indent=2))

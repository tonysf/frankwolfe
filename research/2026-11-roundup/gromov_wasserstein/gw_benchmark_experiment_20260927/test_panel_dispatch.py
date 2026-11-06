"""Fail-closed scheduling tests; no scheduler or numerical experiments are invoked."""
from copy import deepcopy
import json
import pytest
from run_panel_array import sha256, validate_plan


def make_plan(tmp_path):
    tasks = []
    for panel, count in [('product', 3), ('randomized', 2)]:
        path = tmp_path / f'{panel}.json'
        path.write_text(json.dumps({'runs': [1, 2, 3, 4, 5]}))
        tasks.extend(dict(manifest=str(path), manifest_sha256=sha256(path),
                          out_root=str(tmp_path / f'outputs_{panel}'),
                          task_index=i, task_count=count) for i in range(count))
    return dict(schema_version=1, tasks=tasks)


def test_dispatch_maps_global_indices_to_distinct_local_shards(tmp_path):
    plan = make_plan(tmp_path)
    assert [validate_plan(plan, i)['task_index'] for i in range(5)] == [0, 1, 2, 0, 1]


@pytest.mark.parametrize('mutation', ['duplicate', 'missing', 'mixed_count', 'shared_root', 'relative_path',
                                    'aliased_root', 'nested_root', 'duplicated_manifest', 'aliased_manifest'])
def test_invalid_scheduling_maps_fail_before_dispatch(tmp_path, mutation):
    plan = make_plan(tmp_path)
    if mutation == 'duplicate':
        plan['tasks'].append(deepcopy(plan['tasks'][0]))
    elif mutation == 'missing':
        plan['tasks'].pop()
    elif mutation == 'mixed_count':
        plan['tasks'][0]['task_count'] = 4
    elif mutation == 'shared_root':
        for task in plan['tasks']:
            task['out_root'] = str(tmp_path / 'shared')
    elif mutation == 'relative_path':
        plan['tasks'][0]['manifest'] = 'relative.json'
    else:
        from pathlib import Path
        first_manifest = plan['tasks'][0]['manifest']
        if mutation == 'aliased_manifest':
            alias = tmp_path / 'alias.json'
            alias.symlink_to(first_manifest)
        for task in plan['tasks'][3:]:
            if mutation == 'aliased_root':
                task['out_root'] = str(tmp_path / 'unused/../outputs_product')
            elif mutation == 'nested_root':
                task['out_root'] = str(tmp_path / 'outputs_product/nested')
            elif mutation == 'duplicated_manifest':
                task['manifest'] = first_manifest
            else:
                task['manifest'] = str(alias)
    with pytest.raises(ValueError):
        validate_plan(plan, 0)


def test_changed_manifest_is_rejected(tmp_path):
    plan = make_plan(tmp_path)
    from pathlib import Path
    Path(plan['tasks'][0]['manifest']).write_text('{}')
    with pytest.raises(ValueError, match='hash mismatch'):
        validate_plan(plan, 0)


@pytest.mark.parametrize('index', [-1, 5])
def test_out_of_bounds_task_fails(tmp_path, index):
    with pytest.raises(ValueError):
        validate_plan(make_plan(tmp_path), index)

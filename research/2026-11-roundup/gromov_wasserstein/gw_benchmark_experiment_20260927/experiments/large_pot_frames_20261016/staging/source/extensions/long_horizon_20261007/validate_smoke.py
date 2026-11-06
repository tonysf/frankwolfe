"""Allocated GPU integration smoke for the new controller and unchanged runners."""
import argparse
import copy
from pathlib import Path

import numpy as np

import campaign


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    prepared = args.out / 'prepared'
    data = prepared / 'smoke16'
    data.mkdir(parents=True)
    rng = np.random.default_rng(20261007)
    arrays = {}
    for name, dimensions in (('A', 3), ('B', 4)):
        coordinates = rng.normal(size=(16, dimensions))
        distances = np.linalg.norm(coordinates[:, None, :] - coordinates[None, :, :], axis=2)
        arrays[name] = (distances / distances.max()).astype('float32')
    arrays.update(a=np.full(16, 1/16), b=np.full(16, 1/16))
    for name, value in arrays.items():
        np.save(data / (name + '.npy'), value, allow_pickle=False)
    hashes = {name + '.npy': campaign.sha256(data / (name + '.npy')) for name in arrays}
    campaign.write_json(data / 'manifest.json', dict(id='smoke16', array_files_sha256=hashes))
    config_path = args.source / 'extensions/long_horizon_20261007/campaign.json'
    config = copy.deepcopy(campaign.load_config(config_path))
    config.pop('_config_path')
    dataset = next(iter(config['instances']))
    candidates = config['selected_candidates'][dataset]
    instance = dict(n=16, world_size=1, array_files_sha256=hashes,
                    instance_manifest_sha256=campaign.sha256(data / 'manifest.json'))
    frozen = campaign.read_json(config_path.parent / config['selections_provenance']['file'])
    evidence = copy.deepcopy(frozen['selections'][dataset])
    for record in evidence.values():
        record['array_files_sha256'] = hashes
        record['instance_manifest_sha256'] = instance['instance_manifest_sha256']
    frozen.update(selections={'smoke16': evidence}, scope='Synthetic controller integration fixture, not a scientific run.')
    campaign.write_json(args.out / 'frozen_selections.json', frozen)
    config.update(instances={'smoke16': instance}, selected_candidates={'smoke16': candidates},
                  final_iterations=4, final_checkpoint_iterations=[1, 2, 4],
                  selections_provenance=dict(file='frozen_selections.json',
                      sha256=campaign.sha256(args.out / 'frozen_selections.json')),
                  run_timeout_seconds_by_method={method: 180 for method in campaign.METHODS})
    campaign.write_json(args.out / 'smoke_config.json', config)
    records = []
    for method in campaign.METHODS:
        for task in (0, 3):
            invocation = argparse.Namespace(config=str(args.out / 'smoke_config.json'),
                source=str(args.source), prepared=str(prepared), out=str(args.out / 'runs'),
                instance='smoke16', method=method, task_index=task)
            status = campaign.run_task(invocation)
            records.append(dict(method=method, task_index=task, **status))
    campaign.require(len(records) == 6 and all(record['status'] == 'completed' for record in records),
                     'All three methods and both starts must pass the controller GPU smoke.')
    campaign.write_json(args.out / 'gate.json', dict(status='passed', records=records,
        scope='Six four-update synthetic GPU runs through the new controller; unchanged production numerical sources have prior full-size A100 validation.'))
    print('Passed six GPU integration smokes.', flush=True)


if __name__ == '__main__':
    main()

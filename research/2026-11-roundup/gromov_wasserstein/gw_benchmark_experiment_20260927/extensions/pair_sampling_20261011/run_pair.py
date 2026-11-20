"""One entry point for the three unchanged updates with a selectable oracle.

Default sketch behavior remains available.  --oracle pair-sampling selects
the same conditional iid coupling-pair estimator for every method.  Method
schedules, momentum indexing, proximal certificates and output RNG retain
their existing implementations.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import tempfile
import traceback

import torch
import torch.distributed as dist

ROOT = Path(__file__).resolve().parent
for directory in (ROOT, ROOT.parent/'published_stochastic_20261002',
                  ROOT.parent/'msbpg_20261005', ROOT.parent/'shareseq_25000_20260930'):
    sys.path.insert(0, str(directory))
import run_published
import run_msbpg
from run_sharded import setup, sha256, write_json
from oracle import METHOD_NAMES


def parser():
    p = run_published.parser()
    p.description = __doc__
    for action in p._actions:
        if action.dest == 'method':
            action.choices = tuple(METHOD_NAMES)
    p.add_argument('--coupling-fraction', type=float, default=1.)
    p.add_argument('--momentum-prefactor', type=float, default=1.)
    return p


def implementation(args):
    return run_msbpg if args.method == 'published-msbpg' else run_published


def validate_args(args):
    if args.method not in METHOD_NAMES:
        raise ValueError('Unknown method.')
    implementation(args).validate_args(args)


def source_hashes():
    # Union covers the common wrapper and every implementation it can select.
    hashes = dict(run_published.source_hashes(), **run_msbpg.source_hashes())
    for name in ('run_pair.py', 'oracle.py'):
        hashes[str((ROOT/name).relative_to(ROOT.parent))] = sha256(ROOT/name)
    return hashes


def run(args, device):
    validate_args(args)
    metadata = implementation(args).run(args, device)
    metadata['source_file_sha256'] = source_hashes()
    metadata['unified_entrypoint'] = 'pair_sampling_20261011/run_pair.py'
    if dist.get_rank() == 0:
        write_json(Path(args.out)/'metadata.json', metadata)
    return metadata


def main():
    args = parser().parse_args()
    rendezvous = None
    try:
        validate_args(args)
        if args.device == 'cpu' and 'RANK' not in os.environ:
            torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', '1')))
            torch.use_deterministic_algorithms(True)
            rendezvous = tempfile.TemporaryDirectory(prefix='pair_gw_rendezvous_')
            dist.init_process_group('gloo', init_method='file://'+str(Path(rendezvous.name)/'store'),
                                    rank=0, world_size=1)
            device = torch.device('cpu')
        else:
            device = setup(args.device)
        run(args, device)
    except Exception as exc:
        directory = Path(args.out)
        directory.mkdir(parents=True, exist_ok=True)
        rank = dist.get_rank() if dist.is_initialized() else int(os.environ.get('RANK', '0'))
        write_json(directory/f'failure_rank{rank}.json', dict(status='failed', rank=rank,
                   error=repr(exc), traceback=traceback.format_exc(), config=vars(args)))
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
        if rendezvous is not None:
            rendezvous.cleanup()


if __name__ == '__main__':
    main()

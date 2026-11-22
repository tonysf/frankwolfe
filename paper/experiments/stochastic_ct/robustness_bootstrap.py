"""Prepare frozen CT inputs, then execute the approved persistent campaign."""
import os
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS',
             'VECLIB_MAXIMUM_THREADS','BLIS_NUM_THREADS'):
    os.environ[name] = '1'
import argparse
import fcntl
from pathlib import Path
import traceback

try:
    from . import robustness_campaign as campaign
    from .robustness_inputs import prepare_inputs
except ImportError:
    import robustness_campaign as campaign
    from robustness_inputs import prepare_inputs


def execute(root, publish=True):
    root = Path(root).resolve()
    with (root/'bootstrap.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        gate = campaign.read(root/'implementation_validation.json')
        campaign.require(gate['status'] == 'passed', 'Implementation has not passed validation')
        for name, record in gate['source_files'].items():
            campaign.require(campaign.digest(root/'frozen_sources'/name) == record['sha256'],
                             'Frozen source changed: '+name)
        if ('robustness_reuse.py' in gate['source_files']
                or (root/'historical_reuse_catalog.json').exists()
                or 'historical_reuse_catalog_sha256' in gate):
            campaign.require('robustness_reuse.py' in gate['source_files'],
                             'Historical reuse helper was not validated')
            campaign.require(bool(gate.get('historical_reuse_catalog_sha256')),
                             'Historical reuse catalog was not validated')
            campaign.require(campaign.digest(root/'historical_reuse_catalog.json')
                             == gate['historical_reuse_catalog_sha256'],
                             'Historical reuse catalog changed before preparation')
        available = os.sched_getaffinity(0)
        campaign.require(set(campaign.CPUS) <= available, 'Expected Elite cores unavailable')
        try:
            if not (root/'protocol.json').exists():
                campaign.update(root, 'running', 'data_and_calibration_preparation')
                os.sched_setaffinity(0, {campaign.CPUS[0]})
                cohort, inputs = prepare_inputs(root, allow_download=True)
                campaign.frozen(root/'cohort.json', cohort)
                campaign.frozen(root/'inputs.json', inputs)
                os.sched_setaffinity(0, set(campaign.CPUS))
                campaign.prepare(root)
            else:
                campaign.verify_prepared(root)
            with (root/'campaign.lock').open('a') as campaign_lock:
                fcntl.flock(campaign_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                campaign.execute(root, publish=publish)
        except Exception as error:
            state = campaign.read(root/'progress.json') if (root/'progress.json').exists() else {}
            if state.get('phase') != 'publication_pending':
                campaign.update(root, 'failed', state.get('phase','preparation'),
                                error=repr(error), traceback=traceback.format_exc())
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=campaign.ROOT)
    parser.add_argument('--no-publish', action='store_true')
    args = parser.parse_args()
    execute(args.root, publish=not args.no_publish)

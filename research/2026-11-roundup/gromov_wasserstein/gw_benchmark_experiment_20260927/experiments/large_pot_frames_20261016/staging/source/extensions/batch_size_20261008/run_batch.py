"""Batch-size ablation adapter; all numerical code stays in the frozen runner.

Only the campaign-specific q128 argument restriction and source provenance are
extended. The estimator, recurrence, schedules, initialization, rounding,
output sampling, counters, and FP64 validation are inherited without changes.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / 'published_stochastic_20261002'))
import run_published as original

ALLOWED_BATCHES = (128, 1024, 2048)
_original_validate_args = original.validate_args
_original_source_hashes = original.source_hashes


def validate_args(args):
    if args.method != 'smoothing-fw':
        raise ValueError('This diagnostic adapter supports FraMES only.')
    if (not isinstance(args.batch, int) or isinstance(args.batch, bool)
            or args.batch not in ALLOWED_BATCHES):
        raise ValueError('Batch must be one of 128, 1024, 2048 independent probes.')
    # Validate every unrelated legacy option against its unchanged restriction.
    # This copy is never sent to run(); actual q remains in args and metadata.
    compatible = argparse.Namespace(**vars(args))
    compatible.batch = 128
    _original_validate_args(compatible)


def source_hashes():
    return dict(_original_source_hashes(),
                **{'batch_size_20261008/run_batch.py': original.sha256(Path(__file__))})


original.validate_args = validate_args
original.source_hashes = source_hashes


if __name__ == '__main__':
    original.main()

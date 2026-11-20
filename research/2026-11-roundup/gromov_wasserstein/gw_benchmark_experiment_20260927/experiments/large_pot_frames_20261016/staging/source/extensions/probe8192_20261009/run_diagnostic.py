"""Higher-probe diagnostic using the existing FraMES update without changes."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / 'published_stochastic_20261002'))
import run_published as original
from direction_diagnostics import direction_diagnostics

_original_validate_args = original.validate_args
_original_source_hashes = original.source_hashes
_original_parser = original.parser
_run_without_hook = original.run


def parser():
    result = _original_parser()
    result.add_argument('--direction-diagnostics', choices=('enabled', 'disabled'), default='enabled',
                        help='Disable only for gate/CPU trajectory-parity controls.')
    return result


def validate_args(args):
    if args.method != 'smoothing-fw':
        raise ValueError('Only FraMES is supported in this probe diagnostic.')
    if (not isinstance(args.batch, int) or isinstance(args.batch, bool)
            or args.batch not in (2048, 8192)):
        raise ValueError('Use q8192, or q2048 only for comparison with the prior runner.')
    if args.mu_scale != 10. or args.gamma0 != 1. or args.rho0 != .3 or args.initialization != 'product':
        raise ValueError('Keep mu_scale=10, gamma0=1, rho0=0.3 and product initialization fixed.')
    if args.direction_diagnostics not in ('enabled', 'disabled'):
        raise ValueError('Invalid direction diagnostics mode.')
    compatible = argparse.Namespace(**vars(args))
    compatible.batch = 128
    _original_validate_args(compatible)


def source_hashes():
    result = _original_source_hashes()
    for name in ('run_diagnostic.py', 'direction_diagnostics.py'):
        result['probe8192_20261009/'+name] = original.sha256(ROOT/name)
    return result


def run(args, device):
    hook = direction_diagnostics if args.direction_diagnostics == 'enabled' else None
    return _run_without_hook(args, device, diagnostic_hook=hook)


original.parser = parser
original.validate_args = validate_args
original.source_hashes = source_hashes
original.run = run


if __name__ == '__main__':
    original.main()

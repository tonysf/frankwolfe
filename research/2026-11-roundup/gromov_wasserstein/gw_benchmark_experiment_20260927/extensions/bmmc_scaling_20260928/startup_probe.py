"""Run the immutable environment check with bounded diagnostic stack dumps."""
import faulthandler
from pathlib import Path
import runpy
import sys
import time

if __name__ == '__main__':
    target = Path(sys.argv[1]).resolve()
    sys.argv = [str(target), *sys.argv[2:]]
    sys.path.insert(0, str(target.parent))
    print('Starting immutable environment validation at', time.time(), flush=True)
    faulthandler.enable()
    faulthandler.dump_traceback_later(120, repeat=True)
    try:
        runpy.run_path(str(target), run_name='__main__')
    finally:
        faulthandler.cancel_dump_traceback_later()
        print('Environment validation returned at', time.time(), flush=True)

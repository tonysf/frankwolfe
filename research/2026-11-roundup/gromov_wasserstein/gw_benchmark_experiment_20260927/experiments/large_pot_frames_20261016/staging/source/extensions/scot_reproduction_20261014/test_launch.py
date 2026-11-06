"""Exercise the complete scheduler dependency graph without submitting a job."""
import hashlib
import json
import os
from pathlib import Path
import subprocess


def test_submission_graph_and_single_submission_lock(tmp_path):
    source = tmp_path/'source'
    root = tmp_path/'campaign'
    stub = tmp_path/'bin'
    for directory in (source, root, stub, root/'inputs', root/'wheelhouse'):
        directory.mkdir(parents=True, exist_ok=True)
    for directory in (source, root/'inputs', root/'wheelhouse'):
        content = b'frozen scientific input\n'
        (directory/'fixture').write_bytes(content)
        (directory/'SHA256SUMS').write_text(hashlib.sha256(content).hexdigest()+'  fixture\n')
    for name in ('cpu.json', 'gpu.json'):
        (root/name).write_text('{}\n')
    mock = stub/'sbatch'
    mock.write_text("""#!/usr/bin/env python3
import json, os, pathlib, sys
p = pathlib.Path(os.environ['SCOT_MOCK_SUBMISSIONS'])
rows = p.read_text().splitlines() if p.exists() else []
with p.open('a') as stream:
    stream.write(json.dumps(sys.argv[1:])+'\\n')
print(9001+len(rows))
""")
    mock.chmod(0o755)
    log = root/'mock_submissions.jsonl'
    environment = dict(os.environ, PATH=str(stub)+os.pathsep+os.environ['PATH'],
                       SCOT_MOCK_SUBMISSIONS=str(log))
    command = ['bash', str(Path(__file__).with_name('submit.sh')), str(source), str(root),
               str(root/'cpu.json'), str(root/'gpu.json')]
    result = subprocess.run(command, env=environment, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    assert len(calls) == 6
    assert '--dependency=afterok:9001' in calls[1]
    assert '--dependency=afterok:9002' in calls[2]
    assert '--dependency=afterok:9003' in calls[3]
    assert '--dependency=afterok:9004' in calls[4]
    assert '--dependency=afterany:9001:9002:9003:9004:9005' in calls[5]
    assert '--array=0-26%4' in calls[1] and '--array=0-26%4' in calls[2]
    assert '--array=0-8%4' in calls[4]
    assert all('--kill-on-invalid-dep=yes' in calls[i] for i in range(1, 5))
    assert (root/'submission.tsv').read_text().splitlines() == [
        'stage\tjob_id', 'reference\t9001', 'preflight\t9002', 'tuning\t9003',
        'selection\t9004', 'confirmation\t9005', 'report\t9006']
    repeated = subprocess.run(command, env=environment, capture_output=True, text=True)
    assert repeated.returncode != 0
    assert len(log.read_text().splitlines()) == 6

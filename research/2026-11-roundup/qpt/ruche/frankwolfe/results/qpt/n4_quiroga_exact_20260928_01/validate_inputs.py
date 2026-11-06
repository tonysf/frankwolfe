import argparse
import hashlib
import json
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--source", type=Path, required=True)
p.add_argument("--record", type=Path, required=True)
p.add_argument("--data", type=Path, required=True)
a = p.parse_args()
record = json.loads(a.record.read_text())
for relative, expected in record["source_sha256"].items():
    actual = hashlib.sha256((a.source / relative).read_bytes()).hexdigest()
    if actual != expected:
        raise RuntimeError(f"source mismatch: {relative}")
actual = hashlib.sha256(a.data.read_bytes()).hexdigest()
if actual != "3e56164cfa8a32c97074ae08846c65543e92c4c419373ffdce771628d213f8e8":
    raise RuntimeError("n=4 data mismatch")
print(f"validated {len(record['source_sha256'])} source files and n=4 data {actual}")

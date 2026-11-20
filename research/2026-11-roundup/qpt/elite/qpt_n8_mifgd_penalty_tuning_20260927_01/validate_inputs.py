import argparse, hashlib, json
from pathlib import Path

p=argparse.ArgumentParser(); p.add_argument("--source",type=Path,required=True); p.add_argument("--record",type=Path,required=True); p.add_argument("--data",type=Path,required=True); a=p.parse_args()
record=json.loads(a.record.read_text())
for relative, expected in record["source_sha256"].items():
    actual=hashlib.sha256((a.source/relative).read_bytes()).hexdigest()
    if actual != expected: raise RuntimeError(f"source mismatch: {relative}")
actual=hashlib.sha256(a.data.read_bytes()).hexdigest()
if actual != "c05b02bdbc1b8955a89ae0cff2da77a986c76f96d2971860f3d2f7d0b925e439": raise RuntimeError("data mismatch")
print(f"validated {len(record['source_sha256'])} source files and data {actual}")

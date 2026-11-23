import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from checkpoint import CheckpointStore


class CheckpointTests(unittest.TestCase):
    def test_roundtrip_and_old_generation_cleanup(self):
        with tempfile.TemporaryDirectory() as directory:
            store = CheckpointStore(directory)
            first = store.save({"plan": torch.arange(6)}, identity={"seed": 7}, iteration=1)
            second = store.save({"plan": torch.arange(6)+1}, identity={"seed": 7}, iteration=2)
            state, record = store.load(identity={"seed": 7})
            self.assertTrue(torch.equal(state["plan"], torch.arange(6)+1))
            self.assertEqual(record, second)
            self.assertFalse((Path(directory)/first["generation"]).exists())
            self.assertEqual(len(list(Path(directory).glob("generation-*"))), 1)

    def test_interruption_before_commit_retains_previous_state(self):
        with tempfile.TemporaryDirectory() as directory:
            store = CheckpointStore(directory)
            first = store.save({"plan": torch.arange(6)}, identity={"seed": 7}, iteration=1)
            import checkpoint
            original = checkpoint.atomic_json
            def fail_marker(path, value):
                if Path(path).name == "COMMITTED.json":
                    raise OSError("simulated interruption before commit")
                return original(path, value)
            with patch("checkpoint.atomic_json", side_effect=fail_marker):
                with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                    store.save({"plan": torch.arange(6)+1}, identity={"seed": 7}, iteration=2)
            state, record = store.load(identity={"seed": 7})
            self.assertEqual(record, first)
            self.assertTrue(torch.equal(state["plan"], torch.arange(6)))
            self.assertEqual(len(list(Path(directory).glob("generation-*"))),1)

    def test_corrupt_shard_and_changed_identity_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            store = CheckpointStore(directory)
            record = store.save({"x": torch.ones(2)}, identity={"seed": 7}, iteration=1)
            with self.assertRaisesRegex(ValueError, "changed"):
                store.load(identity={"seed": 8})
            path = Path(directory)/record["generation"]/record["shards"][0]["file"]
            with path.open("ab") as stream:
                stream.write(b"corruption")
            with self.assertRaisesRegex(RuntimeError, "hash mismatch"):
                store.load(identity={"seed": 7})

    def test_completed_cleanup_is_scoped(self):
        with tempfile.TemporaryDirectory() as directory:
            store = CheckpointStore(directory)
            Path(directory,"preserve.txt").write_text("receipt")
            store.save({"x": torch.ones(2)}, identity={}, iteration=1)
            store.remove_completed_state()
            self.assertEqual([p.name for p in Path(directory).iterdir()], ["preserve.txt"])


if __name__ == "__main__":
    unittest.main()

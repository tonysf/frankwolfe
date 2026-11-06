"""Transactional, rank-sharded checkpoints with one committed generation.

The old complete generation remains readable until all new rank files and
their hashes are durable and COMMITTED.json is atomically replaced. Only
then is the superseded generation removed. This deliberately needs enough
storage for two generations: overwriting rank files in place is not safe.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import uuid

import torch
import torch.distributed as dist


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fsync_directory(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    fsync_directory(path.parent)


def tensor_bytes(value):
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size()
    if isinstance(value, dict):
        return sum(tensor_bytes(x) for x in value.values())
    if isinstance(value, (tuple, list)):
        return sum(tensor_bytes(x) for x in value)
    return 0


class CheckpointStore:
    def __init__(self, directory, *, rank=0, world_size=1, group=None):
        self.directory = Path(directory)
        self.rank, self.world_size, self.group = rank, world_size, group
        self.marker = self.directory / "COMMITTED.json"

    def _barrier(self):
        if self.world_size > 1:
            dist.barrier(group=self.group)

    def _broadcast(self, value, owner=0):
        if self.world_size == 1:
            return value
        values = [value]
        source = dist.get_global_rank(self.group, owner) if self.group is not None else owner
        dist.broadcast_object_list(values, src=source, group=self.group)
        return values[0]

    def _owner_action(self, owner, action):
        result, error = None, None
        if self.rank == owner:
            try:
                result = action()
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
        result, error = self._broadcast((result, error), owner)
        if error:
            raise RuntimeError(f"Checkpoint rank {owner}: {error}")
        return result

    def save(self, state, *, identity, iteration):
        self._owner_action(0, lambda: self.directory.mkdir(parents=True, exist_ok=True))
        name = self._broadcast(f"generation-{iteration:09d}-{uuid.uuid4().hex}" if self.rank == 0 else None)
        generation = self.directory / name
        self._owner_action(0, lambda: generation.mkdir())
        records = []
        # One rank writes at a time, limiting staging RAM and filesystem bursts.
        for owner in range(self.world_size):
            def write():
                required = tensor_bytes(state) + 64 * 1024 * 1024
                if shutil.disk_usage(self.directory).free < required:
                    raise OSError("Insufficient filesystem space for a new checkpoint shard")
                path = generation / f"rank-{owner:04d}.pt"
                temporary = path.with_suffix(".tmp")
                with temporary.open("wb") as stream:
                    torch.save(state, stream)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary, path)
                return {"file": path.name, "sha256": sha256(path), "bytes": path.stat().st_size}
            records.append(self._owner_action(owner, write))
        record = dict(format_version=1, generation=name, iteration=int(iteration),
                      world_size=self.world_size, identity=identity, shards=records)
        def commit():
            fsync_directory(generation)
            atomic_json(generation / "manifest.json", record)
            atomic_json(self.marker, record)
        self._owner_action(0, commit)
        # The new marker is durable before any prior-generation deletion.
        self._owner_action(0, lambda: self._remove_other_generations(name))
        self._barrier()
        return record

    def _remove_other_generations(self, keep):
        for path in self.directory.glob("generation-*"):
            if path.name != keep and path.is_dir() and not path.is_symlink():
                shutil.rmtree(path)
        fsync_directory(self.directory)

    def load(self, *, identity, map_location="cpu"):
        record = self._owner_action(0, lambda: json.loads(self.marker.read_text()) if self.marker.exists() else None)
        if record is None:
            return None, None
        if record["identity"] != identity or record["world_size"] != self.world_size:
            raise ValueError("Checkpoint source, data, configuration or process topology changed")
        name = record["generation"]
        if Path(name).name != name or not name.startswith("generation-"):
            raise ValueError("Invalid checkpoint generation")
        state = None
        for owner in range(self.world_size):
            def read():
                nonlocal state
                shard = record["shards"][owner]
                if shard["file"] != f"rank-{owner:04d}.pt":
                    raise ValueError("Invalid checkpoint shard")
                path = self.directory / name / shard["file"]
                if sha256(path) != shard["sha256"]:
                    raise ValueError("Checkpoint shard hash mismatch")
                state = torch.load(path, map_location=map_location, weights_only=False)
                return True
            self._owner_action(owner, read)
        # A killed writer may have left an incomplete newer generation.
        # Verify every committed shard before deleting anything; otherwise a
        # retry could accumulate three full generations before its next save.
        self._owner_action(0, lambda: self._remove_other_generations(name))
        self._barrier()
        return state, record

    def remove_completed_state(self):
        """Only call after a durable completed-run receipt has been written."""
        def remove():
            if self.marker.exists():
                self.marker.unlink()
            self._remove_other_generations(None)
        self._owner_action(0, remove)
        self._barrier()

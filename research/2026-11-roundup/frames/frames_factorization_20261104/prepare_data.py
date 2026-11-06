#!/usr/bin/env python3
"""Prepare the exact Deng Cai ORL64 and YaleB32 variants for factorization.

Output A has shape (pixels, images), dtype float64, and values in [0, 1].
The only intensity transformation is division by the raw global maximum, one
of the two transformations documented on the official FaceData.html page.
Original MATLAB files and their SHA256 digests are retained. No images are
resized, reordered, centered, or normalized individually here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import urllib.request

import numpy as np
from scipy.io import loadmat


ROOT = Path(__file__).resolve().parent
INDEX_URL = "http://www.cad.zju.edu.cn/home/dengcai/Data/FaceData.html"
SPECS = {
    "orl64": {
        "filename": "ORL_64x64.mat",
        "url": "http://www.cad.zju.edu.cn/home/dengcai/Data/ORL/ORL_64x64.mat",
        "raw_shape": (400, 4096),
        "image_shape": (64, 64),
        "classes": 40,
    },
    "yaleb32": {
        "filename": "YaleB_32x32.mat",
        "url": "http://www.cad.zju.edu.cn/home/dengcai/Data/YaleB/YaleB_32x32.mat",
        "raw_shape": (2414, 1024),
        "image_shape": (32, 32),
        "classes": 38,
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, target: Path) -> dict:
    """Download a public original with atomic replacement after a full read."""
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".partial")
    with urllib.request.urlopen(url, timeout=45) as response:
        content = response.read()
        info = {
            "requested_url": url,
            "resolved_url": response.geturl(),
            "http_status": response.status,
            "content_type": response.headers.get("Content-Type"),
            "last_modified": response.headers.get("Last-Modified"),
            "retrieved_utc": datetime.now(timezone.utc).isoformat(),
        }
    partial.write_bytes(content)
    partial.replace(target)
    return info


def prepare(name: str, do_download: bool, previous: dict) -> dict:
    spec = SPECS[name]
    raw_path = ROOT / "data" / "raw" / spec["filename"]
    retrieval = previous.get("retrieval", {})
    if not raw_path.exists():
        if not do_download:
            raise FileNotFoundError(f"Missing {raw_path}; pass --download.")
        retrieval = download(spec["url"], raw_path)
    raw_hash = sha256(raw_path)
    if previous.get("raw_sha256") and previous["raw_sha256"] != raw_hash:
        raise ValueError(f"Previously recorded raw SHA256 differs for {name}.")
    raw = loadmat(raw_path)
    if "fea" not in raw or "gnd" not in raw:
        raise ValueError(f"{name}: expected MATLAB variables fea and gnd.")
    fea = np.asarray(raw["fea"])
    if fea.shape != spec["raw_shape"]:
        raise ValueError(f"{name}: expected {spec['raw_shape']}; got {fea.shape}.")
    if not np.isfinite(fea).all() or np.min(fea) < 0:
        raise ValueError(f"{name}: nonfinite or negative raw pixel values.")
    scale = float(np.max(fea))
    if scale <= 0:
        raise ValueError(f"{name}: empty intensity range.")
    labels_raw = np.asarray(raw["gnd"]).reshape(-1)
    if len(labels_raw) != fea.shape[0] or not np.equal(labels_raw, np.floor(labels_raw)).all():
        raise ValueError(f"{name}: malformed labels.")
    labels = labels_raw.astype(np.int64)
    if len(np.unique(labels)) != spec["classes"]:
        raise ValueError(f"{name}: wrong number of subjects.")
    A = np.ascontiguousarray(fea.T, dtype=np.float64) / scale
    if A.dtype != np.float64 or A.min() < 0 or A.max() > 1:
        raise ValueError(f"{name}: processed precision or range is invalid.")
    processed = ROOT / "data" / "processed" / f"{name}.npz"
    processed.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "dataset": name,
        "official_index_url": INDEX_URL,
        "source_url": spec["url"],
        "raw_sha256": raw_hash,
        "matrix_layout": "pixels_by_images",
        "image_shape": list(spec["image_shape"]),
        "pixel_flattening": "unchanged MATLAB fea ordering; reshape images with order='F'",
        "scaling": "A = float64(fea.T) / max(fea)",
        "scale_divisor": scale,
        "centering": False,
        "per_image_normalization": False,
    }
    np.savez_compressed(processed, A=A, labels=labels, metadata=json.dumps(metadata, sort_keys=True))
    record = {
        **metadata,
        "raw_path": str(raw_path.relative_to(ROOT)),
        "processed_path": str(processed.relative_to(ROOT)),
        "raw_size_bytes": raw_path.stat().st_size,
        "raw_shape": list(fea.shape),
        "raw_dtype": str(fea.dtype),
        "raw_min": float(fea.min()),
        "raw_max": scale,
        "processed_shape": list(A.shape),
        "processed_dtype": str(A.dtype),
        "processed_min": float(A.min()),
        "processed_max": float(A.max()),
        "processed_sha256": sha256(processed),
        "matrix_sha256_float64_c_order": hashlib.sha256(A.tobytes(order="C")).hexdigest(),
        "subjects": len(np.unique(labels)),
        "subject_counts": {str(int(k)): int(v) for k, v in zip(*np.unique(labels, return_counts=True))},
        "retrieval": retrieval,
        "checksum_provenance": "Computed locally from downloaded original; publisher supplies no checksum on index.",
        "validation": "Exact variant shape, finite nonnegative pixels, integer labels, subject count, FP64 [0,1] range.",
    }
    print(json.dumps({k: record[k] for k in ("dataset", "processed_path", "processed_shape", "raw_sha256", "scale_divisor")}))
    return record


def prepare_spring_reproduction(name: str, dataset: dict) -> dict:
    """Preserve the author code's unit-column normalization and saved start."""
    source_root = ROOT / "provenance" / "spring_original_code" / "stochastic-proximal-alternating-minimization-master"
    source_names = {"orl64": ("ORL64.mat", "init_snmf_orl_s.mat"),
                    "yaleb32": ("YaleB.mat", "init_snmf_yale.mat")}
    data_name, init_name = source_names[name]
    bundled = loadmat(source_root / "data" / data_name)
    official = loadmat(ROOT / dataset["raw_path"])
    if not np.array_equal(bundled["fea"], official["fea"]) or not np.array_equal(bundled["gnd"], official["gnd"]):
        raise ValueError(f"{name}: author-bundled data differ from official originals.")
    A = np.ascontiguousarray(official["fea"].T, dtype=np.float64)
    norms = np.linalg.norm(A, axis=0)
    if np.any(norms == 0):
        raise ValueError(f"{name}: zero image invalidates original normalization.")
    A /= norms[None, :]
    initialization = loadmat(source_root / "data" / init_name)
    U0 = np.asarray(initialization["Ain"], dtype=np.float64)
    V0 = np.ascontiguousarray(initialization["xin"].T, dtype=np.float64)
    assert U0.shape[0] == A.shape[0] and V0.shape[0] == A.shape[1] and U0.shape[1] == V0.shape[1]
    metadata = {
        "dataset": name + "_unitnorm",
        "matrix_layout": "pixels_by_images",
        "scaling": "Each image divided by its original L2 norm, exactly as author demos.",
        "initialization": "Exact saved author Ain and xin, including negative entries; U0=Ain,V0=xin.T.",
        "raw_array_identical_to_author_bundle": True,
        "rank": U0.shape[1],
        "dictionary_max_nonzeros_per_column": A.shape[0] // 4,
        "norm_constraint": "Original published reproduction uses l0 plus nonnegativity; no spectral constraint.",
        "image_shape": list(SPECS[name]["image_shape"]),
    }
    path = ROOT / "data" / "processed" / f"{name}_unitnorm.npz"
    np.savez_compressed(path, A=A, U0=U0, V0=V0, labels=official["gnd"].reshape(-1).astype(np.int64), metadata=json.dumps(metadata, sort_keys=True))
    return {
        **metadata,
        "processed_path": str(path.relative_to(ROOT)),
        "processed_sha256": sha256(path),
        "matrix_sha256_float64_c_order": hashlib.sha256(A.tobytes(order="C")).hexdigest(),
        "author_data_file": str((source_root / "data" / data_name).relative_to(ROOT)),
        "author_data_sha256": sha256(source_root / "data" / data_name),
        "initialization_file": str((source_root / "data" / init_name).relative_to(ROOT)),
        "initialization_sha256": sha256(source_root / "data" / init_name),
        "column_l2_norm_max_error": float(np.max(np.abs(np.linalg.norm(A, axis=0) - 1))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=SPECS, default=list(SPECS))
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--spring-reproduction", action="store_true", help="Also export author normalization and saved initial factors from archived code.")
    args = parser.parse_args()
    manifest_path = ROOT / "provenance" / "data_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    existing = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    records = existing.get("datasets", {})
    index_path = ROOT / "provenance" / "deng_cai_face_data_index.html"
    index_info = existing.get("official_index_retrieval", {})
    if args.download and not index_path.exists():
        index_info = download(INDEX_URL, index_path)
    for name in args.datasets:
        records[name] = prepare(name, args.download, records.get(name, {}))
    reproductions = existing.get("spring_reproduction", {})
    if args.spring_reproduction:
        for name in args.datasets:
            reproductions[name] = prepare_spring_reproduction(name, records[name])
    manifest = {
        "schema_version": 1,
        "prepared_utc": datetime.now(timezone.utc).isoformat(),
        "official_index_url": INDEX_URL,
        "official_index_retrieval": index_info,
        "official_index_sha256": sha256(index_path) if index_path.exists() else None,
        "datasets": records,
        "spring_reproduction": reproductions,
        "reproduction_scope": "Exact published image dimensions; [0,1] global scaling is documented here, not claimed as verified SPRING preprocessing without source code.",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()

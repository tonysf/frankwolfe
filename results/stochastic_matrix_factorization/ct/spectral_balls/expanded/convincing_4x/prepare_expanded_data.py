"""Prepare the prespecified expanded cohort, including detector-1 correction.

Only selected TIFF members are transferred from the public ZIP archives.
Existing detector-2 data are linked without rewriting their numerical arrays.
"""

import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import re
import time
from urllib.request import Request, urlopen
import zipfile

import numpy as np
from PIL import Image
from scipy.interpolate import interp1d


ROOT = Path(__file__).resolve().parent
ORIGINAL = ROOT.parents[2]
MIRROR = (
    "https://huggingface.co/datasets/deepinv/images/resolve/main/"
    "2DeteCT_slices_4001-5000_slice04531.zip"
)
ZENODO_ARCHIVES = (
    ("1-1000", 8014758, 8017583),
    ("1001-2000", 8014766, 8017604),
    ("2001-3000", 8014787, 8017612),
    ("3001-4000", 8014829, 8017618),
    ("4001-5000", 8014874, 8017624),
    ("OOD", 8014907, 8017653),
)
PREPROCESS_SOURCE = "https://deepinv.org/auto_examples/external-libraries/demo_astra_2detect.html"
SINOGRAM_FILES = ("sinogram", "dark", "flat1", "flat2")
N_VIEWS, N_DETECTOR = 3600, 956
GROUPS = {
    "mix_1": {"label": "Mix 1", "slices": [601, 1201], "detector": 1},
    "mix_2": {"label": "Mix 2", "slices": [3041, 3381, 3611], "detector": 2},
    "mix_3": {"label": "Mix 3", "slices": [3791, 4131, 4931], "detector": 2},
    "almond_pure": {"label": "Almond (OOD Pure)", "slices": [5581, 5611], "detector": 1},
    "walnut_pure": {"label": "Walnut (OOD Pure)", "slices": [5731, 5761], "detector": 1},
    "lava_stone_pure": {"label": "Lava stone (OOD Pure)", "slices": [5831, 5861], "detector": 1},
}
NEW_SLICES = [601, 1201, 5581, 5611, 5731, 5761, 5831, 5861]
EXISTING_EVALUATION = [3041, 3381, 3611, 3791, 4131, 4931]
DEVELOPMENT = [4300, 3150]
AUTHOR_SOURCE = "https://github.com/mbkiss/2DeteCTcodes/blob/main/Reconstructions_2DeteCT.py"
PAPER_SOURCE = "https://arxiv.org/html/2306.05907"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def freeze_groups():
    payload = dict(
        status="frozen_before_new_optimization_outcomes",
        frozen_utc=datetime.now(timezone.utc).isoformat(),
        selection_basis="Prespecified well-separated slice indices within published acquisition/material groups; no optimizer outcomes inspected.",
        groups=GROUPS,
        evaluation_slices=[f"slice{i:05d}" for group in GROUPS.values() for i in group["slices"]],
        evaluation_slice_groups={f"slice{i:05d}": group for group, d in GROUPS.items() for i in d["slices"]},
        group_labels={group: d["label"] for group, d in GROUPS.items()},
        new_slices=[f"slice{i:05d}" for i in NEW_SLICES],
        existing_slices=[f"slice{i:05d}" for i in EXISTING_EVALUATION],
        development_slices=[f"slice{i:05d}" for i in DEVELOPMENT],
        dataset_source=PAPER_SOURCE,
        detector_source=AUTHOR_SOURCE,
        detector_correction="Detector 1: the published 1-index linear transmission shift with extrapolation, before log and detector reversal. Detector 2: zero shift.",
        grouping_caution="Groups are acquisition/material groups, not a claim that all slices or material classes are statistically independent specimens.",
    )
    output = ROOT / "data_group_manifest.json"
    if output.exists():
        old = json.loads(output.read_text())
        for field in ("groups", "evaluation_slices", "evaluation_slice_groups", "new_slices", "development_slices"):
            if old[field] != payload[field]:
                raise ValueError(f"Frozen cohort mismatch: {field}")
        return old
    output.write_text(json.dumps(payload, indent=2) + "\n")
    return payload


class RangeReader(io.RawIOBase):
    """Seek within a public ZIP without transferring the complete archive."""

    def __init__(self, url):
        self.url = url
        self.position = 0
        _, self.size = self._request(0, 1)

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=io.SEEK_SET):
        base = {io.SEEK_SET: 0, io.SEEK_CUR: self.position, io.SEEK_END: self.size}[whence]
        self.position = base + offset
        if self.position < 0:
            raise ValueError("Negative ZIP seek")
        return self.position

    def _request(self, start, length):
        if not 0 < length <= 24_000_000:
            raise ValueError(f"Unexpected 2DeteCT range length {length}")
        stop = start + length - 1
        request = Request(self.url, headers={"Range": f"bytes={start}-{stop}"})
        for attempt in range(5):
            try:
                with urlopen(request, timeout=60) as response:
                    match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", ""))
                    if response.status != 206 or match is None or tuple(map(int, match.group(1, 2))) != (start, stop):
                        raise OSError(f"Server did not return the requested ZIP range {start}-{stop}")
                    payload = response.read(length + 1)
                    if len(payload) != length:
                        raise OSError(f"Wrong byte count for ZIP range {start}-{stop}")
                break
            except OSError:
                if attempt == 4:
                    raise
                time.sleep(min(2 ** attempt, 8))
        return payload, int(match[3])

    def read(self, length=-1):
        remaining = max(0, self.size - self.position)
        length = remaining if length < 0 else min(remaining, length)
        if length == 0:
            return b""
        payload, total = self._request(self.position, length)
        if total != self.size:
            raise OSError("ZIP length changed while downloading")
        self.position += length
        return payload


def archive_entry(archive, slice_number, mode, name):
    suffix = f"/slice{slice_number:05d}/mode{mode}/{name}.tif"
    matching = [entry for entry in archive.namelist() if ("/" + entry).endswith(suffix)]
    if len(matching) != 1:
        raise ValueError(f"Expected exactly one archive entry ending {suffix}")
    entry = matching[0]
    return entry, archive.read(entry)


def make_slice(slice_number, mirror=None):
    freeze_groups()
    with ExitStack() as stack:
        if mirror:
            if slice_number != 4531:
                raise ValueError("The single-slice mirror only provides slice 4531")
            archive = stack.enter_context(zipfile.ZipFile(mirror))
            raw_archive = ref_archive = archive
            raw_url = ref_url = MIRROR
        else:
            if not (1 <= slice_number <= 5000 or 5521 <= slice_number <= 6370):
                raise ValueError("Slice is outside the published standard/OOD ranges")
            index = (slice_number - 1) // 1000 if slice_number <= 5000 else -1
            block, raw_record, ref_record = ZENODO_ARCHIVES[index]
            raw_url = f"https://zenodo.org/records/{raw_record}/files/2DeteCT_slices{block}.zip?download=1"
            ref_url = f"https://zenodo.org/records/{ref_record}/files/2DeteCT_slices{block}_RecSeg.zip?download=1"
            raw_archive = stack.enter_context(zipfile.ZipFile(RangeReader(raw_url)))
            ref_archive = stack.enter_context(zipfile.ZipFile(RangeReader(ref_url)))
        entries = [archive_entry(raw_archive, slice_number, 1, name) for name in SINOGRAM_FILES]
        entries.append(archive_entry(ref_archive, slice_number, 2, "reconstruction"))

    arrays = []
    for _, blob in entries:
        with Image.open(io.BytesIO(blob)) as image:
            arrays.append(np.array(image, dtype=np.float64))
    sinogram, dark, flat1, flat2, reference = arrays
    if sinogram.shape == (2 * N_DETECTOR, N_VIEWS + 1):
        sinogram = sinogram.T
    if sinogram.shape != (N_VIEWS + 1, 2 * N_DETECTOR):
        raise ValueError(f"Unexpected sinogram shape {sinogram.shape}")
    if reference.shape != (1024, 1024):
        raise ValueError(f"Unexpected reconstruction shape {reference.shape}")
    if not np.isfinite(reference).all() or reference.min() < 0:
        raise ValueError("Invalid Mode 2 reconstruction")

    def bin_detectors(arr):
        return arr[..., ::2] + arr[..., 1::2]

    dark = bin_detectors(dark.reshape(2 * N_DETECTOR))
    flat = bin_detectors(0.5 * (flat1 + flat2).reshape(2 * N_DETECTOR))
    if np.any(flat <= dark):
        raise ValueError("Mode 1 flat-field must exceed dark-field on all binned channels")
    transmission = (bin_detectors(sinogram[:-1]) - dark) / (flat - dark)
    detector_index = 1 if (slice_number <= 2830 or 5521 <= slice_number <= 5870) else 2
    shift_indices = 1.0 if detector_index == 1 else 0.0
    if shift_indices:
        # Exact published detector correction, Reconstructions_2DeteCT.py lines
        # 118--126: interpolate transmission on detGrid + corr[0]*detPix.
        # In that code detGrid has spacing detPix; corr[0]=1.0 shifts one index.
        det_grid = np.arange(N_DETECTOR) * 0.0748
        shift = interp1d(det_grid, transmission, kind="linear", bounds_error=False,
                        fill_value="extrapolate")
        transmission = shift(det_grid + shift_indices * 0.0748)
    y = np.ascontiguousarray(-np.log(np.maximum(transmission, 1e-6))[:, ::-1])
    if not np.isfinite(y).all():
        raise ValueError("Nonfinite processed sinogram")
    angles = np.pi - np.arange(N_VIEWS) * (2 * np.pi / N_VIEWS)
    det_spacing = 2 * 0.0748
    sod, sdd = 431.019989, 529.000488
    width = det_spacing * N_DETECTOR * sod / sdd
    center = -width / 1024
    metadata = dict(
        slice_number=slice_number,
        measurement_mode=1,
        reference_mode=2,
        detector_index=detector_index,
        transmission_shift_indices=shift_indices,
        raw_source=raw_url,
        reference_source=ref_url,
        preprocess_source=PREPROCESS_SOURCE,
        dataset_paper=PAPER_SOURCE,
        tiff_entries={entry: hashlib.sha256(blob).hexdigest() for entry, blob in entries},
        clamped_count=int(np.count_nonzero(transmission < 1e-6)),
        measured_shape=list(y.shape),
        reference_shape=list(reference.shape),
        sinogram_convention="Bin adjacent raw detector cells, drop last view, correct flat/dark, "
                            "apply published detector-1 transmission shift if needed, clip at 1e-6, "
                            "negate log, reverse detector (DeepInverse reflected convention).",
        reference_convention="Mode 2 TIFF stored without conversion. For DeepInverse angles "
                             "with detector reversal, a physical-mm ASTRA model uses a vertical "
                             "flip and divides attenuation by native voxel width in mm.",
        reference_source_code="https://github.com/mbkiss/2DeteCTcodes/blob/main/Reconstructions_2DeteCT.py",
        preparation_source_sha256=sha256(Path(__file__)),
        author_source_sha256=sha256(ROOT / "data_sources" / "Reconstructions_2DeteCT.py"),
        geometry_mm=dict(detector_spacing=det_spacing, source_origin=sod,
                         origin_detector=sdd-sod, volume_width=width, volume_center=[center, center],
                         reference_voxel_width=width/1024),
    )
    output = ROOT / "data" / f"slice{slice_number:05d}.npz"
    output.parent.mkdir(parents=True, exist_ok=True)
    pending = output.with_suffix(".npz.tmp")
    with pending.open("wb") as fp:
        np.savez_compressed(
            fp, y=y, reference=reference, angles_rad=angles,
            detector_spacing_mm=det_spacing, source_origin_mm=sod, origin_detector_mm=sdd-sod,
            vol_extent_mm=width, vol_center_mm=np.array([center, center]), measurement_mode=1,
        )
    pending.replace(output)
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(output, flush=True)


def link_existing():
    freeze_groups()
    (ROOT / "data").mkdir(exist_ok=True)
    for sid in EXISTING_EVALUATION + DEVELOPMENT:
        for suffix in (".npz", ".json"):
            source = ORIGINAL / "data" / f"slice{sid:05d}{suffix}"
            output = ROOT / "data" / source.name
            if not output.exists():
                output.symlink_to(source)
            elif output.resolve() != source.resolve():
                raise ValueError(f"Unexpected existing data link {output}")


def preparation_manifest():
    frozen = freeze_groups()
    records = {}
    for sid in EXISTING_EVALUATION + NEW_SLICES + DEVELOPMENT:
        data = ROOT / "data" / f"slice{sid:05d}.npz"
        metadata = data.with_suffix(".json")
        if not data.exists() or not metadata.exists():
            continue
        meta = json.loads(metadata.read_text())
        records[f"slice{sid:05d}"] = dict(
            path=str(data), sha256=sha256(data), metadata_sha256=sha256(metadata),
            detector_index=meta.get("detector_index", 2),
            transmission_shift_indices=meta.get("transmission_shift_indices", 0),
            geometry_mm=meta["geometry_mm"], clamped_count=meta["clamped_count"],
            measurement_mode=meta["measurement_mode"], reference_mode=meta["reference_mode"],
            source="existing_immutable_dataset" if sid not in NEW_SLICES else "selected_public_zip_members",
        )
    output = dict(
        status="complete" if len(records) == 16 else "partial",
        group_manifest_sha256=sha256(ROOT / "data_group_manifest.json"),
        preparation_source_sha256=sha256(Path(__file__)),
        evaluation_slice_groups=frozen["evaluation_slice_groups"],
        records=records,
    )
    (ROOT / "data_preparation_manifest.json").write_text(json.dumps(output, indent=2) + "\n")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("slice_number", type=int, nargs="?")
    parser.add_argument("--all", action="store_true", help="Prepare every frozen new slice")
    parser.add_argument("--freeze-only", action="store_true")
    parser.add_argument("--mirror", type=Path, help="Use the downloaded DeepInverse 4531 ZIP")
    args = parser.parse_args()
    link_existing()
    if not args.freeze_only:
        selected = NEW_SLICES if args.all else [args.slice_number]
        if any(sid not in NEW_SLICES for sid in selected):
            raise ValueError("Only prespecified new slices can be prepared")
        for sid in selected:
            if not (ROOT / "data" / f"slice{sid:05d}.npz").exists():
                make_slice(sid, args.mirror)
            preparation_manifest()
    print(json.dumps({"status": preparation_manifest()["status"]}, indent=2))

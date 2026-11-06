"""Fetch individual 2DeteCT slices and prepare the DeepInverse sinogram convention."""

import argparse
from contextlib import ExitStack
import hashlib
import io
import json
from pathlib import Path
import re
from urllib.request import Request, urlopen
import zipfile

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parent
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
)
PREPROCESS_SOURCE = "https://deepinv.org/auto_examples/external-libraries/demo_astra_2detect.html"
SINOGRAM_FILES = ("sinogram", "dark", "flat1", "flat2")
N_VIEWS, N_DETECTOR = 3600, 956


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
        with urlopen(request, timeout=30) as response:
            match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", ""))
            if response.status != 206 or match is None or tuple(map(int, match.group(1, 2))) != (start, stop):
                raise OSError(f"Server did not return the requested ZIP range {start}-{stop}")
            payload = response.read(length + 1)
            if len(payload) != length:
                raise OSError(f"Wrong byte count for ZIP range {start}-{stop}")
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
    with ExitStack() as stack:
        if mirror:
            if slice_number != 4531:
                raise ValueError("The single-slice mirror only provides slice 4531")
            archive = stack.enter_context(zipfile.ZipFile(mirror))
            raw_archive = ref_archive = archive
            raw_url = ref_url = MIRROR
        else:
            if not 1 <= slice_number <= 5000:
                raise ValueError("Only standard 2DeteCT slices 1–5000 are supported")
            block, raw_record, ref_record = ZENODO_ARCHIVES[(slice_number - 1) // 1000]
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
        raw_source=raw_url,
        reference_source=ref_url,
        preprocess_source=PREPROCESS_SOURCE,
        dataset_paper="https://arxiv.org/html/2306.05907",
        tiff_entries={entry: hashlib.sha256(blob).hexdigest() for entry, blob in entries},
        clamped_count=int(np.count_nonzero(transmission < 1e-6)),
        measured_shape=list(y.shape),
        reference_shape=list(reference.shape),
        sinogram_convention="DeepInverse: bin adjacent raw detector cells, drop last view, "
                            "correct flat/dark, clip at 1e-6, negate log, reverse detector",
        reference_convention="Mode 2 TIFF stored without conversion. For DeepInverse angles "
                             "with detector reversal, a physical-mm ASTRA model uses a vertical "
                             "flip and divides attenuation by native voxel width in mm.",
        reference_source_code="https://github.com/mbkiss/2DeteCTcodes/blob/main/Reconstructions_2DeteCT.py",
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("slice_number", type=int)
    parser.add_argument("--mirror", type=Path, help="Use the downloaded DeepInverse 4531 ZIP")
    args = parser.parse_args()
    make_slice(args.slice_number, args.mirror)

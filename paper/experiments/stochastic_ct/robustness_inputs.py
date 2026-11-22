"""Frozen inputs for the broader CT evaluation, without changing old experiments.

Only the view selection extends the established ASTRA model.  Calibration uses
the same 120 training views for every condition.  Alternative starts share the
reference spectral radii and curvature scales; changing a start never changes
the optimization domain.  Fresh confirmation identifiers are reserved from
metadata before their measurements or optimization outcomes are inspected.
"""

import os

for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[_name] = "1"

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

try:
    from . import ct_problem as base_problem
except ImportError:
    import ct_problem as base_problem


REFERENCE = Path('/home/tony/frames_ct_2detect_20261104/spectral_balls/expanded/convincing_4x_recovery')
ROOT = REFERENCE.parent / 'reviewer_robustness_20261122'
TRAINING_VIEWS = (120, 360, 1080, 2160)
CALIBRATION_VIEWS = np.arange(0, 3600, 30, dtype=np.intp)
GROUP_RANGES = {
    'Mix1': [1, 1800, 100], 'Mix2': [1801, 3720, 100],
    'Mix3': [3721, 5000, 100], 'Almond': [5571, 5620, 10],
    'Walnut': [5721, 5770, 10], 'LavaStone': [5821, 5870, 10],
}
ARCHIVES = (
    ('1-1000', 8014758, 8017583), ('1001-2000', 8014766, 8017604),
    ('2001-3000', 8014787, 8017612), ('3001-4000', 8014829, 8017618),
    ('4001-5000', 8014874, 8017624), ('OOD', 8014907, 8017653),
)


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def view_split(n_train):
    if isinstance(n_train, bool) or n_train not in TRAINING_VIEWS:
        raise ValueError('Training views must be 120, 360, 1080, or 2160')
    if n_train == 120:
        train = CALIBRATION_VIEWS.copy()
    else:
        offsets = {360: [0], 1080: [0, 3, 6], 2160: [0, 1, 3, 4, 6, 7]}[n_train]
        train = np.sort((np.arange(0, 3600, 10)[:, None] + offsets).ravel()).astype(np.intp)
    return train, np.arange(9, 3600, 10, dtype=np.intp)


class NewCTProblem(base_problem.CTProblem):
    """The frozen CT operator with an extended, nested train-view selection."""

    def __init__(self, data_file, size, n_train):
        train, validation = view_split(n_train)
        # The base constructor does not apply any operator.  Keep its geometry,
        # sinogram precision, reference conversion and projector implementation.
        super().__init__(data_file, size, 360)
        self.train_ids, self.validation_ids = train, validation


class _CalibrationProblem(NewCTProblem):
    def __init__(self, *args, **kwargs):
        self.calibration_work = dict(forward_calls=0, backward_calls=0,
                                     forward_views=0, backward_views=0)
        super().__init__(*args, **kwargs)

    def forward(self, image, views):
        value = super().forward(image, views)
        self.calibration_work['forward_calls'] += 1
        self.calibration_work['forward_views'] += len(views)
        return value

    def backward(self, sinogram, views):
        value = super().backward(sinogram, views)
        self.calibration_work['backward_calls'] += 1
        self.calibration_work['backward_views'] += len(views)
        return value


def input_key(slice_id, rank=64, initialization='standard'):
    return f'{slice_id}:{int(rank)}:{initialization}'


def get_input(inputs, slice_id, rank=64, initialization='standard'):
    return dict(inputs[input_key(slice_id, rank, initialization)])


def select_confirmation(excluded_ids):
    old = [int(s[5:]) for s in excluded_ids]
    chosen = {}
    for group, (low, high, separation) in GROUP_RANGES.items():
        eligible = [i for i in range(low, high + 1)
                    if all(abs(i - j) >= separation for j in old)]
        if not eligible:
            raise ValueError(f'No separated confirmation slice remains in {group}')
        chosen[f'slice{eligible[(len(eligible) - 1) // 2]:05d}'] = group
    return chosen


def reserve_confirmation(root=ROOT, reference=REFERENCE):
    """Inspect filenames/metadata only, then freeze identifiers before loading data."""
    root, reference = Path(root), Path(reference)
    destination = root / 'confirmation_reservation.json'
    if destination.exists():
        reservation = read(destination)
        if (reservation['reference_cohort_sha256'] != sha256(reference / 'cohort.json')
                or reservation['group_ranges_and_minimum_separation'] != GROUP_RANGES
                or reservation['confirmation_slice_groups'] != select_confirmation(reservation['excluded_slice_ids'])):
            raise ValueError('The frozen confirmation reservation changed')
        for record in reservation['exclusion_metadata'].values():
            if sha256(record['path']) != record['sha256']:
                raise ValueError('Historical slice inventory evidence changed')
        return reservation
    evidence = {}
    experiment_root = reference.parents[2]
    for directory, dirs, files in os.walk(experiment_root):
        dirs[:] = [d for d in dirs if d not in ('deps', '__pycache__', '.git', root.name)]
        for name in sorted(files):
            if name != 'metadata.json' and not re.fullmatch(r'slice\d{5}\.json', name):
                continue
            path = Path(directory) / name
            for sid in re.findall(r'slice\d{5}', str(path)):
                evidence.setdefault(sid, dict(path=str(path.resolve()), sha256=sha256(path)))
    if not evidence:
        raise ValueError('No historical slice metadata found for reservation')
    selected = select_confirmation(sorted(evidence))
    reservation = dict(schema_version=1, status='reserved_before_new_data_or_optimizer_outcomes',
        frozen_utc=datetime.now(timezone.utc).isoformat(),
        selection_rule='Lower median eligible integer slice ID in each published group; exclude all historically prepared/run slice IDs and require distance at least 100 for mixed-sample groups or 10 for pure-sample groups.',
        source='https://arxiv.org/html/2306.05907#S2.T2',
        reference_cohort_sha256=sha256(reference / 'cohort.json'),
        group_ranges_and_minimum_separation=GROUP_RANGES,
        excluded_slice_ids=sorted(evidence), exclusion_metadata=evidence,
        confirmation_slices=sorted(selected), confirmation_slice_groups=selected,
        interpretation='Fresh slices within six already studied acquisition/material groups; not six new independent specimen groups.')
    write(destination, reservation)
    return reservation


def cohort(root=ROOT, reference=REFERENCE):
    old = read(Path(reference) / 'cohort.json')
    reservation = reserve_confirmation(root, reference)
    groups = old['evaluation_slice_groups']
    panel = []
    for group in sorted(set(groups.values())):
        ids = sorted(sid for sid, label in groups.items() if label == group)
        panel.append(ids[(len(ids) - 1) // 2])
    return dict(development_slices=sorted(old['development_slices']),
        evaluation_slices=sorted(old['evaluation_slices']), evaluation_slice_groups=groups,
        confirmation_slices=reservation['confirmation_slices'],
        confirmation_slice_groups=reservation['confirmation_slice_groups'],
        panel_slices=sorted(panel), panel_selection='Lower median existing evaluation slice ID within each group; no outcomes used.',
        training_views=list(TRAINING_VIEWS), development_calibration_seed=110,
        evaluation_calibration_seed=201, reference_cohort_sha256=sha256(Path(reference) / 'cohort.json'),
        confirmation_reservation_sha256=sha256(Path(root) / 'confirmation_reservation.json'))


class _RangeReader(io.RawIOBase):
    """Read only requested TIFF members of a remote ZIP, never the whole archive."""

    def __init__(self, url):
        self.url, self.position = url, 0
        _, self.size = self._request(0, 1)

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=io.SEEK_SET):
        self.position = {io.SEEK_SET: 0, io.SEEK_CUR: self.position, io.SEEK_END: self.size}[whence] + offset
        if self.position < 0:
            raise ValueError('Negative ZIP seek')
        return self.position

    def _request(self, start, length):
        if not 0 < length <= 24_000_000:
            raise ValueError('Unexpected ZIP range size')
        stop = start + length - 1
        request = Request(self.url, headers={'Range': f'bytes={start}-{stop}'})
        for attempt in range(5):
            try:
                with urlopen(request, timeout=45) as response:
                    match = re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)', response.headers.get('Content-Range', ''))
                    if response.status != 206 or not match or tuple(map(int, match.group(1, 2))) != (start, stop):
                        raise OSError('Server did not honor the requested ZIP range')
                    payload = response.read(length + 1)
                    if len(payload) != length:
                        raise OSError('Wrong ZIP range byte count')
                return payload, int(match[3])
            except OSError:
                if attempt == 4:
                    raise
                time.sleep(min(2 ** attempt, 8))

    def read(self, length=-1):
        remaining = max(0, self.size - self.position)
        length = remaining if length < 0 else min(remaining, length)
        if not length:
            return b''
        value, size = self._request(self.position, length)
        if size != self.size:
            raise OSError('Remote ZIP changed during transfer')
        self.position += length
        return value


def prepare_confirmation_data(root=ROOT, reference=REFERENCE, *, only=None):
    """Explicit network stage; cached, checksummed inputs make restart idempotent."""
    from PIL import Image
    from scipy.interpolate import interp1d

    root = Path(root)
    reservation = reserve_confirmation(root, reference)
    requested = reservation['confirmation_slices'] if only is None else list(only)
    if not set(requested) <= set(reservation['confirmation_slices']):
        raise ValueError('Only frozen confirmation slices can be downloaded')
    records = {}
    for sid in requested:
        output = root / 'data' / (sid + '.npz')
        receipt = output.with_suffix('.json')
        if output.exists() and receipt.exists():
            record = read(receipt)
            if record['npz_sha256'] != sha256(output) or record['slice_id'] != sid:
                raise ValueError('Prepared confirmation data changed')
            records[sid] = record
            continue
        number = int(sid[5:])
        block, raw_record, ref_record = ARCHIVES[(number - 1) // 1000 if number <= 5000 else -1]
        urls = [f'https://zenodo.org/records/{raw_record}/files/2DeteCT_slices{block}.zip?download=1',
                f'https://zenodo.org/records/{ref_record}/files/2DeteCT_slices{block}_RecSeg.zip?download=1']
        entries = []
        with ExitStack() as stack:
            for url, mode, names in ((urls[0], 1, ('sinogram', 'dark', 'flat1', 'flat2')),
                                     (urls[1], 2, ('reconstruction',))):
                archive = None
                for name in names:
                    target = root / 'raw' / sid / f'mode{mode}' / (name + '.tif')
                    target_receipt = target.with_suffix('.sha256.json')
                    if target.exists() and target_receipt.exists():
                        entry = read(target_receipt)
                        if sha256(target) != entry['sha256'] or entry['archive_url'] != url:
                            raise ValueError('Cached confirmation TIFF changed')
                    else:
                        if archive is None:
                            archive = stack.enter_context(zipfile.ZipFile(_RangeReader(url)))
                        suffix = f'/{sid}/mode{mode}/{name}.tif'
                        matches = [item for item in archive.namelist() if ('/' + item).endswith(suffix)]
                        if len(matches) != 1:
                            raise ValueError('Expected exactly one requested TIFF member')
                        blob = archive.read(matches[0])
                        target.parent.mkdir(parents=True, exist_ok=True)
                        temporary = target.with_suffix('.tif.tmp')
                        temporary.write_bytes(blob)
                        temporary.replace(target)
                        entry = dict(archive_url=url, member=matches[0], sha256=sha256(target))
                        write(target_receipt, entry)
                    with Image.open(target) as image:
                        entries.append((entry, np.array(image, dtype=np.float64)))
        sinogram, dark, flat1, flat2, ref = [array for _, array in entries]
        if sinogram.shape == (1912, 3601):
            sinogram = sinogram.T
        if sinogram.shape != (3601, 1912) or ref.shape != (1024, 1024):
            raise ValueError('Unexpected confirmation TIFF dimensions')
        if not np.isfinite(ref).all() or np.any(ref < 0):
            raise ValueError('Invalid high-dose reference')
        bin_channels = lambda a: a[..., ::2] + a[..., 1::2]
        dark = bin_channels(dark.reshape(1912))
        flat = bin_channels((0.5 * (flat1 + flat2)).reshape(1912))
        if np.any(flat <= dark):
            raise ValueError('Flat field must exceed dark field')
        transmission = (bin_channels(sinogram[:-1]) - dark) / (flat - dark)
        detector = 1 if number <= 2830 or 5521 <= number <= 5870 else 2
        if detector == 1:
            grid = np.arange(956) * 0.0748
            transmission = interp1d(grid, transmission, kind='linear', bounds_error=False,
                                   fill_value='extrapolate')(grid + 0.0748)
        y = np.ascontiguousarray(-np.log(np.maximum(transmission, 1e-6))[:, ::-1])
        if not np.isfinite(y).all():
            raise ValueError('Nonfinite processed measurements')
        spacing, source, detector_distance = 0.1496, 431.019989, 529.000488 - 431.019989
        width = spacing * 956 * source / (source + detector_distance)
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix('.npz.tmp')
        with temporary.open('wb') as stream:
            np.savez_compressed(stream, y=y, reference=ref,
                angles_rad=np.pi - np.arange(3600) * (2 * np.pi / 3600),
                detector_spacing_mm=spacing, source_origin_mm=source,
                origin_detector_mm=detector_distance, vol_extent_mm=width,
                vol_center_mm=np.array([-width / 1024, -width / 1024]), measurement_mode=1)
        temporary.replace(output)
        record = dict(slice_id=sid, npz_sha256=sha256(output), measurement_mode=1, reference_mode=2,
            detector_index=detector, transmission_shift_indices=1 if detector == 1 else 0,
            clamped_count=int(np.count_nonzero(transmission < 1e-6)),
            tiff_sources=[entry for entry, _ in entries],
            preparation_source_sha256=sha256(__file__),
            source_convention='Same binning, detector correction, signed log measurements and geometry as the frozen expanded CT cohort.',
            confirmation_reservation_sha256=sha256(root / 'confirmation_reservation.json'))
        write(receipt, record)
        records[sid] = record
        print(f'Prepared confirmation measurements: {sid}', flush=True)
    return records


def alternative_start(problem, U, V, reference_caps, initialization, seed):
    """A materially changed positive start, amplitude-fitted within fixed bounds."""
    if initialization not in ('perturbed', 'random'):
        raise ValueError('Expected perturbed or random initialization')
    rng = np.random.default_rng(seed)
    if initialization == 'perturbed':
        factors = [np.maximum(a + 0.5 * np.sqrt(np.mean(a*a)) * rng.standard_normal(a.shape), 0)
                   for a in (U, V)]
    else:
        factors = [rng.random(a.shape) * np.sqrt(np.mean(a*a)) for a in (U, V)]
    views = CALIBRATION_VIEWS
    if not np.isin(views, problem.train_ids).all():
        raise ValueError('Initialization would query nontraining views')
    modeled = problem.forward(factors[0] @ factors[1].T, views).astype(np.float64)
    observed = np.asarray(problem.y[views], dtype=np.float64)
    denominator = float(np.sum(modeled * modeled))
    if denominator <= 0 or not np.isfinite(denominator):
        raise ValueError('Degenerate alternative initialization')
    amplitude = max(float(np.sum(modeled * observed) / denominator), 1e-6)
    norms = [float(np.linalg.norm(a, 2)) for a in factors]
    if min(norms) <= 0:
        raise ValueError('Degenerate alternative factor')
    # Fit inside even the smallest requested radius_scale=1.  Reciprocal factor
    # balancing preserves the fitted image while equalizing relative cap usage.
    limits = np.asarray(reference_caps, dtype=float) / 2
    capacity = (limits[0] / norms[0]) * (limits[1] / norms[1])
    fitted = min(amplitude, capacity)
    balance = (limits[0] / norms[0]) / (limits[1] / norms[1])
    factors[0] *= np.sqrt(fitted * balance)
    factors[1] *= np.sqrt(fitted / balance)
    # This defensive inward rounding changes neither the frozen radii nor the
    # reference initialization; record the actual factor arrays and fit scalars.
    for index, factor in enumerate(factors):
        norm = float(np.linalg.norm(factor, 2))
        if norm > limits[index]:
            factor *= limits[index] / norm
    return (*factors, dict(initialization=initialization, initialization_seed=int(seed),
        unconstrained_amplitude=amplitude, fitted_amplitude=fitted,
        amplitude_limited_by_reference_spectral_bounds=amplitude > capacity,
        fit_views=views.tolist(), reference_curvature_scales_retained=True))


def _calibration_record(data, calibration, seed, rank, initialization):
    with np.load(calibration, allow_pickle=False) as saved:
        if str(saved['data_sha256']) != sha256(data):
            raise ValueError('Calibration data digest mismatch')
        if str(saved['problem_sha256']) != sha256(base_problem.__file__):
            raise ValueError('Calibration base CT implementation changed')
        U, V = saved['U'], saved['V']
        if U.shape != (512, rank) or V.shape != U.shape or not np.isfinite(U).all() or not np.isfinite(V).all():
            raise ValueError('Invalid calibration factors')
        if np.any(U < 0) or np.any(V < 0):
            raise ValueError('Expected nonnegative calibration factors')
        caps, curvature = saved['caps'], saved['curvature']
        if caps.shape != (2,) or curvature.shape != (2,) or np.any(caps <= 0) or np.any(curvature <= 0):
            raise ValueError('Invalid fixed calibration scales')
        if initialization == 'standard' and not np.allclose(caps, [2*np.linalg.norm(U, 2), 2*np.linalg.norm(V, 2)], rtol=1e-12, atol=0):
            raise ValueError('Reference radii must be spectral, not Frobenius')
        if any(np.linalg.norm(a, 2) > cap/2*(1 + 1e-12) for a, cap in zip((U, V), caps)):
            raise ValueError('Start does not fit the smallest reference spectral balls')
        return dict(data_file=str(Path(data).resolve()), data_sha256=sha256(data),
            calibration_file=str(Path(calibration).resolve()), calibration_sha256=sha256(calibration),
            calibration_seed=int(seed), initialization=initialization, rank=rank,
            calibration_problem_sha256=str(saved['problem_sha256']),
            reference_caps=caps.tolist(), reference_curvatures=curvature.tolist())


def _prepare_calibration(root, data, sid, rank, initialization, seed, standard=None):
    output = root / 'calibration' / f'{sid}_rank{rank}_{initialization}_seed{seed}.npz'
    receipt = output.with_suffix('.json')
    if output.exists() and receipt.exists():
        record = read(receipt)
        if (record['calibration_sha256'] != sha256(output)
                or record['preparation_source_sha256'] != sha256(__file__)
                or record['data_sha256'] != sha256(data)):
            raise ValueError('Prepared calibration provenance changed')
        return output
    problem = _CalibrationProblem(data, 512, 120)
    try:
        if initialization == 'standard':
            U, V, _, curvature = problem.calibration(rank, seed)
            # Base calibration historically returned Frobenius caps; the new
            # experiment always uses the requested product of spectral balls.
            caps = np.array([2*np.linalg.norm(U, 2), 2*np.linalg.norm(V, 2)])
            details = dict(initialization='standard', initialization_seed=seed,
                           fit_views=CALIBRATION_VIEWS.tolist())
        else:
            if standard is None:
                raise ValueError('Alternative start requires a fixed reference calibration')
            with np.load(standard, allow_pickle=False) as saved:
                U, V = saved['U'].copy(), saved['V'].copy()
                caps, curvature = saved['caps'].copy(), saved['curvature'].copy()
            U, V, details = alternative_start(problem, U, V, caps, initialization, seed)
            details['reference_calibration_sha256'] = sha256(standard)
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix('.npz.tmp')
        with temporary.open('wb') as stream:
            np.savez_compressed(stream, U=U, V=V, caps=caps, curvature=curvature,
                data_sha256=sha256(data), problem_sha256=sha256(base_problem.__file__))
        temporary.replace(output)
        write(receipt, dict(calibration_sha256=sha256(output), data_sha256=sha256(data),
            preparation_source_sha256=sha256(__file__), base_problem_sha256=sha256(base_problem.__file__),
            rank=rank, **details, calibration_work=problem.calibration_work,
            work_accounting='Preparation work, recorded separately from solver and diagnostic work.'))
    finally:
        problem.close()
    return output


def prepare_inputs(root=ROOT, reference=REFERENCE, *, include_confirmation=True,
                   prepare_ablations=True, allow_download=False):
    """Return (cohort, flat inputs); old rank-64 standard files are never rewritten.

    Rank32/128 reference calibrations and rank64 alternative starts are prepared
    only on the prespecified six-slice panel.  The confirmation set uses rank64
    standard starts and is excluded from all development selection.
    """
    root, reference = Path(root), Path(reference)
    frozen_cohort = cohort(root, reference)
    if allow_download:
        if not include_confirmation:
            raise ValueError('Downloads require include_confirmation=True')
        prepare_confirmation_data(root, reference)
    old_slices = frozen_cohort['development_slices'] + frozen_cohort['evaluation_slices']
    slices = old_slices + (frozen_cohort['confirmation_slices'] if include_confirmation else [])
    inputs = {}
    for sid in slices:
        data = (reference / 'data' / f'{sid}.npz') if sid in old_slices else (root / 'data' / f'{sid}.npz')
        if not data.is_file():
            raise FileNotFoundError(f'Prepare the reserved confirmation data first: {data}')
        seed = 110 if sid in frozen_cohort['development_slices'] else 201
        standard = reference / 'calibration' / f'{sid}_seed{seed}.npz' if sid in old_slices else None
        if standard is None:
            standard = _prepare_calibration(root, data, sid, 64, 'standard', seed)
        inputs[input_key(sid)] = _calibration_record(data, standard, seed, 64, 'standard')
        if prepare_ablations and sid in frozen_cohort['panel_slices']:
            for rank in (32, 128):
                path = _prepare_calibration(root, data, sid, rank, 'standard', seed)
                inputs[input_key(sid, rank)] = _calibration_record(data, path, seed, rank, 'standard')
            for start, offset in (('perturbed', 10_000), ('random', 20_000)):
                path = _prepare_calibration(root, data, sid, 64, start, seed + offset, standard=standard)
                inputs[input_key(sid, 64, start)] = _calibration_record(data, path, seed + offset, 64, start)
    write(root / 'prepared_inputs.json', dict(schema_version=1, cohort=frozen_cohort, inputs=inputs,
        preparation_source_sha256=sha256(__file__), base_problem_sha256=sha256(base_problem.__file__),
        include_confirmation=include_confirmation, prepare_ablations=prepare_ablations))
    return frozen_cohort, inputs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--reference', type=Path, default=REFERENCE)
    parser.add_argument('--reserve-only', action='store_true')
    parser.add_argument('--download-confirmation', action='store_true')
    parser.add_argument('--without-confirmation', action='store_true')
    parser.add_argument('--without-ablations', action='store_true')
    args = parser.parse_args(argv)
    if args.reserve_only:
        result = reserve_confirmation(args.root, args.reference)
        print(json.dumps(result, indent=2))
    else:
        selected, inputs = prepare_inputs(args.root, args.reference,
            include_confirmation=not args.without_confirmation,
            prepare_ablations=not args.without_ablations,
            allow_download=args.download_confirmation)
        print(json.dumps(dict(status='prepared', slices=len(selected['evaluation_slices']),
                             input_configurations=len(inputs)), indent=2))


if __name__ == '__main__':
    main()

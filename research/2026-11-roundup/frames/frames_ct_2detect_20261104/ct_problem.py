"""Matched CPU fan-beam views and a factored finite-sum least-squares loss."""

from collections import OrderedDict
from pathlib import Path

import astra
import numpy as np


def view_split(n_train):
    # The smaller training set is nested inside the larger one.
    offsets = {360: [0], 1080: [0, 3, 6]}[n_train]
    train = np.sort((np.arange(0, 3600, 10)[:, None] + offsets).ravel())
    return train, np.arange(9, 3600, 10)


class CTProblem:
    def __init__(self, data_file, size, n_train):
        self.data_file = Path(data_file)
        self.size = size
        self.train_ids, self.validation_ids = view_split(n_train)
        with np.load(data_file, allow_pickle=False) as data:
            self.y = np.ascontiguousarray(data["y"], dtype=np.float32)
            self.angles = np.array(data["angles_rad"], dtype=np.float64)
            self.spacing = float(data["detector_spacing_mm"])
            self.source = float(data["source_origin_mm"])
            self.detector = float(data["origin_detector_mm"])
            self.width = float(data["vol_extent_mm"])
            x, y = data["vol_center_mm"]
            reference = data["reference"] if "reference" in data.files else None
        if self.y.shape != (3600, 956) or self.angles.shape != (3600,):
            raise ValueError("Expected 3600 distinct angles and 956 binned detector pixels")
        if not np.all(np.isfinite(self.y)):
            raise ValueError("Nonfinite sinogram")
        self.detectors = self.y.shape[1]
        half = self.width / 2
        self.volume_geometry = astra.create_vol_geom(
            size, size, x-half, x+half, y-half, y+half)
        self.reference = None
        if reference is not None and reference.size:
            if reference.shape[0] != reference.shape[1] or reference.shape[0] % size:
                raise ValueError("Reference needs an integer reduction to the selected grid")
            ratio = reference.shape[0] // size
            # The preprocessed sinogram uses DeepInv's reflected convention.
            # Reference TIFFs use native-pixel path lengths; our projector uses mm.
            attenuation = reference[::-1] / (self.width / reference.shape[0])
            self.reference = attenuation.reshape(size, ratio, size, ratio).mean(axis=(1, 3))
        self.projectors = OrderedDict()

    def _projector(self, views):
        key = np.asarray(views, dtype=np.int64).tobytes()
        if key not in self.projectors:
            geom = astra.create_proj_geom("fanflat", self.spacing, self.detectors,
                                          self.angles[views], self.source, self.detector)
            self.projectors[key] = astra.create_projector("line_fanflat", geom, self.volume_geometry)
            if len(self.projectors) > 8:
                astra.projector.delete(self.projectors.popitem(last=False)[1])
        self.projectors.move_to_end(key)
        return self.projectors[key]

    def close(self):
        for handle in self.projectors.values():
            astra.projector.delete(handle)
        self.projectors.clear()

    def forward(self, image, views):
        out = np.zeros((len(views), self.detectors), dtype=np.float32)
        astra.projector.direct_FP(self._projector(views),
                                  np.ascontiguousarray(image, dtype=np.float32), out=out)
        return out

    def backward(self, sinogram, views):
        out = np.zeros((self.size, self.size), dtype=np.float32)
        astra.projector.direct_BP(self._projector(views),
                                  np.ascontiguousarray(sinogram, dtype=np.float32), out=out)
        return out

    def image_loss(self, image, views):
        residual = self.forward(image, views) - self.y[views]
        return float(0.5 * np.mean(np.square(residual, dtype=np.float64)))

    def grad(self, U, V, views, *, block=None):
        residual = self.forward(U @ V.T, views) - self.y[views]
        adjoint = self.backward(residual, views).astype(np.float64) / residual.size
        if block == 0:
            return adjoint @ V
        if block == 1:
            return adjoint.T @ U
        return adjoint @ V, adjoint.T @ U

    def initial_image(self):
        # The same 120 allowed angles are used at both training-view counts.
        views = self.train_ids[self.train_ids % 30 == 0]
        predicted = self.forward(np.ones((self.size, self.size)), views)
        normalizer = self.backward(predicted, views)
        floor = float(normalizer.max()) * 1e-3
        image = np.maximum(self.backward(self.y[views], views), 0) / np.maximum(normalizer, floor)
        return image.astype(np.float64), views

    def calibration(self, rank, seed):
        image, views = self.initial_image()
        U, V = nonnegative_start(image, rank, seed)
        modeled = self.forward(U @ V.T, views).astype(np.float64)
        observed = self.y[views].astype(np.float64)
        amplitude = max(float(np.sum(modeled * observed) / np.sum(modeled**2)), 1e-6)
        U *= np.sqrt(amplitude)
        V *= np.sqrt(amplitude)
        caps = (2 * float(np.linalg.norm(U)), 2 * float(np.linalg.norm(V)))

        direction = np.ones_like(image)
        direction /= np.linalg.norm(direction)
        for _ in range(10):
            next_direction = self.backward(self.forward(direction, views), views).astype(np.float64)
            eigenvalue = np.linalg.norm(next_direction)
            direction = next_direction / eigenvalue
        image_curvature = float(eigenvalue) / (len(views) * self.detectors)
        block_u = image_curvature * float(np.linalg.norm(V, 2))**2
        block_v = image_curvature * float(np.linalg.norm(U, 2))**2
        return U, V, caps, (block_u, block_v)


def nonnegative_start(image, rank, seed):
    left, sigma, right = np.linalg.svd(image, full_matrices=False)
    U = np.zeros((image.shape[0], rank))
    V = np.zeros((image.shape[1], rank))
    for k in range(rank):
        u, v = left[:, k], right[k, :]
        candidates = [(np.maximum(u, 0), np.maximum(v, 0)),
                      (np.maximum(-u, 0), np.maximum(-v, 0))]
        p, q = max(candidates, key=lambda pair: np.linalg.norm(pair[0]) * np.linalg.norm(pair[1]))
        norm_p, norm_q = np.linalg.norm(p), np.linalg.norm(q)
        if norm_p > 0 and norm_q > 0:
            scale = np.sqrt(sigma[k] * norm_p * norm_q)
            U[:, k], V[:, k] = p * (scale / norm_p), q * (scale / norm_q)
    # Reproducible, small, strictly positive perturbation of the NNDSVD start.
    rng = np.random.default_rng(seed)
    U += 1e-3 * np.sqrt(np.mean(U**2)) * rng.random(U.shape)
    V += 1e-3 * np.sqrt(np.mean(V**2)) * rng.random(V.shape)
    return U, V

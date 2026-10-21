
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.datasets import make_blobs, make_circles, make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


@dataclass
class DataSplit:
    x_train: np.ndarray
    x_validation: np.ndarray
    x_test: np.ndarray
    y_train: np.ndarray
    y_validation: np.ndarray
    y_test: np.ndarray
    scaler: StandardScaler
    raw_x: np.ndarray
    raw_y: np.ndarray


def _spiral(samples: int, noise: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    per_class = samples // 2
    remainder = samples - 2 * per_class
    count0, count1 = per_class, per_class + remainder
    def arm(count: int, offset: float) -> np.ndarray:
        radius = np.linspace(0.05, 1.0, count)
        theta = 1.75 * np.pi * radius + offset
        return np.column_stack((radius * np.cos(theta), radius * np.sin(theta)))
    x = np.vstack((arm(count0, 0.0), arm(count1, np.pi)))
    x += rng.normal(0.0, noise, x.shape)
    return x, np.r_[np.zeros(count0, dtype=int), np.ones(count1, dtype=int)]


def generate_dataset(
    name: str,
    samples: int = 800,
    noise: float = 0.12,
    separation: float = 1.5,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate balanced, binary, 2D synthetic data.

    ``separation`` affects blob centers and circle factor.  XOR uses it as an
    axis scale, while Two Moons and Spiral primarily use the noise control.
    """
    if not 100 <= samples <= 5000:
        raise ValueError("samples must be between 100 and 5000")
    if noise < 0:
        raise ValueError("noise must be non-negative")
    if separation <= 0:
        raise ValueError("separation must be positive")
    name = name.lower()
    rng = np.random.default_rng(seed)
    if name == "xor":
        x = rng.uniform(-1.0, 1.0, size=(samples, 2)) * separation
        y = ((x[:, 0] * x[:, 1]) < 0).astype(int)
        x += rng.normal(0.0, noise, size=x.shape)
    elif name == "two moons":
        x, y = make_moons(n_samples=samples, noise=noise, random_state=seed)
    elif name == "concentric circles":
        factor = float(np.clip(1.0 / (1.0 + separation), 0.15, 0.85))
        x, y = make_circles(n_samples=samples, factor=factor, noise=noise, random_state=seed)
    elif name == "gaussian blobs":
        centers = np.array([[-separation / 2, -separation / 3], [separation / 2, separation / 3]])
        x, y = make_blobs(n_samples=samples, centers=centers, cluster_std=max(0.05, noise * 3.0), random_state=seed)
    elif name == "spiral":
        x, y = _spiral(samples, noise, seed)
    else:
        raise ValueError("unknown dataset: " + name)
    return np.asarray(x, dtype=float), np.asarray(y, dtype=int)


def split_and_scale(x: np.ndarray, y: np.ndarray, seed: int = 42) -> DataSplit:
    """Make stratified 60/20/20 splits and fit scaling on training data only."""
    x_train, x_holdout, y_train, y_holdout = train_test_split(
        x, y, test_size=0.40, stratify=y, random_state=seed
    )
    x_validation, x_test, y_validation, y_test = train_test_split(
        x_holdout, y_holdout, test_size=0.50, stratify=y_holdout, random_state=seed + 1
    )
    scaler = StandardScaler().fit(x_train)
    return DataSplit(
        scaler.transform(x_train), scaler.transform(x_validation), scaler.transform(x_test),
        np.asarray(y_train), np.asarray(y_validation), np.asarray(y_test), scaler,
        np.asarray(x), np.asarray(y),
    )

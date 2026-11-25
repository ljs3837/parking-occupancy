"""Synthetic partial occlusion; complete slot labels stay unchanged."""
from __future__ import annotations
import cv2
import numpy as np


def occlude_slots(image, polygons, seed=42):
    """Cover a small portion of selected slots without changing label geometry.

    This supplements real occlusions; it cannot teach recovery of invisible,
    unannotated slots. Coordinates are normalized to the image dimensions.
    """
    rng = np.random.default_rng(seed)
    output = image.copy()
    height, width = image.shape[:2]
    changed = 0
    for polygon in polygons:
        points = np.asarray(polygon["points"], dtype=float).reshape(-1, 2)
        if len(points) < 3 or not np.isfinite(points).all() or rng.random() > .2:
            continue
        points = points * (width, height)
        low, high = points.min(0), points.max(0)
        extent = high - low
        if extent.min() < 12:
            continue
        # One narrow obstacle crosses part of a boundary, never the entire slot.
        center = points[int(rng.integers(len(points)))]
        size = extent * rng.uniform(.18, .35, size=2)
        start = np.maximum(np.floor(center - size / 2), 0).astype(int)
        end = np.minimum(np.ceil(center + size / 2), (width, height)).astype(int)
        if np.any(end <= start):
            continue
        color = tuple(int(v) for v in rng.integers(25, 140, size=3))
        cv2.rectangle(output, tuple(start), tuple(end - 1), color, -1)
        changed += 1
    return output, changed

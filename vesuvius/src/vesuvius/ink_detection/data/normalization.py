"""Ink image normalization and flat-window masking semantics."""

from __future__ import annotations

import math
from functools import lru_cache

import numpy as np

from vesuvius.ink_detection.config import NormalizationConfig
import vc_reader.transforms as A
from functools import lru_cache


@lru_cache(maxsize=32)
def _normalizer(mode, options):
    return A.Normalize(mode, **dict(options))


def normalize_image(
    image: np.ndarray, config: NormalizationConfig
) -> np.ndarray:
    """Normalize one crop as float32 using the selected mode."""
    image_f32 = np.asarray(image).astype(np.float32, copy=False)
    if image_f32.size == 0 or config.mode == "none":
        return image_f32
    if config.mode == "divide":
        mode, options = "fixed", dict(scale=float(config.divisor))
    elif config.mode == "clip_zscore":
        mode, options = "fixed", dict(center=float(config.mean), scale=float(config.std),
                                     bounds=(float(config.clip_min), float(config.clip_max)))
    elif config.mode == "robust_mad":
        mode, options = "robust", dict(lower=float(config.percentile_lower), upper=float(config.percentile_upper))
    elif config.mode == "minmax":
        mode, options = "percentile", dict(lower=0.0, upper=100.0)
    else:
        mode, options = "percentile", dict(lower=float(config.percentile_lower), upper=float(config.percentile_upper),
                                           median=config.mode == "robust_percentile_span")
    normalize = _normalizer(mode, tuple(sorted(options.items())))
    return normalize(image_f32.reshape(1, 1, -1)).reshape(image_f32.shape)


# The percentile modes read only order statistics, which a histogram fixes;
# the fixed modes map each voxel on its own. Either way a uint8 crop
# normalizes through one 256-entry table.
HISTOGRAM_TABLE_MODES = frozenset({"percentile_minmax", "minmax"})
_FIXED_TABLE_MODES = frozenset({"none", "divide", "clip_zscore"})
UINT8_TABLE_MODES = HISTOGRAM_TABLE_MODES | _FIXED_TABLE_MODES


def uint8_normalization_table(
    histogram: np.ndarray | None, config: NormalizationConfig
) -> np.ndarray:
    """Return the float32 value normalize_image gives each uint8 code.

    ``table[crop]`` equals ``normalize_image(crop, config)`` bit for bit for
    a uint8 ``crop`` whose 256-bin histogram is ``histogram``. The fixed modes
    ignore the histogram (pass None). The percentile modes repeat vc_reader's
    float64 steps: NumPy's linear virtual index and its two-sided lerp
    (vc_reader/src/cpp/transforms/shared/percentile.hpp), a float32 clamp,
    then ``(x - low) / (high - low)`` in float64 rounded to float32, and
    zeros when the span is under 1e-6 (normalize_order in
    vc_reader/src/cpp/transforms/cpu/normalization.hpp). Those steps assume
    a build without fused multiply-adds; see uint8_table_matches_normalizer.
    """
    if config.mode in _FIXED_TABLE_MODES:
        return normalize_image(np.arange(256, dtype=np.float32), config)
    if config.mode not in HISTOGRAM_TABLE_MODES:
        raise ValueError(f"no uint8 normalization table for mode {config.mode!r}")
    counts = np.asarray(histogram, dtype=np.int64)
    if counts.shape != (256,):
        raise ValueError(f"expected a 256-bin histogram, got shape {counts.shape}")
    return _percentile_table(counts, *_percentile_bounds(config))


def _percentile_bounds(config: NormalizationConfig) -> tuple[float, float]:
    if config.mode == "minmax":
        return 0.0, 100.0
    return float(config.percentile_lower), float(config.percentile_upper)


def _percentile_table(
    counts: np.ndarray, lower: float, upper: float, fused: frozenset[str] = frozenset()
) -> np.ndarray:
    """The percentile table; ``fused`` names multiply-adds a build may fuse.

    The fused variants exist only to find crops where fusing changes the
    result (_fma_sensitive_crops); the inference table is unfused.
    """
    codes = np.arange(256, dtype=np.float32)
    total = int(counts.sum())
    if total == 0:
        return np.zeros(256, dtype=np.float32)
    cumulative = np.cumsum(counts)

    def order_statistic(rank: int) -> float:
        return float(np.searchsorted(cumulative, rank, side="right"))

    def percentile(percent: float) -> float:
        q = percent / 100
        if "index" in fused:
            position = math.fma(float(total), q, 1 + q * -1.0) - 1
        else:
            position = float(total) * q + (1 + q * -1.0) - 1
        if position >= float(total - 1):
            return order_statistic(total - 1)
        if position < 0:
            return order_statistic(0)
        below = math.floor(position)
        fraction = position - below
        low_value, high_value = order_statistic(below), order_statistic(below + 1)
        span = high_value - low_value
        if fraction >= 0.5:
            if "lerp" in fused:
                return math.fma(-span, 1 - fraction, high_value)
            return high_value - span * (1 - fraction)
        if "lerp" in fused:
            return math.fma(span, fraction, low_value)
        return low_value + span * fraction

    low, high = percentile(lower), percentile(upper)
    scale = high - low
    if not (math.isfinite(scale) and scale >= 1e-6):
        return np.zeros(256, dtype=np.float32)
    low32, high32 = np.float32(low), np.float32(high)
    clamped = np.where(codes < low32, low32, np.where(codes > high32, high32, codes))
    with np.errstate(over="ignore", invalid="ignore"):
        table = ((clamped.astype(np.float64) - low) / scale).astype(np.float32)
    return np.where(np.isfinite(table), table, np.float32(0))


# Percentile pairs where fused arithmetic changes float32 outputs (35.01 and
# 59.01 come from a review counterexample). Fusing is a property of the
# build, so these probe it whatever percentiles a model uses
_FMA_PROBE_BOUNDS = ((1.0, 99.0), (35.01, 59.01))
_FUSED_VARIANTS = (frozenset({"index"}), frozenset({"lerp"}), frozenset({"index", "lerp"}))


def _fma_sensitive_crops(bounds: tuple[float, float]) -> list[np.ndarray]:
    """Two-valued crops whose table changes if vc_reader's multiply-adds fuse.

    vc_reader's virtual index ``n * q + (1 - q) - 1`` and its lerp are plain
    multiply-adds in C++, which a compiler may contract to FMA (ARM64, or x86
    builds with FMA enabled). For each way of fusing them this keeps up to
    two crops, with unequal neighbours at one percentile's ranks, whose fused
    table differs from the unfused one at a code the crop holds.
    """
    if not hasattr(math, "fma"):
        return []
    crops = []
    for fused in _FUSED_VARIANTS:
        found = 0
        for size in range(2, 4096):
            for percent in bounds:
                q = percent / 100
                position = float(size) * q + (1 + q * -1.0) - 1
                if not 0 <= position < size - 1 or position == math.floor(position):
                    continue
                rank = math.floor(position)
                fraction = position - rank
                if "index" in fused:
                    index_differs = position != math.fma(float(size), q, 1 + q * -1.0) - 1
                else:
                    index_differs = False
                for low, high in ((3, 250), (10, 200), (57, 254), (0, 7)):
                    span = float(high - low)
                    if fraction >= 0.5:
                        lerp_differs = high - span * (1 - fraction) != math.fma(-span, 1 - fraction, high)
                    else:
                        lerp_differs = low + span * fraction != math.fma(span, fraction, low)
                    if not (index_differs or ("lerp" in fused and lerp_differs)):
                        continue  # the cheap scalar check: this variant cannot change the table
                    crop = np.r_[np.full(rank + 1, low), np.full(size - rank - 1, high)]
                    crop = crop.astype(np.uint8)
                    counts = np.bincount(crop, minlength=256)
                    plain = _percentile_table(counts, *bounds)[crop]
                    if not np.array_equal(plain, _percentile_table(counts, *bounds, fused)[crop]):
                        crops.append(crop)
                        found += 1
                        break
            if found >= 2:
                break
    return crops


@lru_cache(maxsize=32)
def uint8_table_matches_normalizer(config: NormalizationConfig) -> bool:
    """Check uint8_normalization_table against this vc_reader build.

    The table repeats vc_reader's float64 arithmetic without fused
    multiply-adds. A build that fuses them rounds some percentiles
    differently, so this compares the two on crops where fusing changes the
    result, for this config's percentiles and fixed probe percentiles, plus
    random crops. Callers refuse the table path when it fails.
    """
    if config.mode not in UINT8_TABLE_MODES:
        return False
    rng = np.random.default_rng(0)
    probes = [
        (config, rng.integers(0, 256, size).astype(np.uint8)) for size in (5, 77, 1420, 2762)
    ]
    if config.mode in HISTOGRAM_TABLE_MODES:
        probes += [(config, crop) for crop in _fma_sensitive_crops(_percentile_bounds(config))]
        for lower, upper in _FMA_PROBE_BOUNDS:
            probe = NormalizationConfig.from_value(
                {"mode": "percentile_minmax", "percentile_lower": lower, "percentile_upper": upper}
            )
            probes.append((probe, np.array([0, 7, 200, 255, 0, 7], dtype=np.uint8)))
            probes += [(probe, crop) for crop in _fma_sensitive_crops((lower, upper))]
    for probe, crop in probes:
        table = uint8_normalization_table(np.bincount(crop, minlength=256), probe)
        if not np.array_equal(table[crop], normalize_image(crop.astype(np.float32), probe)):
            return False
    return True


def exclude_validation_voxels(
    supervision: np.ndarray,
    validation: np.ndarray | None,
) -> np.ndarray:
    """Remove held-out voxels from training supervision without mutating input."""
    if validation is None:
        return supervision
    supervision = np.asarray(supervision)
    validation = np.asarray(validation)
    if supervision.shape != validation.shape:
        raise ValueError(
            "supervision and validation must have matching shapes, "
            f"got {supervision.shape!r} and {validation.shape!r}"
        )
    if supervision.size == 0 or not np.any(validation):
        return supervision
    masked = np.array(supervision, copy=True)
    masked[validation > 0] = 0
    return masked

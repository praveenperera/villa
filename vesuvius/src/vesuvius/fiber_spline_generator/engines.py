"""The engines that predict each block's fibers and fit its polylines.

``torch`` runs this package's own ``predict.predict_fibers`` and
``splines.extract_splines``. ``inkrs`` runs the same two steps natively with
the inkrs package: the model on a CUDA GPU with ``inkrs.NnUNet.predict_block``,
and the polylines on the CPU, on every platform, with ``inkrs.fiber_splines``.
Both are drop-ins for this package's functions: the polylines are the ones
this package fits on x86-64, point for point, on every platform (on Apple
silicon SciPy's fused multiply-adds make this package's own differ slightly),
and the probabilities differ only as a different choice of cuDNN algorithm
rounds them. inkrs is imported only when it is asked for.

When inkrs cannot run a step, that step falls back to PyTorch with a
warning, or the run fails with ``EngineUnavailable`` under
``--engine-fallback error``. A step falls back when the inkrs package is
missing, when inkrs raises ``UnsupportedError`` while loading the model
(a model, export or GPU it does not implement) and, for prediction only,
before inkrs is asked: when the device is not a CUDA GPU (macOS among them)
or the volume is not uint8. Any other inkrs error, such as running out of
GPU memory, fails the run.
"""

from __future__ import annotations

import contextlib
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

ENGINES = ("torch", "inkrs")
FALLBACKS = ("torch", "error")
# VC3D passes its environment to the generator, so this selects the engine for runs started from VC3D too
ENGINE_VARIABLE = "VESUVIUS_FIBER_ENGINE"

Progress = Callable[[int, int], None]
SplineProgress = Callable[[str, int, int], None]


class EngineUnavailable(RuntimeError):
    """inkrs was asked for and cannot run a step, and ``--engine-fallback error`` forbids PyTorch instead."""


def default_engine() -> str:
    """The engine named by ``VESUVIUS_FIBER_ENGINE``, else ``torch``; checked by ``check_engine``."""
    return os.environ.get(ENGINE_VARIABLE) or "torch"


def check_engine(engine: str) -> None:
    if engine not in ENGINES:
        raise ValueError(f"Unknown fiber engine {engine!r} (from --engine or {ENGINE_VARIABLE}): choose {' or '.join(ENGINES)}")


@dataclass
class Engine:
    """A block's prediction and polyline steps, and which engine runs each."""

    predict: Callable[..., np.ndarray]
    splines: Callable[..., list[tuple[str, np.ndarray]]]
    # "torch" or "inkrs" for each step, and the inkrs version when it ran one, recorded in the generator metadata
    used: dict[str, str]

    def close(self) -> None:
        """Drops the network, which only ``predict`` holds, so that its device memory can be released."""
        self.predict = _closed


def _closed(*args: Any, **kwargs: Any) -> np.ndarray:
    raise RuntimeError("the engine is closed")


def torch_predict(load_network: Callable[[], tuple[Any, tuple[int, int, int]]], device) -> Callable[..., np.ndarray]:
    """This package's prediction, with the network ``load_network`` loads."""
    from .predict import predict_fibers

    network, patch = load_network()

    def predict(ct: np.ndarray, *, mirror: bool, progress: Progress) -> np.ndarray:
        return predict_fibers(network, ct, patch, device=device, mirror=mirror, progress=progress)

    return predict


def torch_splines(probabilities: np.ndarray, threshold: int, progress: SplineProgress | None) -> list[tuple[str, np.ndarray]]:
    from .splines import extract_splines

    return extract_splines(probabilities, threshold, progress)


def open_engine(
    engine: str,
    fallback: str,
    folder: Path,
    device,
    volume_dtype: np.dtype,
    load_network: Callable[[], tuple[Any, tuple[int, int, int]]],
    warn: Callable[[str], None],
) -> Engine:
    """The engine for a run: ``load_network`` loads this package's network, and runs only when PyTorch predicts."""
    check_engine(engine)
    if fallback not in FALLBACKS:
        raise ValueError(f"Unknown engine fallback {fallback!r}: choose {' or '.join(FALLBACKS)}")
    if engine == "torch":
        return Engine(torch_predict(load_network, device), torch_splines, {"predict": "torch", "splines": "torch"})

    def unavailable(step: str, reason: str) -> None:
        if fallback == "error":
            raise EngineUnavailable(f"inkrs cannot {step}: {reason}")
        warn(f"inkrs cannot {step}, so PyTorch does: {reason}")

    try:
        import inkrs
    except ImportError as error:
        unavailable("run", f"the inkrs package does not import ({error})")
        return open_engine("torch", fallback, folder, device, volume_dtype, load_network, warn)

    # villa's own options, checked before inkrs is asked, so that no export or GPU work starts for them
    reason = None
    if device.type != "cuda":
        reason = f"it predicts on CUDA GPUs only, and the device is {device}"
    elif np.dtype(volume_dtype) != np.uint8:
        reason = f"it reads uint8 CT only, and the volume is {np.dtype(volume_dtype)}"
    network = None
    if reason is None:
        try:
            # the first run of a model folder exports it, and the exporter prints to stdout, which carries the
            # JSON progress events
            with contextlib.redirect_stdout(sys.stderr):
                network = inkrs.NnUNet(folder, device.index if device.index is not None else _current_cuda_device())
        except inkrs.UnsupportedError as error:
            reason = str(error)
    if network is None:
        unavailable("predict fibers", reason)
        return Engine(torch_predict(load_network, device), inkrs.fiber_splines,
                      {"predict": "torch", "splines": "inkrs", "inkrs": inkrs.__version__})

    def inkrs_predict(ct: np.ndarray, *, mirror: bool, progress: Progress) -> np.ndarray:
        probabilities = network.predict_block(ct, mirror=mirror)
        # the engine predicts a block in one call, with no progress of its own
        progress(1, 1)
        return probabilities

    return Engine(inkrs_predict, inkrs.fiber_splines, {"predict": "inkrs", "splines": "inkrs", "inkrs": inkrs.__version__})


def _current_cuda_device() -> int:
    import torch

    return torch.cuda.current_device()

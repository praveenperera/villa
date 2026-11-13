"""Optional inkrs engine for `vesuvius.predict --blend_in_memory`.

The default engine is villa's PyTorch path. With `--engine inkrs`, the whole run (sliding window, mirroring TTA,
Gaussian blend and binary finalisation) goes to inkrs's native nnU-Net engine, which writes the same
`(1, Z, Y, X)` uint8 array the in-memory blender writes. villa keeps the argument parsing, the threshold rules
and the model location. When inkrs cannot run a request, villa logs why and runs it with PyTorch, unless
`--engine_fallback error` (or `--engine-fallback error`) is given.

villa's own options are checked here, before inkrs is imported, so a request inkrs never sees falls back without
loading it. inkrs then checks the model, the volume and the GPU itself and raises `inkrs.UnsupportedError` for
anything it does not implement, which falls back too. Options that change only how villa schedules or stores
intermediate work (`--batch_size`, `--num_workers`, `--writer_workers`, `--chunk_cache_mb`, `--zarr_compressor`,
`--zarr_compression_level`, `--output_dir`) have no inkrs counterpart and are ignored, as is `--normalization`,
which nnU-Net models replace with their plans' own.
"""

from __future__ import annotations

import importlib
import logging
import os
import re
import time
from pathlib import Path

LOGGER = logging.getLogger(__name__)

#: the first inkrs release with the nnU-Net engine (`inkrs.nnunet_predict`)
MIN_VERSION = "0.4.0"
#: a URL such as s3://, http:// or file://; inkrs reads and writes local paths only
URL = re.compile(r'^[a-z][a-z0-9+.-]*://')


class EngineUnavailable(RuntimeError):
    """`--engine inkrs` cannot run this request and `--engine_fallback error` was given."""


def add_engine_arguments(parser):
    """Register `--engine` and `--engine_fallback` on `vesuvius.predict`'s parser."""
    parser.add_argument('--engine', choices=['torch', 'inkrs'], default='torch',
                        help="Inference engine. 'torch' (default) is the PyTorch path. 'inkrs' uses the optional "
                             f"inkrs package ({MIN_VERSION} or later) for nnU-Net models on a CUDA GPU with "
                             '--blend_in_memory --mode binary; anything it cannot run falls back to PyTorch')
    parser.add_argument('--engine_fallback', choices=['torch', 'error'], default='torch',
                        help='What to do when --engine cannot run a request: continue with PyTorch and log why '
                             '(default), or stop with an error')


def cuda_index(device):
    """The CUDA ordinal of villa's `--device`, or None when it names no CUDA device.

    No `--device` and a bare `cuda` mean PyTorch's current CUDA device, which villa would also pick, or None
    when PyTorch sees no CUDA device (villa then runs on MPS or the CPU).
    """
    if device is None or device == 'cuda':
        import torch
        return torch.cuda.current_device() if torch.cuda.is_available() else None
    m = re.fullmatch(r'cuda:(\d+)', str(device))
    return int(m.group(1)) if m else None


def local_zarr_v2(path):
    """Whether `path` is what inkrs opens: a local zarr v2 array, or a zarr v2 group with an array `0`.

    inkrs reports any other input (a zarr v3 store, a TIFF stack) as an error rather than as unsupported, so villa
    checks it here to fall back instead of stopping.
    """
    p = Path(path)
    return (p / '.zarray').is_file() or ((p / '.zgroup').is_file() and (p / '0' / '.zarray').is_file())


def refusal(args):
    """Why inkrs cannot reproduce this `vesuvius.predict` request, or None; checked before inkrs is imported."""
    if not args.blend_in_memory:
        return 'it writes the blended output only; pass --blend_in_memory'
    if args.mode != 'binary':
        return f'it implements --mode binary, got {args.mode}'
    if args.num_parts != 1:
        return f'it runs one part, got --num_parts {args.num_parts}'
    if not args.disable_tta and args.tta_type != 'mirroring':
        return f'it implements mirroring TTA, got --tta_type {args.tta_type}'
    if args.patch_size:
        return "it runs the model's own patch size; drop --patch_size"
    if args.save_softmax:
        return 'it writes no logits; drop --save_softmax'
    if not args.skip_empty_patches:
        return 'it always skips constant patches, as villa does by default; drop --no_skip_empty_patches'
    if args.max_patches is not None:
        return 'it runs every patch; drop --max_patches'
    if args.input_anon or URL.match(args.input_dir):
        return f'it reads local zarr only, got {args.input_dir}'
    if not local_zarr_v2(args.input_dir):
        return f'it reads a zarr v2 array, or a zarr v2 group with array 0, got {args.input_dir}'
    if URL.match(args.blend_in_memory):
        return f'it writes local zarr only, got --blend_in_memory {args.blend_in_memory}'
    if args.model_type not in ('auto', 'nnunet'):
        return f'it runs nnU-Net models, got --model_type {args.model_type}'
    if str(args.model_path).endswith('.pth'):
        return 'it runs nnU-Net model folders, and a .pth checkpoint is a train.py model'
    if cuda_index(args.device) is None:
        return f"it runs on a CUDA GPU, got --device {args.device or 'none (no CUDA device found)'}"
    if args.blend_sigma_scale != 8.0:
        return f"it blends with villa's default sigma scale 8.0, got --blend_sigma_scale {args.blend_sigma_scale}"
    if args.blend_compression_level != 1:
        return f'it writes compression level 1, got --blend_compression_level {args.blend_compression_level}'
    if args.blend_chunk_size and len(set(args.blend_chunk_size.split(','))) != 1:
        return f'it writes cubic chunks, got --blend_chunk_size {args.blend_chunk_size}'
    return None


def local_model_dir(args):
    """The local nnU-Net model folder villa would load: hf:// repositories are downloaded with huggingface_hub
    (villa's own loader does the same), s3:// paths go through villa's model cache."""
    path = args.model_path
    if isinstance(path, str) and path.startswith('hf://'):
        from huggingface_hub import snapshot_download
        return Path(snapshot_download(repo_id=path[len('hf://'):], token=args.hf_token))
    from vesuvius.models.run.inference import _resolve_model_path
    return Path(_resolve_model_path(path, args.model_cache_dir, verbose=args.verbose))


def fall_back(reason, *, fallback):
    """Log that PyTorch runs instead, or raise under `--engine_fallback error`."""
    if fallback == 'error':
        raise EngineUnavailable(reason)
    LOGGER.warning('%s. Running it with PyTorch.', reason)


def import_inkrs():
    """The inkrs module, or the reason it cannot run nnU-Net models."""
    try:
        inkrs = importlib.import_module('inkrs')
    except ImportError as e:
        return None, f'the inkrs package is not installed ({e}); install inkrs {MIN_VERSION} or later'
    # inkrs before 0.4.0, installed for its ink models, imports but has no nnU-Net engine
    if not all(hasattr(inkrs, name) for name in ('nnunet_predict', 'UnsupportedError')):
        version = getattr(inkrs, '__version__', 'of unknown version')
        return None, f'inkrs {version} has no nnU-Net engine, which needs inkrs {MIN_VERSION} or later'
    return inkrs, None


def predict(args, threshold):
    """Run the request on inkrs; returns the output path, or None when PyTorch should run instead.

    `threshold` is villa's resolved cutoff (None for none). Raises EngineUnavailable under
    `--engine_fallback error`; any inkrs failure other than UnsupportedError is raised as is.
    """
    def decline(reason):
        fall_back(f'The inkrs engine cannot run this command: {reason}', fallback=args.engine_fallback)

    reason = refusal(args)
    if reason is not None:
        return decline(reason)
    inkrs, reason = import_inkrs()
    if reason is not None:
        return decline(reason)

    chunk = int(args.blend_chunk_size.split(',')[0]) if args.blend_chunk_size else None
    try:
        model = local_model_dir(args)
        t0 = time.perf_counter()
        report = inkrs.nnunet_predict(
            model, args.input_dir, args.blend_in_memory,
            device=cuda_index(args.device), bbox=args.bbox, overlap=args.overlap, tta=not args.disable_tta,
            threshold=threshold, chunk=chunk, overwrite=True,
        )
    except inkrs.UnsupportedError as e:
        return decline(f'inkrs does not support it: {e}')
    print(f"inkrs {getattr(inkrs, '__version__', '?')}: {report['patches']} patches "
          f"({report['patches_skipped']} empty) x {report['flips']} flips, "
          f"{report['chunks_written']} chunks written in {time.perf_counter() - t0:.1f} s")
    return os.fspath(args.blend_in_memory)

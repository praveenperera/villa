"""Opt-in inkrs engine for `vesuvius.predict --blend_in_memory`.

`--engine inkrs` hands the whole run (sliding window, mirroring TTA, Gaussian blend and binary finalisation) to
inkrs's native nnU-Net engine, which writes the same `(1, Z, Y, X)` uint8 array the in-memory blender writes.
villa keeps the argument parsing, the threshold rules and the model location; inkrs runs only what it
implements and raises `inkrs.UnsupportedError` for anything else, and villa then runs PyTorch as usual
(`--engine-fallback torch`, the default) or stops (`--engine-fallback error`).

Options inkrs never sees are checked here, before inkrs is imported, so a run inkrs cannot reproduce falls back
without loading it. Options that change only how villa schedules work (`--batch_size`, `--num_workers`,
`--writer_workers`, `--chunk_cache_mb`, `--zarr_compressor`) have no inkrs counterpart and are ignored.
"""

from __future__ import annotations

import os
import re
import time
from pathlib import Path


class EngineUnavailable(RuntimeError):
    """`--engine inkrs` could not run this request and `--engine-fallback error` was given."""


def add_engine_arguments(parser):
    """Register `--engine` and `--engine-fallback` on `vesuvius.predict`'s parser."""
    parser.add_argument('--engine', choices=['torch', 'inkrs'], default='torch',
                        help='Inference engine. inkrs runs nnU-Net models natively on CUDA with '
                             '--blend_in_memory --mode binary; anything it does not implement falls back '
                             'to PyTorch. Default: torch')
    parser.add_argument('--engine_fallback', choices=['torch', 'error'], default='torch',
                        help='What --engine inkrs does with a request it cannot run: run PyTorch instead '
                             '(torch, default) or stop with an error (error)')


def cuda_index(device):
    """The CUDA ordinal of villa's `--device` (None picks cuda first, as villa does), or None if not CUDA."""
    if device is None or device == 'cuda':
        return 0
    m = re.fullmatch(r'cuda:(\d+)', str(device))
    return int(m.group(1)) if m else None


def refusal(args):
    """Why inkrs cannot reproduce this `vesuvius.predict` request, or None; checked before inkrs is imported."""
    if not args.blend_in_memory:
        return 'inkrs writes the blended output only; pass --blend_in_memory'
    if args.mode != 'binary':
        return f'--mode {args.mode}: inkrs implements binary'
    if args.num_parts != 1:
        return f'--num_parts {args.num_parts}: inkrs runs one part'
    if not args.disable_tta and args.tta_type != 'mirroring':
        return f'--tta_type {args.tta_type}: inkrs implements mirroring'
    if args.patch_size:
        return '--patch_size: inkrs runs the model\'s own patch size'
    if args.save_softmax:
        return '--save_softmax: inkrs writes no logits'
    if not args.skip_empty_patches:
        return '--no_skip_empty_patches: inkrs always skips constant patches, as villa does by default'
    if args.max_patches is not None:
        return '--max_patches: inkrs runs every patch'
    if args.input_anon or re.match(r'^[a-z][a-z0-9+.-]*://', args.input_dir):
        return f'input {args.input_dir}: inkrs reads local zarr only'
    if args.model_type not in ('auto', 'nnunet'):
        return f'--model_type {args.model_type}: inkrs runs nnU-Net models'
    if str(args.model_path).endswith('.pth'):
        return 'a .pth checkpoint is a train.py model; inkrs runs nnU-Net model folders'
    if cuda_index(args.device) is None:
        return f'--device {args.device}: inkrs runs on CUDA'
    if args.blend_sigma_scale != 8.0:
        return f'--blend_sigma_scale {args.blend_sigma_scale}: inkrs blends with villa\'s default 8.0'
    if args.blend_compression_level != 1:
        return f'--blend_compression_level {args.blend_compression_level}: inkrs writes level 1'
    if args.blend_chunk_size:
        edges = set(args.blend_chunk_size.split(','))
        if len(edges) != 1:
            return f'--blend_chunk_size {args.blend_chunk_size}: inkrs writes cubic chunks'
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


def predict(args, threshold):
    """Run the request on inkrs; returns the output path, or None when PyTorch should run instead.

    `threshold` is villa's resolved cutoff (None for none). Raises EngineUnavailable under
    `--engine-fallback error`; any inkrs failure other than UnsupportedError is raised as is.
    """
    def decline(reason):
        if args.engine_fallback == 'error':
            raise EngineUnavailable(f'--engine inkrs: {reason}')
        print(f'--engine inkrs: {reason}; running PyTorch instead')
        return None

    reason = refusal(args)
    if reason is not None:
        return decline(reason)
    try:
        import inkrs
    except ImportError as e:
        return decline(f'inkrs is not installed ({e})')

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
        return decline(str(e))
    print(f"inkrs: {report['patches']} patches ({report['patches_skipped']} empty) x {report['flips']} flips, "
          f"{report['chunks_written']} chunks written in {time.perf_counter() - t0:.1f} s")
    return os.fspath(args.blend_in_memory)

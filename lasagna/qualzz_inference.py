"""AFV V2 inference using the shared Lasagna tile engine and plain Zarr v2."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import itertools
import json
import math
import os
from pathlib import Path
import shutil
import time

import numpy as np
import torch
from vc_reader.reader import create_group, open_zarr

from lasagna.compact_zarr import _chunk_index, _create_array, _writer, atomic_chunk
from lasagna.cooperative_inference import cooperative_signals
from lasagna.input_level import InputLevel
from lasagna.tiled_predict3d import (
    OutputProductSpec, _canonical_local_tile_positions, _cleanup_predict3d_temp_files,
    _plan_circular_z_depth, omezarr_compressor_config, resolve_inference_devices,
    run_tiled_inference_3d,
)

# Default memory for one XY block's float32 accumulator rings. A block's rings
# hold only the Z planes the engine keeps live, so the budget bounds them
# whatever the array's size.
DEFAULT_ACCUMULATOR_GIB = 4.0
SHARED_MEMORY_DIR = Path("/dev/shm")


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class _NormalizedModel(torch.nn.Module):
    def __init__(self, model, normalize):
        super().__init__()
        self.model, self.normalize = model, normalize

    def forward(self, tile):
        # The shared reader supplies uint8-scale CT divided by 255.
        # Delegate normalization to the checkpoint's own inference helper.
        return self.model(self.normalize((tile.float() * 255.0).round()))


class QualzzModelAdapter:
    """Use reviewed local HF model.py, not a duplicated architecture/normalizer."""
    native_output_stride = 1

    def __init__(self, model_dir, chunk_size=128, *, model_hashes=None):
        self.model_dir = str(Path(model_dir).resolve())
        self.model_hashes = model_hashes
        self.output_products = (OutputProductSpec(
            name="semantic", level=0, scaledown=1, inference_scaledown=1,
            channels=("vertical", "horizontal"), chunk_size=chunk_size,
        ),)

    def load_model(self, *, device):
        for name, expected in (self.model_hashes or {}).items():
            if _sha256(Path(self.model_dir) / name) != expected:
                raise ValueError(f"model file changed during inference: {name}")
        path = Path(self.model_dir) / "model.py"
        spec = importlib.util.spec_from_file_location("qualzz_afv_model", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return _NormalizedModel(module.load_model(self.model_dir, device=device),
                                module.normalize_ct).eval()

    def run_tile_inference(self, model, tile, *, device):
        # FP32, matching the published helper; no implicit precision change.
        return model(tile)

    def product_tensors_from_output(self, output):
        logits = output["seg"]
        if logits.ndim != 5 or logits.shape[1] != 4:
            raise ValueError("AFV V2 requires B,4,Z,Y,X segmentation logits (BG,V,H,I)")
        return {"semantic": logits.float().softmax(dim=1)[:, 1:3]}

    def finalize_product_slab(self, product, raw_slab):
        # Keep averaged probabilities floating point until BOTH output scales
        # have been formed. Quantizing here would bias the tracing downsample.
        return dict(zip(product.channel_names, raw_slab))


def _quantize(probabilities):
    if not np.isfinite(probabilities).all():
        raise ValueError("non-finite AFV probabilities")
    return np.rint(np.clip(probabilities, 0, 1) * 255).astype(np.uint8)


def _pool2(probabilities):
    """Match the generator's ordered eight additions, dropping odd end planes."""
    shape = tuple(n // 2 for n in probabilities.shape[1:])
    pooled = np.zeros((2, *shape), np.float32)
    for z, y, x in itertools.product((0, 1), repeat=3):
        pooled += probabilities[:, z:shape[0]*2:2, y:shape[1]*2:2, x:shape[2]*2:2]
    return pooled / 8


class ProbabilityZarrOutputAdapter:
    """Atomic paired-scale publication without any Lasagna schema metadata."""
    def __init__(self, root, product, shape, *, voxel_size_um, provenance):
        self.root = str(root)
        self.product = product
        self.shape = tuple(shape)
        self.voxel_size_um = float(voxel_size_um)
        self.provenance = provenance
        if product.chunk_size < 2 or product.chunk_size % 2 or min(shape) < 2:
            raise ValueError("paired output needs even chunk size >=2 and spatial shape >=2")

    def prepare(self):
        root = Path(self.root)
        metadata = dict(channel_order="VH", axes=["c", "z", "y", "x"],
                        voxel_size_um=self.voxel_size_um, inference=self.provenance)
        if root.exists() and any(root.iterdir()):
            attrs = root / ".zattrs"
            if not attrs.exists() or json.loads(attrs.read_text()) != metadata:
                raise ValueError("output provenance differs; use a new output directory")
        group = create_group(root)
        codec = omezarr_compressor_config("blosc-zstd")
        for name, factor in (("semantic", 1), ("tracing", 2)):
            shape = tuple(n // factor for n in self.shape)
            chunk = self.product.chunk_size // factor
            _create_array(group, name, (2, *shape), (2, chunk, chunk, chunk),
                          np.uint8, codec)
        group.attrs.update(metadata)

    def _stored(self, name, origin):
        path = str(Path(self.root) / name)
        return _writer(path).chunk_exists(_chunk_index(path, origin))

    def product_chunk_complete(self, product, *, chunk_origin_zyx):
        origin = tuple(chunk_origin_zyx)
        native = self._stored("semantic", origin)
        if any(o // 2 >= n // 2 for o, n in zip(origin, self.shape)):
            return native  # Last unpaired plane has no tracing output.
        return native and self._stored("tracing", tuple(o//2 for o in origin))

    def write_product_chunk(self, product, *, chunk_origin_zyx, data):
        values = np.stack([data[name] for name in product.channel_names]).astype(np.float32)
        coarse = _pool2(values)
        # Native chunk is published last. An interruption between writes leaves
        # the pair incomplete and the common resume scheduler recomputes it.
        if all(n > 0 for n in coarse.shape[1:]):
            atomic_chunk(Path(self.root) / "tracing", tuple(o//2 for o in chunk_origin_zyx),
                         _quantize(coarse))
        atomic_chunk(Path(self.root) / "semantic", chunk_origin_zyx, _quantize(values))

    def update_metadata(self, products):
        pass


def _ring_bytes(nz, ny, nx, *, tile_size, overlap, border, chunk_size, channels, flush_workers):
    """Bytes of one engine call's float32 product and weight rings, sized as the engine does."""
    zp, yp, xp = (n + 2 * border for n in (nz, ny, nx))
    positions = _canonical_local_tile_positions(
        volume_size=nz, crop_start=0, crop_padded_size=zp, tile_size=tile_size,
        stride=tile_size - overlap, border=border, scaledown_multiple=1)
    depth = _plan_circular_z_depth(
        z_positions=positions, tile_size=tile_size, scaledown=1, z_size=zp,
        chunk_size=chunk_size, output_begin=border, output_end=border + nz,
        retain_one_flush=flush_workers > 0)
    bricks = math.prod(-(-n // chunk_size) for n in (depth, yp, xp))
    return (channels + 1) * bricks * chunk_size ** 3 * 4


def _block_for_budget(budget_bytes, shape, *, tile_size, overlap, border, chunk_size, channels,
                      flush_workers):
    """Largest square XY block side whose rings fit ``budget_bytes``.

    Sides are multiples of the chunk size, which keeps every output chunk
    inside one block, and preferably of the tile stride too: with overlap =
    2 x border, tile cores then tile the array and block edges fall on core
    edges, so only the last tile row of an axis, shifted to end at the
    array's edge, can reach two blocks and run twice.
    """
    nz, ny, nx = shape
    def ring(side):
        return _ring_bytes(nz, min(ny, side), min(nx, side), tile_size=tile_size, overlap=overlap,
                           border=border, chunk_size=chunk_size, channels=channels,
                           flush_workers=flush_workers)
    aligned = math.lcm(chunk_size, tile_size - overlap)
    step = aligned if ring(aligned) <= budget_bytes else chunk_size
    if ring(step) > budget_bytes:
        raise ValueError(f"accumulator budget too small: one {step}-voxel block needs "
                         f"{ring(step) / 2**30:.2f} GiB")
    side = step
    while side < max(ny, nx) and ring(side + step) <= budget_bytes:
        side += step
    return side


def _tile_slot_bytes(device_count, *, tile_size, channels):
    """Shared-memory tile buffers of the engine's multi-device path.

    The engine keeps two slots per device (its ``slots_per_gpu`` default), each
    holding a uint8 input tile and the float32 product and weight tiles.
    """
    tile = tile_size ** 3
    return int(device_count) * 2 * (tile + (channels + 1) * tile * 4)


def _first_symlinked_dir(root):
    """The first directory symlink under ``root``, or None.

    The stored-chunk listing does not descend into symlinked directories, so
    the CT behind one would read as absent and its tiles would not run.
    """
    for parent, dirs, files in os.walk(root):
        for name in dirs:
            if os.path.islink(os.path.join(parent, name)):
                return Path(parent, name)
    return None


def _block_has_work(writer, product, input_level, shape, block, chunk_size):
    """Whether an output chunk in the block is incomplete and has stored CT.

    The engine leaves chunks without stored CT absent and skips completed ones,
    so a block with neither kind of chunk would start its workers for nothing.
    """
    nz, ny, nx = shape
    y0, y1, x0, x1 = block
    for z, y, x in itertools.product(range(0, nz, chunk_size), range(y0, y1, chunk_size),
                                     range(x0, x1, chunk_size)):
        if writer.product_chunk_complete(product, chunk_origin_zyx=(z, y, x)):
            continue
        footprint = (z, min(nz, z + chunk_size), y, min(ny, y + chunk_size), x, min(nx, x + chunk_size))
        if input_level.region_has_chunks(footprint):
            return True
    return False


def _scratch_dir(requested, output, needed_bytes):
    """Shared memory when one block's rings fit there; else beside the output.

    File-backed rings on a disk are written back as they are dirtied, which
    costs more than the tiles themselves on large arrays. A tmpfs that fills
    up faults the writer with SIGBUS, hence the free-space margin.
    """
    if requested is not None:
        return Path(requested).resolve()
    if SHARED_MEMORY_DIR.is_dir() and shutil.disk_usage(SHARED_MEMORY_DIR).free >= needed_bytes * 5 // 4:
        return SHARED_MEMORY_DIR
    return output.parent


@cooperative_signals
def run_inference(*, model_dir, input_path, output, voxel_size_um, devices,
                  tile_size=128, overlap=32, border=16, chunk_size=128,
                  flush_workers=8, accumulator_workers=8, accumulator_gib=DEFAULT_ACCUMULATOR_GIB,
                  block_size=None, scratch_dir=None):
    if not np.isfinite(voxel_size_um) or not 8.64 <= voxel_size_um <= 9.6:
        raise ValueError("AFV V2 expects input voxels between 8.64 and 9.6 micrometres")
    if tile_size < 128 or tile_size % 64 or not 0 <= 2*border <= overlap < tile_size:
        raise ValueError("tile size must be >=128 and divisible by64; require 0<=2*border<=overlap<tile")
    if flush_workers < 0 or accumulator_workers < 0:
        raise ValueError("worker counts must be nonnegative")
    if block_size is not None and (block_size < 0 or block_size % chunk_size):
        raise ValueError("block size must be 0 (one pass) or a positive multiple of the chunk size")
    if block_size is None and not (np.isfinite(accumulator_gib) and accumulator_gib > 0):
        raise ValueError("accumulator budget must be a positive number of GiB")
    model_dir, input_path = Path(model_dir).resolve(), Path(input_path).resolve()
    hashes = {name: _sha256(model_dir / name)
              for name in ("model.py", "plans.json", "checkpoint_final.pth")}
    source = open_zarr(input_path)
    if len(source.shape) != 3 or np.dtype(source.dtype) != np.uint8:
        raise ValueError("AFV inference expects a 3D uint8 CT Zarr array")
    shape = tuple(int(n) for n in source.shape)
    adapter = QualzzModelAdapter(model_dir, chunk_size, model_hashes=hashes)
    product, = adapter.output_products
    output = Path(output).resolve()
    if output == input_path or input_path in output.parents or output in input_path.parents:
        raise ValueError("output and input stores must not contain one another")
    writer = ProbabilityZarrOutputAdapter(output, product, shape, voxel_size_um=voxel_size_um,
        provenance=dict(model_sha256=hashes, input=str(input_path), input_shape=list(shape),
                        tile_size=tile_size, overlap=overlap, border=border,
                        precision="fp32", accumulator_dtype="float32",
                        normalization="model.py:normalize_ct per tile",
                        # version 2 leaves chunks without stored CT absent; version 1 predicted
                        # them, so resuming a version 1 output would keep those predictions
                        version=2))
    writer.prepare()
    if any(d.type == "cuda" for d in devices) and os.environ.get("LASAGNA_COOPERATIVE") != "1":
        raise ValueError("CUDA inference requires GPU-manager handling (LASAGNA_COOPERATIVE=1)")
    model = adapter.load_model(device=devices[0]) if devices[0].type == "cpu" else None
    nz, ny, nx = shape
    rings = dict(tile_size=tile_size, overlap=overlap, border=border, chunk_size=chunk_size,
                 channels=product.raw_channel_count, flush_workers=flush_workers)
    if block_size is None:
        block_size = _block_for_budget(int(accumulator_gib * 2**30), shape, **rings)
    # Each block is one engine pass over the full Z depth. The engine places
    # tiles on the array's lattice and reads their halos from the array, so a
    # voxel gets the same tiles, weights and add order as in one whole pass.
    by, bx = block_size or ny, block_size or nx
    blocks = [(y0, min(ny, y0 + by), x0, min(nx, x0 + bx))
              for y0 in range(0, ny, by) for x0 in range(0, nx, bx)]
    needed = _ring_bytes(nz, min(ny, by), min(nx, bx), **rings)
    tile_slots = _tile_slot_bytes(len(devices), tile_size=tile_size, channels=product.raw_channel_count)
    scratch = _scratch_dir(scratch_dir, output, needed + tile_slots)
    scratch.mkdir(parents=True, exist_ok=True)
    _cleanup_predict3d_temp_files(scratch)
    print(f"[qualzz] {len(blocks)} block(s) of up to {min(ny, by)}x{min(nx, bx)}; "
          f"float32 rings up to {needed / 2**30:.2f} GiB in {scratch}", flush=True)
    # Tiles without stored CT are skipped, so CT the listing cannot see would
    # silently leave its output absent
    linked = _first_symlinked_dir(input_path)
    if linked is not None:
        raise ValueError(f"input has a symlinked directory, which the chunk listing does not follow: {linked}; "
                         "link chunk files instead")
    with InputLevel(input_path) as input_level:
        if not input_level.region_has_chunks((0, nz, 0, ny, 0, nx)):
            raise ValueError(f"input lists no stored chunks: {input_path}")
        for index, (y0, y1, x0, x1) in enumerate(blocks):
            if not _block_has_work(writer, product, input_level, shape, (y0, y1, x0, x1), chunk_size):
                continue
            print(f"[qualzz] block {index + 1}/{len(blocks)} y=[{y0},{y1}) x=[{x0},{x1})", flush=True)
            run_tiled_inference_3d(model, source, crop_slices=(0,nz,y0,y1,x0,x1), device=devices[0],
                model_adapter=adapter, output_adapter=writer, products=(product,),
                output_regions_zyx={product.name: (0,y0,x0,nz,y1,x1)},
                full_output_shapes_zyx={product.name: shape},
                output_scaledown_base=1, tile_size=tile_size, overlap=overlap, border=border,
                tmp_dir=str(scratch), temp_prefix=output.stem + "_", devices=devices,
                progress=dict(t0=time.time(), finalized_base_z=0, finalized_base_z_total=nz),
                flush_workers=flush_workers, accumulator_workers=accumulator_workers,
                product_accumulator_dtype="float32", input_level=input_level)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, help="Reviewed HF snapshot folder; executes its model.py")
    parser.add_argument("--input", required=True, help="Local uint8 CT Zarr array at model resolution")
    parser.add_argument("--output", required=True, help="Standard Zarr v2 output group")
    parser.add_argument("--voxel-size-um", type=float, required=True)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--device", default=None)
    selection.add_argument("--devices", default=None)
    parser.add_argument("--tile-size", type=int, default=128)
    parser.add_argument("--overlap", type=int, default=32)
    parser.add_argument("--border", type=int, default=16)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--flush-workers", type=int, default=8)
    parser.add_argument("--accumulator-workers", type=int, default=8)
    parser.add_argument("--accumulator-gib", type=float, default=DEFAULT_ACCUMULATOR_GIB,
                        help="Memory for one XY block's float32 accumulator rings; picks the largest "
                             f"block that fits. Default {DEFAULT_ACCUMULATOR_GIB:g}")
    parser.add_argument("--block-size", type=int, default=None,
                        help="XY block side in voxels instead of --accumulator-gib, a multiple of "
                             "--chunk-size; 0 runs one pass over the whole array")
    parser.add_argument("--scratch-dir", default=None,
                        help="Directory for the float32 accumulator rings. Default: /dev/shm when "
                             "one block's rings fit there, else the output's parent")
    args = vars(parser.parse_args(argv))
    args["devices"] = resolve_inference_devices(device=args.pop("device"), devices=args["devices"])
    args["input_path"] = args.pop("input")
    os.environ["LASAGNA_COOPERATIVE"] = "1"
    run_inference(**args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

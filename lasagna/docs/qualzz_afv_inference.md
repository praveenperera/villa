# Qualzz AFV V2 volume inference

`python -m lasagna.qualzz_inference` adapts the published
[Qualzz20/afv_fiber_9um_v2](https://huggingface.co/Qualzz20/afv_fiber_9um_v2)
model to Lasagna's existing tiled inference engine. It reuses its incremental GPU
manager allocation, readers, probability blending, accumulators, parallel flush
and chunk-based resume. It does not change existing Fiber3D/Lasagna inference.

## Model and input

Provide a **reviewed local snapshot directory** containing `model.py`, `plans.json`
and `checkpoint_final.pth`. The adapter executes that local `model.py` and calls
its `load_model` and `normalize_ct` functions, rather than duplicating the model.
Do not use untrusted model Python files. The supported contract is V2's dictionary
output with `seg` logits ordered background, vertical, horizontal, intersection.
The upstream `dynamic-network-architectures` dependency must be installed in the
inference environment (optional Lasagna extra `qualzz`). Nothing is downloaded
or installed automatically.

Input is a local **uint8, ZYX Zarr array at 8.64–9.6 µm**, already downsampled.
For 2.4 µm Paris4, select level `2` (9.6 µm). This command does not fetch missing
CT chunks; populate/verify the input cache first. Keep input files immutable.

Default tiles are 128³ (the published training patch size), overlap 32, border
16. The upstream per-tile z-score normalization, including its standard-deviation
floor, is used on uint8-scale values. This is not global-volume normalization.
Inference and accumulation are FP32. Four-class softmax is applied before tile
blending; only V/H probabilities are retained. Auxiliary heads are not exported.

## Run

```bash
python -m lasagna.qualzz_inference \
  --model-dir "$MODEL" --input "$CT_LEVEL2" \
  --output "$OUTPUT/predictions.zarr" --voxel-size-um 9.6 \
  --devices cuda:0,cuda:1,cuda:2,cuda:3,cuda:4,cuda:5
```

CUDA always uses the GPU manager. The usual `LASAGNA_GPU_MEMORY_GIB`,
`LASAGNA_GPU_PRIORITY`, `VILLA_GPU_MANAGER_SOCKET` and `CUDA_VISIBLE_DEVICES`
settings apply. Without a measured memory budget, workers request whole GPUs.
CPU execution is available with `--device cpu` for tests. This is a foreground
inference CLI, not a new `las_manager` catalog model type.

## Memory and empty regions

The array is predicted in XY blocks, each one engine pass over the full Z depth.
Tiles sit on the whole array's lattice and read their halos from the array, so
the output is the same as one pass over the whole array. A block's float32
accumulator rings are bounded by `--accumulator-gib` (default 4), whatever the
array's size: the command picks the largest square block whose rings fit. Block
sides are multiples of the chunk size and, where the budget allows, of the tile
stride, so tiles do not run twice, except perhaps the last tile row of an axis,
which is shifted to end at the array's edge. With the default tiles and chunks,
sides of 384, 768 and 1536 voxels need about 1.1, 3.5 and 11.9 GiB. Each block
starts the engine again, so very small blocks cost start-up time. `--block-size`
sets the side directly (`0` runs one pass).

The rings live in `/dev/shm` when one block's rings and the engine's tile
buffers fit there with a quarter to spare, else beside the output store; `--scratch-dir` chooses another directory.
Shared memory counts as used RAM. Rings in a disk directory are written back as
the page cache fills, which can cost more time than the tiles on a large array.

Tiles whose input has no stored CT chunk are not run, and output chunks whose own
CT chunks are all absent are left absent, so they read as zero. Blocks whose
output chunks are all complete or without stored CT are skipped without starting
the engine. Stored chunks are found from one listing of the input store, which
does not descend into symlinked directories, so the command refuses an input with
a symlinked directory (symlinked chunk files are fine) and stops if the listing
finds no chunk at all.

Repeat the same command to resume completed chunk pairs. Outputs written before
chunks without CT were left absent record an older provenance version and are
refused; use a new output directory. Checkpoint, model code,
plans, input path/shape and numerical settings are recorded and checked on resume.
Input chunk contents are not hashed; do not replace them during a run/resume.

## Standard output, no Lasagna manifest

The Zarr v2 group contains only two arrays plus descriptive root attributes:

| Array | Shape | Chunks | Meaning |
| --- | --- | --- | --- |
| `semantic` | `(2,Z,Y,X)` | `(2,128,128,128)` | V/H probabilities at input resolution |
| `tracing` | `(2,Z//2,Y//2,X//2)` | `(2,64,64,64)` | 2× mean-pooled V/H probabilities |

Both are uint8 (0–255), Blosc/Zstd compressed. Chunk size is configurable with
`--chunk-size` (even). All channels share each spatial chunk. There is no
`lasagna` attribute, sidecar manifest, or additional pyramid. The tracing array
is pooled from **blended floating-point probabilities before quantization**;
odd final planes are omitted, matching the generator. Explicit compressed zero
chunks record completed empty predictions; chunks with no stored CT remain
absent. Paired publication ensures an interrupted pair is recomputed on resume.

Feed the result directly to the unmodified updated generator:

```bash
avf-spline-generator \
  --predictions "$OUTPUT/predictions.zarr" --array semantic \
  --tracing-predictions "$OUTPUT/predictions.zarr" --tracing-array tracing \
  --voxel-size-um 9.6 --storage compressed --field-cache-mib 16384 \
  --output "$OUTPUT/fibers.afv" --work-dir "$OUTPUT/tracing-work"
```

The generator may use a separate environment: its compressed backend requires
Zarr 2.x, whereas Lasagna supports Zarr 3.x writing v2 stores. Extraction defaults
to CPU; GPU extraction must separately reserve resources through the manager.

## Validation

CPU tests: `python -m pytest -q lasagna/tests/test_qualzz_inference.py`.
They cover four-class softmax, upstream normalization dispatch, pooled-before-
quantized output, odd edges, v2 layout, paired completion/provenance validation,
serial/spawned-flush inference with exact expected output and resume, and XY
blocks with empty-tile skipping against one dense pass: the same bytes wherever
CT is stored, absent chunks elsewhere, and no tile run twice.
These synthetic tests do not establish full-checkpoint throughput or GPU memory.

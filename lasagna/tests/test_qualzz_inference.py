"""CPU-only tests of model adaptation, standard Zarr, pooling and tile resume."""
import itertools
import json

import numpy as np
import pytest
import torch
from vc_reader.reader import ZarrWriter, create_group, open_group, open_zarr

from lasagna import qualzz_inference
from lasagna.qualzz_inference import (
    QualzzModelAdapter, ProbabilityZarrOutputAdapter, _NormalizedModel, _block_for_budget,
    _pool2, _quantize, run_inference,
)
from lasagna.tiled_predict3d import run_tiled_inference_3d


def output(tmp_path, shape=(9, 8, 7), chunk=4):
    adapter = QualzzModelAdapter(tmp_path, chunk)
    product, = adapter.output_products
    writer = ProbabilityZarrOutputAdapter(tmp_path / "prediction.zarr", product, shape,
                                          voxel_size_um=9.6, provenance={"test": True})
    writer.prepare()
    return adapter, product, writer


def test_channel_selection_is_four_class_softmax(tmp_path):
    adapter = QualzzModelAdapter(tmp_path)
    logits = torch.arange(8.).reshape(2, 4, 1, 1, 1)
    actual = adapter.product_tensors_from_output({"seg": logits})["semantic"]
    torch.testing.assert_close(actual, logits.softmax(1)[:, 1:3], rtol=0, atol=0)
    assert actual.shape == (2, 2, 1, 1, 1)
    with pytest.raises(ValueError, match="segmentation logits"):
        adapter.product_tensors_from_output({"seg": logits[:, :2]})


def test_model_uses_supplied_normalization_on_uint8_scale():
    seen = []
    def normalize(value):
        seen.append(value)
        return value - 42
    model = _NormalizedModel(torch.nn.Identity(), normalize)
    ct = torch.tensor([0., 1., 128., 255.]).reshape(1, 1, 1, 1, 4)
    torch.testing.assert_close(model(ct / 255), ct - 42)
    torch.testing.assert_close(seen[0], ct, rtol=0, atol=0)


def test_standard_arrays_and_pool_before_quantization(tmp_path):
    adapter, product, writer = output(tmp_path)
    shape = writer.shape
    values = np.random.default_rng(7).random((2, *shape), dtype=np.float32)
    for origin in itertools.product(*(range(0, n, 4) for n in shape)):
        block = values[(slice(None), *(slice(o, o+4) for o in origin))]
        writer.write_product_chunk(product, chunk_origin_zyx=origin,
                                   data=adapter.finalize_product_slab(product, block))
        assert writer.product_chunk_complete(product, chunk_origin_zyx=origin)
    group = open_group(writer.root)
    assert set(group.array_keys()) == {"semantic", "tracing"}
    assert "lasagna" not in json.loads((tmp_path / "prediction.zarr/.zattrs").read_text())
    np.testing.assert_array_equal(group["semantic"][:], _quantize(values))
    np.testing.assert_array_equal(group["tracing"][:], _quantize(_pool2(values)))
    assert group["semantic"].dtype == np.uint8
    assert group["semantic"].chunks == (2, 4, 4, 4)
    meta = json.loads((tmp_path / "prediction.zarr/semantic/.zarray").read_text())
    assert meta["zarr_format"] == 2
    assert meta["compressor"]["cname"] == "zstd"
    # Show that this isn't pooling already-quantized predictions.
    assert np.any(_quantize(_pool2(values)) != _quantize(_pool2(_quantize(values)/255)))


def test_completion_requires_both_scales_and_resume_validates(tmp_path):
    adapter, product, writer = output(tmp_path)
    values = np.zeros((2, 4, 4, 4), np.float32)
    writer.write_product_chunk(product, chunk_origin_zyx=(0, 0, 0),
                               data=adapter.finalize_product_slab(product, values))
    writer.prepare()
    ZarrWriter.open(tmp_path / "prediction.zarr/tracing").remove_chunk((0, 0, 0, 0))
    assert not writer.product_chunk_complete(product, chunk_origin_zyx=(0, 0, 0))
    writer.provenance = {"test": False}
    with pytest.raises(ValueError, match="provenance"):
        writer.prepare()


def test_nonfinite_prediction_rejected():
    with pytest.raises(ValueError, match="non-finite"):
        _quantize(np.array([np.nan]))


class TinyModel(torch.nn.Module):
    def forward(self, tile):
        return {"seg": torch.cat((tile, tile*2, tile*3, tile*4), dim=1)}


@pytest.mark.parametrize("flush_workers", [0, 2])
def test_tiled_engine_and_resume(tmp_path, flush_workers):
    shape = (12, 10, 8)
    adapter, product, writer = output(tmp_path, shape)
    writer_ct = ZarrWriter.create(tmp_path / "ct.zarr/0", shape=shape, chunks=(4, 4, 4), dtype=np.uint8,
                                  compressor=None, separator="/")
    writer_ct[:] = np.random.default_rng(2).integers(1, 256, size=shape, dtype=np.uint8)
    source = open_zarr(tmp_path / "ct.zarr/0")
    progress = {}
    model = TinyModel()
    def run():
        run_tiled_inference_3d(model, source, crop_slices=(0,12,0,10,0,8),
            device=torch.device("cpu"), devices=(torch.device("cpu"),),
            model_adapter=adapter, output_adapter=writer, products=(product,),
            output_regions_zyx={"semantic": (0,0,0,*shape)},
            full_output_shapes_zyx={"semantic": shape}, tile_size=8, overlap=4, border=2,

            tmp_dir=str(tmp_path), progress=progress, product_accumulator_dtype="float32",
            flush_workers=flush_workers)
    run()
    predicted = open_group(writer.root)
    expected = adapter.product_tensors_from_output(model(torch.from_numpy(source[:].copy())
                    .float()[None,None]/255))["semantic"][0].numpy()
    np.testing.assert_array_equal(predicted["semantic"][:], _quantize(expected))
    np.testing.assert_array_equal(predicted["tracing"][:], _quantize(_pool2(expected)))
    model.forward = lambda _: pytest.fail("complete output must not infer again")
    run()


def test_entry_point_rejects_wrong_resolution_before_loading(tmp_path):
    with pytest.raises(ValueError, match="8.64 and 9.6"):
        run_inference(model_dir=tmp_path, input_path=tmp_path, output=tmp_path,
                      voxel_size_um=2.4, devices=(torch.device("cpu"),))


def test_local_model_loader_and_full_cpu_entry_point(tmp_path):
    # Exercise the supplied-code contract without downloading weights or
    # requiring the optional nnU-Net dependency in the base regression suite.
    folder = tmp_path / "model"
    folder.mkdir()
    (folder / "model.py").write_text('''
import torch
class Model(torch.nn.Module):
    def forward(self, x):
        return {"seg": torch.cat((x*0, x*0+1, x*0+2, x*0+3), dim=1)}
def load_model(folder, device):
    return Model().to(device).eval()
def normalize_ct(x):
    return (x - x.mean()) / x.std().clamp_min(10)
''')
    (folder / "plans.json").write_text("{}")
    (folder / "checkpoint_final.pth").write_bytes(b"fixture")
    ZarrWriter.create(tmp_path / "ct.zarr/0", shape=(8,8,8), chunks=(8,8,8), dtype=np.uint8,
                      compressor=None, separator="/")[:] = 128
    path = run_inference(model_dir=folder, input_path=tmp_path / "ct.zarr/0",
                         output=tmp_path / "out.zarr", voxel_size_um=9.6,
                         devices=(torch.device("cpu"),), chunk_size=4,
                         flush_workers=0, accumulator_workers=0)
    result = open_group(path)
    p = torch.arange(4.).softmax(0)[1:3].numpy()
    np.testing.assert_array_equal(result["semantic"][:],
                                  np.broadcast_to(_quantize(p)[:,None,None,None], (2,8,8,8)))
    np.testing.assert_array_equal(result["tracing"][:],
                                  np.broadcast_to(_quantize(p)[:,None,None,None], (2,4,4,4)))


def test_cli_always_enables_cooperative_gpu_handling(monkeypatch):
    from lasagna import qualzz_inference as module
    monkeypatch.delenv("LASAGNA_COOPERATIVE", raising=False)
    monkeypatch.setattr(module, "resolve_inference_devices", lambda **kw: (torch.device("cuda:0"),))
    seen = []
    monkeypatch.setattr(module, "run_inference", lambda **kw: seen.append(kw))
    module.main(["--model-dir", "model", "--input", "ct", "--output", "out", "--voxel-size-um", "9.6"])
    import os
    assert os.environ["LASAGNA_COOPERATIVE"] == "1"
    assert seen[0]["devices"] == (torch.device("cuda:0"),)


CONTEXT_MODEL = '''
import torch
class Model(torch.nn.Module):
    def forward(self, x):
        # Neighbourhood mean, so a tile read without its halo would differ.
        a = torch.nn.functional.avg_pool3d(x, 5, stride=1, padding=2)
        return {"seg": torch.cat((a*0, a*0.7, a*-0.4, a*0.2), dim=1)}
def load_model(folder, device):
    return Model().to(device).eval()
def normalize_ct(x):
    return (x - x.mean()) / x.std().clamp_min(10)
'''


def _context_model(tmp_path):
    folder = tmp_path / "model"
    folder.mkdir()
    (folder / "model.py").write_text(CONTEXT_MODEL)
    (folder / "plans.json").write_text("{}")
    (folder / "checkpoint_final.pth").write_bytes(b"fixture")
    return folder


def _sparse_ct(path, shape=(40, 480, 480), chunk=64):
    """CT whose only stored chunks are y,x chunk indices 1-4; the rest read as 0."""
    writer = ZarrWriter.create(path, shape=shape, chunks=(chunk,)*3, dtype=np.uint8,
                               compressor=None, separator="/")
    rng = np.random.default_rng(5)
    writer[:, chunk:5*chunk, chunk:5*chunk] = rng.integers(1, 256, (shape[0], 4*chunk, 4*chunk), np.uint8)
    return open_zarr(path)


@pytest.mark.parametrize("overlap", [32, 64])
def test_blocks_and_empty_tiles_match_one_dense_pass(tmp_path, monkeypatch, overlap):
    # overlap 32 = 2 x border: tile cores tile space, so stride-aligned blocks
    # must run no tile twice (480 keeps the last tile on the stride lattice).
    # The 192 block edge lies inside the stored CT, where an engine that tested
    # tile footprints would rerun the neighbouring block's border tiles.
    # overlap 64 blends cores, so voxels sum several tiles and must keep their
    # add order across block edges.
    folder = _context_model(tmp_path)
    source = _sparse_ct(tmp_path / "ct.zarr/0")
    shape, chunk = source.shape, 64
    calls = []
    original = QualzzModelAdapter.run_tile_inference
    def counted(self, model, tile, *, device):
        calls.append(int(tile.shape[0]))
        return original(self, model, tile, device=device)
    monkeypatch.setattr(QualzzModelAdapter, "run_tile_inference", counted)

    # Dense reference: one engine pass over every tile, without an input listing.
    adapter = QualzzModelAdapter(folder, chunk)
    product, = adapter.output_products
    dense = ProbabilityZarrOutputAdapter(tmp_path / "dense.zarr", product, shape,
                                         voxel_size_um=9.6, provenance={"test": True})
    dense.prepare()
    run_tiled_inference_3d(adapter.load_model(device=torch.device("cpu")), source,
        crop_slices=(0, shape[0], 0, shape[1], 0, shape[2]), device=torch.device("cpu"),
        model_adapter=adapter, output_adapter=dense, products=(product,),
        output_regions_zyx={"semantic": (0, 0, 0, *shape)}, full_output_shapes_zyx={"semantic": shape},
        output_scaledown_base=1, tile_size=128, overlap=overlap, border=16,
        tmp_dir=str(tmp_path), product_accumulator_dtype="float32")
    dense_tiles = sum(calls)

    runs = {}
    for block_size in (0, 192):
        calls.clear()
        scratch = tmp_path / f"scratch{block_size}"
        scratch.mkdir()
        path = run_inference(model_dir=folder, input_path=tmp_path / "ct.zarr/0",
                             output=tmp_path / f"blocks{block_size}.zarr", voxel_size_um=9.6,
                             devices=(torch.device("cpu"),), overlap=overlap, chunk_size=chunk,
                             flush_workers=0, accumulator_workers=0, block_size=block_size,
                             scratch_dir=scratch)
        assert not any(scratch.iterdir()), "accumulator rings must be removed"
        runs[block_size] = (open_group(path), sum(calls))

    reference = open_group(dense.root)
    stored = np.zeros(shape, bool)
    stored[:, chunk:5*chunk, chunk:5*chunk] = True
    for name, factor in (("semantic", 1), ("tracing", 2)):
        expected = reference[name][:]
        has_ct = stored[::factor, ::factor, ::factor][tuple(slice(n) for n in expected.shape[1:])]
        for group, _ in runs.values():
            actual = group[name][:]
            np.testing.assert_array_equal(actual[:, has_ct], expected[:, has_ct])
            assert not actual[:, ~has_ct].any()
        assert expected[:, ~has_ct].any(), "dense pass must predict over absent CT"
        np.testing.assert_array_equal(runs[192][0][name][:], runs[0][0][name][:])
    one_pass_tiles = runs[0][1]
    assert one_pass_tiles < dense_tiles
    if overlap == 32:
        assert runs[192][1] == one_pass_tiles

    def no_engine(*args, **kwargs):
        raise AssertionError("resuming a finished output started the engine")
    monkeypatch.setattr(qualzz_inference, "run_tiled_inference_3d", no_engine)
    run_inference(model_dir=folder, input_path=tmp_path / "ct.zarr/0", output=tmp_path / "blocks192.zarr",
                  voxel_size_um=9.6, devices=(torch.device("cpu"),), overlap=overlap, chunk_size=chunk,
                  flush_workers=0, accumulator_workers=0, block_size=192, scratch_dir=tmp_path / "scratch192")


def test_input_without_stored_chunks_is_refused(tmp_path):
    folder = _context_model(tmp_path)
    ZarrWriter.create(tmp_path / "ct.zarr/0", shape=(8, 8, 8), chunks=(8, 8, 8), dtype=np.uint8,
                      compressor=None, separator="/")
    with pytest.raises(ValueError, match="no stored chunks"):
        run_inference(model_dir=folder, input_path=tmp_path / "ct.zarr/0", output=tmp_path / "out.zarr",
                      voxel_size_um=9.6, devices=(torch.device("cpu"),), chunk_size=4,
                      flush_workers=0, accumulator_workers=0, scratch_dir=tmp_path)


def test_input_with_symlinked_directory_is_refused(tmp_path):
    folder = _context_model(tmp_path)
    level = tmp_path / "ct.zarr/0"
    writer = ZarrWriter.create(level, shape=(8, 8, 8), chunks=(4, 4, 4), dtype=np.uint8,
                               compressor=None, separator="/")
    writer[:] = np.ones((8, 8, 8), np.uint8)
    moved = tmp_path / "elsewhere"
    (level / "1").rename(moved)
    (level / "1").symlink_to(moved, target_is_directory=True)
    assert open_zarr(level)[4:].min() == 1, "the reader follows the link the listing skips"
    with pytest.raises(ValueError, match="symlinked directory"):
        run_inference(model_dir=folder, input_path=level, output=tmp_path / "out.zarr",
                      voxel_size_um=9.6, devices=(torch.device("cpu"),), chunk_size=4,
                      flush_workers=0, accumulator_workers=0, scratch_dir=tmp_path)


def test_accumulator_budget_picks_stride_aligned_blocks():
    # A PHerc1545 band at the default tiles and chunks: 384-voxel steps are
    # both 3 chunks and 4 tile strides.
    rings = dict(tile_size=128, overlap=32, border=16, chunk_size=128, channels=2, flush_workers=2)
    band = (512, 7506, 7506)
    assert [_block_for_budget(gib * 2**30, band, **rings) for gib in (1.2, 2, 4, 12)] == [384, 384, 768, 1536]
    assert _block_for_budget(2**40, (512, 1000, 600), **rings) == 1152
    # Below one aligned block it falls back to whole chunks, then refuses.
    assert _block_for_budget(2**29, band, **rings) == 128
    with pytest.raises(ValueError, match="budget too small"):
        _block_for_budget(2**27, band, **rings)

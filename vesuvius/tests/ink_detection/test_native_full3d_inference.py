"""Native volume planning, prediction, and OME-Zarr lifecycle tests."""

from __future__ import annotations

import itertools
import json
import math
from pathlib import Path
import pickle
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from vc_reader.reader import open_group

from vesuvius.ink_detection.inference import infer_full3d_tifxyz as native


def _plan(**overrides) -> native.NativePlan:
    values = {
        "volume_path": "memory://native",
        "array_shape_zyx": (8, 8, 8),
        "chunk_shape_zyx": (4, 4, 4),
        "x": np.array([0.25, 4.0, 7.9, np.nan, -1.0, 8.0]),
        "y": np.array([0.25, 0.0, 7.9, 1.0, 1.0, 1.0]),
        "z": np.array([0.25, 0.0, 7.9, 1.0, 1.0, 1.0]),
        "valid": np.array([True, True, True, False, False, True]),
        "patch_size_zyx": (5, 5, 5),
        "overlap": 0.5,
        "chunk_halo": 0,
        "write_region": "occupied",
        "max_target_chunks": None,
    }
    values.update(overrides)
    return native.build_native_plan(**values)


def test_native_weight_selection_uses_only_ema_then_model():
    ema = {"weight": torch.tensor(1.0)}
    flat_alias = {"weight": torch.tensor(2.0)}
    model = {"weight": torch.tensor(3.0)}

    selected, state = native.select_native_inference_weights(
        {"ema_model": ema, "state_dict": flat_alias, "model": model}
    )
    assert selected == "ema_model"
    assert state is ema

    selected, state = native.select_native_inference_weights(
        {"state_dict": flat_alias, "model": model}
    )
    assert selected == "model"
    assert state is model


def test_negative_resolution_fails_before_native_volume_open(monkeypatch):
    config = SimpleNamespace(
        data=SimpleNamespace(mode="full_3d"),
        model=SimpleNamespace(crop_size=(3, 3, 3)),
    )
    monkeypatch.setattr(
        native,
        "_load_checkpoint_config",
        lambda checkpoint: ({}, config),
    )

    with pytest.raises(ValueError, match="must be >= 0"):
        monkeypatch.setattr(
            native,
            "_open_shared_volume",
            lambda *args, **kwargs: pytest.fail(
                "negative resolution opened the volume"
            ),
        )
        native._plan_from_args(
            SimpleNamespace(checkpoint=Path("unused"), resolution="-1")
        )


def test_cli_accepts_both_flag_spellings_and_explicit_workers_alias(tmp_path):
    common = [str(tmp_path), str(tmp_path / "model.ckpt"), str(tmp_path / "out")]
    hyphen = native.parse_args(
        [
            *common,
            "--num-workers",
            "0",
            "--downsample-workers",
            "3",
            "--max-target-chunks",
            "2",
            "--no-compile",
        ]
    )
    underscore = native.parse_args(
        [
            *common,
            "--num_workers",
            "0",
            "--downsample_workers",
            "3",
            "--max_target_chunks",
            "2",
            "--no_compile",
        ]
    )
    legacy = native.parse_args([*common, "--workers", "7"])
    assert hyphen.num_workers == underscore.num_workers == 0
    assert hyphen.downsample_workers == underscore.downsample_workers == 3
    assert hyphen.max_target_chunks == underscore.max_target_chunks == 2
    assert hyphen.compile_model is underscore.compile_model is False
    assert legacy.num_workers == 7
    assert native.parse_args([*common, "--overlap", "1"]).overlap == 1.0
    with pytest.raises(SystemExit):
        native.parse_args([*common, "--foreground-channel", "0"])


def test_volume_source_is_anchored_to_tifxyz_directory(tmp_path, monkeypatch):
    tifxyz = tmp_path / "map"
    tifxyz.mkdir()
    source = tifxyz / "volume_source.txt"

    with pytest.raises(ValueError, match="empty"):
        source.write_text("\n", encoding="utf-8")
        native.read_volume_source(tifxyz)

    source.write_text("https://example.invalid/a.zarr\n", encoding="utf-8")
    assert native.read_volume_source(tifxyz) == "https://example.invalid/a.zarr"

    absolute = tmp_path / "absolute.zarr"
    source.write_text(str(absolute), encoding="utf-8")
    assert native.read_volume_source(tifxyz) == str(absolute)

    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    source.write_text("../native/volume.zarr", encoding="utf-8")
    assert native.read_volume_source(tifxyz) == str(
        (tifxyz / "../native/volume.zarr").resolve()
    )

    source.unlink()
    with pytest.raises(FileNotFoundError, match="volume_source.txt"):
        native.read_volume_source(tifxyz)


def test_occupied_plan_handles_boundaries_halo_stride_and_truncation():
    occupied = {
        native.ChunkId(0, 0, 0),
        native.ChunkId(0, 0, 1),
        native.ChunkId(1, 1, 1),
    }
    plan = _plan()
    assert plan.occupied_chunks == occupied
    assert plan.target_chunks == occupied
    assert plan.stride_zyx == (2, 2, 2)
    assert list(plan.patches) == sorted(set(plan.patches))
    assert set(plan.contribution_counts) == occupied
    assert all(count > 0 for count in plan.contribution_counts.values())

    halo = _plan(chunk_halo=1, write_region="expanded")
    assert halo.expanded_chunks == {
        native.ChunkId(z, y, x)
        for z in range(2)
        for y in range(2)
        for x in range(2)
    }
    truncated = _plan(chunk_halo=1, write_region="expanded", max_target_chunks=2)
    assert truncated.target_chunks == frozenset(sorted(halo.expanded_chunks)[:2])


def test_patch_lattice_covers_small_volume_and_validates_direct_calls():
    small = _plan(
        array_shape_zyx=(3, 4, 2),
        chunk_shape_zyx=(2, 2, 2),
        x=np.array([1.9]),
        y=np.array([3.9]),
        z=np.array([2.9]),
        valid=np.array([True]),
        patch_size_zyx=(7, 6, 5),
        overlap=0.0,
    )
    assert small.patches == (native.PatchSpec(0, 0, 0),)
    assert small.contribution_counts == {native.ChunkId(1, 1, 0): 1}

    for field, bad in (
        ("array_shape_zyx", (0, 2, 2)),
        ("chunk_shape_zyx", (1, 2)),
        ("patch_size_zyx", (1, -1, 1)),
        ("overlap", 1.0),
        ("chunk_halo", -1),
        ("max_target_chunks", 0),
        ("write_region", "everything"),
    ):
        with pytest.raises(ValueError):
            _plan(**{field: bad})


def test_probability_rules_resize_and_exact_eight_way_tta():
    shape = (3, 4, 5)
    one = native.logits_to_probabilities(
        torch.zeros((1, 1, 1, 2, 2)),
        patch_size_zyx=shape,
    )
    assert one.shape == (1, 1, *shape)
    assert torch.equal(one, torch.full_like(one, 0.5))

    variants = native.tta_variants(True)
    assert variants == [
        (),
        (0,),
        (1,),
        (2,),
        (0, 1),
        (0, 2),
        (1, 2),
        (0, 1, 2),
    ]
    images = torch.arange(2 * 3 * 4, dtype=torch.float32).reshape(1, 1, 2, 3, 4)
    expected = torch.sigmoid(images)
    for batch_size in (1, 3, None):
        actual = native.predict_batch(
            lambda value: value,
            images,
            variants=variants,
            tta_batch_size=batch_size,
            patch_size_zyx=(2, 3, 4),
        )
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=2e-7)


def test_native_model_pads_input_depth_and_resizes_like_training(monkeypatch):
    from vesuvius.ink_detection.training.train import prepare_loss_inputs

    class EchoInk(torch.nn.Module):
        def forward(self, image):
            return {"ink": image}

    monkeypatch.setattr(native, "make_model", lambda config: EchoInk())
    monkeypatch.setattr(native, "load_model_state", lambda model, state: None)
    config = SimpleNamespace(model=SimpleNamespace(input_pad_depth_to=4))
    args = native.parse_args(["tifxyz", "memory.pth", "out.zarr", "--no-compile"])
    bundle = native._load_native_model({"model": {}}, config, args)
    logits = bundle.model(torch.randn(1, 1, 3, 8, 8).to(bundle.device)).cpu()
    assert logits.shape[2] == 4  # the depth the model was trained on

    patch = torch.zeros(1, 1, 3, 8, 8)
    batch = {"image": patch, "inklabels": patch, "supervision_mask": patch + 1}
    expected = prepare_loss_inputs(logits, batch, mode="full_3d")[0].sigmoid()
    actual = native.logits_to_probabilities(logits, patch_size_zyx=(3, 8, 8))
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=1e-6)


def test_importance_maps_and_sparse_accumulator_round_half_to_even():
    gaussian = native.create_importance_map((3, 4, 5), mode="gaussian")
    assert gaussian.shape == (3, 4, 5)
    assert gaussian.dtype == np.float32
    assert gaussian.max() == 1.0
    assert gaussian.min() >= np.finfo(np.float32).eps
    assert np.array_equal(
        native.create_importance_map((2, 2, 2), mode="constant"),
        np.ones((2, 2, 2), dtype=np.float32),
    )

    output = np.zeros((4, 4, 4), dtype=np.uint8)
    target = native.ChunkId(0, 0, 0)
    accumulator = native.ChunkAccumulator3D(
        output=output,
        target_chunks={target},
        contribution_counts={target: 2},
        chunk_shape_zyx=(2, 2, 2),
    )
    weights = np.ones((3, 3, 3), dtype=np.float32)
    accumulator.add_patch(
        patch_start_zyx=(0, 0, 0),
        probabilities=np.full((3, 3, 3), 0.25, dtype=np.float32),
        weights=weights,
    )
    assert not output.any()
    accumulator.add_patch(
        patch_start_zyx=(0, 0, 0),
        probabilities=np.full((3, 3, 3), 0.75, dtype=np.float32),
        weights=weights,
    )
    assert np.all(output[:2, :2, :2] == 128)
    assert not output[2:, :, :].any()
    assert not output[:, 2:, :].any()
    assert not output[:, :, 2:].any()


def test_native_dataset_normalizes_only_valid_voxels_and_builds_surface_channel(
    monkeypatch,
):
    from vesuvius.ink_detection.config import NormalizationConfig

    config = SimpleNamespace(
        data=SimpleNamespace(
            mode="full_3d",
            normalization=NormalizationConfig.from_value(
                {"mode": "divide", "divisor": 10}
            ),
        )
    )
    current_volume = {"value": np.full((2, 2, 2), 10, dtype=np.uint8)}

    def open_shared_volume_stub(path, resolution, *, cache_dir, cache_max_gb):
        return current_volume["value"]

    monkeypatch.setattr(native, "_open_shared_volume", open_shared_volume_stub)
    original_params = native.native_tifxyz_pyramid_params
    params_calls = []
    monkeypatch.setattr(
        native,
        "native_tifxyz_pyramid_params",
        lambda resolution: (
            params_calls.append(resolution) or original_params(resolution)
        ),
    )
    dataset = native.NativePatchDataset(
        tifxyz_dir=Path("unused"),
        volume_path="memory://native",
        resolution="0",
        patches=(native.PatchSpec(-1, -1, -1),),
        patch_size_zyx=(3, 3, 3),
        config=config,
    )
    image, metadata = dataset[0]
    assert metadata.tolist() == [-1, -1, -1]
    assert image.shape == (1, 3, 3, 3)
    assert not image[0, 0].any()
    assert not image[0, :, 0].any()
    assert not image[0, :, :, 0].any()
    assert torch.all(image[0, 1:, 1:, 1:] == 1)

    class FakeTifxyz:
        full_shape = (2, 2)
        scale = (1.0, 1.0)

        def __init__(self):
            yy, xx = np.mgrid[:2, :2]
            self.xyz = np.stack((xx, yy, np.zeros((2, 2))), axis=-1).astype(np.float32)
            self.valid = np.ones((2, 2), dtype=bool)

        def resample(self, origin, step, size, *, interpolation, normals=False):
            rows = int(origin[0]) + int(step[0]) * np.arange(size[0])
            columns = int(origin[1]) + int(step[1]) * np.arange(size[1])
            xyz = self.xyz[rows[:, None], columns[None, :]]
            return xyz, np.ones(xyz.shape[:2], dtype=bool)

    wrap_config = SimpleNamespace(
        data=SimpleNamespace(
            mode="full_3d_single_wrap",
            normalization=NormalizationConfig.from_value("none"),
        )
    )
    current_volume["value"] = np.zeros((2, 2, 2), dtype=np.uint8)
    wrapped = native.NativePatchDataset(
        tifxyz_dir=Path("unused"),
        volume_path="memory://native",
        resolution="0",
        patches=(native.PatchSpec(0, 0, 0),),
        patch_size_zyx=(2, 2, 2),
        config=wrap_config,
    )
    wrapped._tifxyz = FakeTifxyz()
    wrapped_image, _ = wrapped[0]
    assert wrapped_image.shape == (2, 2, 2, 2)
    assert torch.all(wrapped_image[1, 0] == 1)
    assert torch.allclose(wrapped_image[1, 1], torch.full((2, 2), 0.9))
    assert params_calls == [0, 0]

    restored = pickle.loads(pickle.dumps(wrapped))
    assert restored._volume is None
    assert restored._tifxyz is None
    assert restored._coarse_positions is None


@pytest.mark.parametrize("mode", ["full_3d", "full_3d_single_wrap"])
@pytest.mark.parametrize(
    "normalization", ["none", "minmax", "robust_mad", {"mode": "divide", "divisor": 7}]
)
def test_native_dataset_batch_reads_match_single_reads(monkeypatch, normalization, mode):
    from vesuvius.ink_detection.config import NormalizationConfig

    class CountingVolume:
        def __init__(self, values):
            self.values = values
            self.shape = values.shape
            self.dtype = values.dtype
            self.reads = 0

        def __getitem__(self, key):
            self.reads += 1
            return self.values[key]

        def read_regions(self, regions, out):
            self.reads += 1
            for (start, size), destination in zip(regions, out):
                selection = tuple(slice(a, a + n) for a, n in zip(start, size))
                destination[...] = self.values[selection]
            return out

    rng = np.random.default_rng(0)
    volume = CountingVolume(rng.integers(0, 256, (20, 24, 28), dtype=np.uint8))
    monkeypatch.setattr(
        native, "_open_shared_volume", lambda *args, **kwargs: volume
    )
    monkeypatch.setattr(
        native.NativePatchDataset, "_surface_mask",
        lambda self, bbox: np.full(self.patch_size_zyx, bbox[0], dtype=np.float32),
    )
    config = SimpleNamespace(
        data=SimpleNamespace(
            mode=mode,
            normalization=NormalizationConfig.from_value(normalization),
        )
    )
    patches = native.build_patch_plan_for_chunks(
        target_chunks=[native.ChunkId(0, 0, 0), native.ChunkId(1, 1, 1)],
        array_shape_zyx=volume.shape,
        chunk_shape_zyx=(10, 12, 14),
        patch_size_zyx=(8, 8, 8),
        stride_zyx=(4, 4, 4),
    )
    patches = [*patches, native.PatchSpec(-3, 20, 25), native.PatchSpec(30, 0, 0)]
    dataset = native.NativePatchDataset(
        tifxyz_dir=Path("unused"),
        volume_path="memory://native",
        resolution="0",
        patches=patches,
        patch_size_zyx=(8, 8, 8),
        config=config,
    )
    indices = list(range(len(patches)))
    single = [dataset[index] for index in indices]
    volume.reads = 0
    batched = dataset.__getitems__(indices)
    assert volume.reads == 1
    assert len(batched) == len(single)
    for (image, metadata), (expected, expected_metadata) in zip(batched, single):
        assert torch.equal(metadata, expected_metadata)
        assert torch.equal(image, expected)
    assert dataset.__getitems__([]) == []
    reordered = [len(patches) - 1, 2, 0, 2]
    for actual, index in zip(dataset.__getitems__(reordered), reordered):
        assert torch.equal(actual[0], single[index][0])
        assert torch.equal(actual[1], single[index][1])
    loader = torch.utils.data.DataLoader(dataset, batch_size=5)
    images = torch.cat([batch[0] for batch in loader])
    assert torch.equal(images, torch.stack([image for image, _ in single]))

    local = native.NativePatchDataset(
        tifxyz_dir=Path("unused"),
        volume_path="local.zarr",
        resolution="0",
        patches=patches,
        patch_size_zyx=(8, 8, 8),
        config=config,
    )
    volume.reads = 0
    local_batch = local.__getitems__(indices)
    assert volume.reads == len(patches) - 1  # the outside patch reads nothing
    assert all(torch.equal(a[0], b[0]) for a, b in zip(local_batch, single))


def test_explicit_v2_sparse_ome_zarr_and_encoded_uint8_pyramid(tmp_path):
    path = tmp_path / "prediction.zarr"
    arrays = native.create_output_zarr(
        path,
        shape_zyx=(5, 4, 3),
        chunks_zyx=(2, 2, 2),
    )
    assert [tuple(array.shape) for array in arrays] == [
        (5, 4, 3),
        (3, 2, 2),
        (2, 1, 1),
        (1, 1, 1),
        (1, 1, 1),
        (1, 1, 1),
    ]
    root_metadata = json.loads((path / ".zgroup").read_text(encoding="utf-8"))
    array_metadata = json.loads((path / "0" / ".zarray").read_text(encoding="utf-8"))
    assert root_metadata == {"zarr_format": 2}
    assert array_metadata["dimension_separator"] == "/"
    assert array_metadata["compressor"] == {
        "id": "blosc",
        "cname": "zstd",
        "clevel": 3,
        "shuffle": 2,
        "blocksize": 0,
    }
    assert not any((path / "0").glob("[0-9]*"))

    values = np.array(
        [[[0, 1], [2, 3]], [[252, 253], [254, 255]]], dtype=np.uint8
    )
    target = native.ChunkId(0, 0, 0)
    accumulator = native.ChunkAccumulator3D(
        output=arrays[0],
        target_chunks={target},
        contribution_counts={target: 1},
        chunk_shape_zyx=(2, 2, 2),
    )
    accumulator.add_patch(
        patch_start_zyx=(0, 0, 0),
        probabilities=values.astype(np.float32) / 255.0,
        weights=np.ones(values.shape, dtype=np.float32),
    )
    assert (path / "0" / "0" / "0" / "0").is_file()
    assert not (path / "0" / "0" / "0" / "1").exists()

    native.build_downsample_levels(arrays, level0_written_chunks={target}, workers=2)
    expected = np.zeros((5, 4, 3), dtype=np.uint8)
    expected[:2, :2, :2] = values
    multiscales = open_group(path).attrs["multiscales"][0]
    assert multiscales["version"] == "0.4"
    assert [axis["name"] for axis in multiscales["axes"]] == ["z", "y", "x"]
    assert [dataset["path"] for dataset in multiscales["datasets"]] == [
        str(level) for level in range(6)
    ]
    for level, array in enumerate(arrays):
        assert array.attrs["_ARRAY_DIMENSIONS"] == ["z", "y", "x"]
        assert np.array_equal(array[...], expected)
        assert multiscales["datasets"][level]["coordinateTransformations"][0] == {
            "type": "scale", "scale": [float(2**level)] * 3
        }
        if level + 1 < len(arrays):
            expected = _downsample_mean_3d(expected)


def _downsample_mean_3d(block):
    """Rounded mean of each 2x2x2 block, clipped at the array's end."""
    shape = tuple((value + 1) // 2 for value in block.shape)
    output = np.zeros(shape, block.dtype)
    for index in np.ndindex(shape):
        output[index] = np.rint(block[tuple(slice(2 * i, 2 * i + 2) for i in index)].mean())
    return output


def test_output_reject_overwrite_and_coherent_rerun(tmp_path):
    output = tmp_path / "existing.zarr"
    output.mkdir()
    sentinel = output / "sentinel"
    sentinel.write_bytes(b"keep-me")
    with pytest.raises(FileExistsError):
        native.create_output_zarr(
            output,
            shape_zyx=(2, 2, 2),
            chunks_zyx=(2, 2, 2),
            overwrite=False,
        )
    assert sentinel.read_bytes() == b"keep-me"

    first = native.create_output_zarr(
        output,
        shape_zyx=(2, 2, 2),
        chunks_zyx=(2, 2, 2),
        overwrite=True,
    )
    assert not sentinel.exists()
    first[0][:] = 17
    second = native.create_output_zarr(
        output,
        shape_zyx=(2, 2, 2),
        chunks_zyx=(2, 2, 2),
        overwrite=True,
    )
    assert not np.asarray(second[0]).any()
    assert len(list(open_group(output).array_keys())) == 6


def test_plan_only_does_not_load_runtime_or_touch_existing_output(tmp_path, monkeypatch):
    plan = _plan(max_target_chunks=1)
    sentinel = tmp_path / "prediction.zarr"
    sentinel.write_bytes(b"immutable")
    calls = []

    def fake_plan(args):
        calls.append(args.checkpoint)
        return plan, object(), object()

    monkeypatch.setattr(native, "_plan_from_args", fake_plan)
    monkeypatch.setattr(
        native,
        "_load_native_model",
        lambda *args: pytest.fail("model must remain lazy"),
    )
    args = SimpleNamespace(
        plan_only=True,
        checkpoint=tmp_path / "model.ckpt",
        output_zarr=sentinel,
    )
    assert native.run_command(args) == 0
    assert calls == [args.checkpoint]
    assert sentinel.read_bytes() == b"immutable"


def test_injected_command_runtime_writes_then_builds_all_levels(tmp_path, monkeypatch):
    target = native.ChunkId(0, 0, 0)
    plan = native.NativePlan(
        volume_path="memory://native",
        array_shape_zyx=(2, 2, 2),
        chunk_shape_zyx=(2, 2, 2),
        occupied_chunks=frozenset({target}),
        expanded_chunks=frozenset({target}),
        target_chunks=frozenset({target}),
        patches=(native.PatchSpec(0, 0, 0),),
        contribution_counts={target: 1},
        patch_size_zyx=(2, 2, 2),
        stride_zyx=(1, 1, 1),
    )
    from vesuvius.ink_detection.config import NormalizationConfig

    blocker = tmp_path / "embedded-io-blocker"
    blocker.write_bytes(b"")
    config = SimpleNamespace(
        data=SimpleNamespace(
            mode="full_3d",
            normalization=NormalizationConfig.from_value("none"),
            volume_auth_json=str(blocker / "auth.json"),
            volume_cache_dir=str(blocker / "cache"),
            volume_cache_max_gb=120.0,
        )
    )
    from vc_reader.reader import ZarrReader, ZarrWriter

    volume_path = tmp_path / "input.zarr"
    ZarrWriter.create(volume_path, shape=(2, 2, 2), chunks=(2, 2, 2), dtype="uint8")
    volume = ZarrReader.open(volume_path)
    monkeypatch.setattr(
        native,
        "_plan_from_args",
        lambda args: (plan, object(), config),
    )

    class ZeroModel(torch.nn.Module):
        def forward(self, images):
            return torch.zeros(
                (images.shape[0], 1, *images.shape[-3:]),
                dtype=images.dtype,
                device=images.device,
            )

    bundle = native.NativeModelBundle(
        model=ZeroModel(),
        device=torch.device("cpu"),
        amp_dtype=None,
    )
    monkeypatch.setattr(native, "_load_native_model", lambda *args: bundle)

    def open_shared_volume_stub(path, resolution, *, cache_dir, cache_max_gb):
        assert (cache_dir, cache_max_gb) == (None, None)
        return volume

    monkeypatch.setattr(native, "_open_shared_volume", open_shared_volume_stub)

    output = tmp_path / "command.zarr"
    args = SimpleNamespace(
        plan_only=False,
        checkpoint=tmp_path / "model.ckpt",
        tifxyz_dir=tmp_path,
        output_zarr=output,
        resolution="0",
        overwrite=False,
        blend_mode="constant",
        tta=False,
        tta_batch_size=None,
        batch_size=1,
        gpu_ids=[],
        num_workers=0,
        prefetch_factor=2,
        cache_dir=None,
        cache_max_gb=None,
        downsample_workers=1,
    )
    assert native.run_command(args) == 0
    group = open_group(output)
    assert sorted(group.array_keys()) == [str(level) for level in range(6)]
    assert all(np.all(np.asarray(group[str(level)]) == 128) for level in range(6))
    with pytest.raises(FileExistsError):
        native.run_command(args)
    assert np.all(np.asarray(open_group(output)["0"]) == 128)
    args.overwrite = True
    assert native.run_command(args) == 0
    assert all(
        np.all(np.asarray(open_group(output)[str(level)]) == 128)
        for level in range(6)
    )


def _blend_with(accumulator_factory, *, device):
    """Blend a tiny synthetic model's patches over a random volume in plan order."""

    shape, chunks, patch = (13, 23, 17), (4, 7, 5), (6, 10, 8)
    grid = [np.arange(size) + 0.5 for size in shape]
    z, y, x = (values.ravel() for values in np.meshgrid(*grid, indexing="ij"))
    plan = _plan(
        array_shape_zyx=shape,
        chunk_shape_zyx=chunks,
        x=x,
        y=y,
        z=z,
        valid=np.ones(z.size, dtype=bool),
        patch_size_zyx=patch,
        chunk_halo=0,
    )
    generator = torch.Generator().manual_seed(7)
    volume = torch.rand(shape, generator=generator)
    model = torch.nn.Conv3d(1, 1, 3, padding=1)
    with torch.no_grad():
        model.weight.copy_(torch.randn(model.weight.shape, generator=generator))
        model.bias.fill_(-0.3)
    model = model.to(device)
    output = np.zeros(shape, dtype=np.uint8)
    weights = native.create_importance_map(patch, mode="gaussian")
    accumulator = accumulator_factory(output, plan, weights)
    with torch.inference_mode():
        for spec in plan.patches:
            starts = (spec.z0, spec.y0, spec.x0)
            crop = volume[tuple(slice(s, s + p) for s, p in zip(starts, patch))]
            probabilities = model(crop[None, None].to(device)).sigmoid()[0, 0]
            if isinstance(accumulator, native.DeviceChunkAccumulator3D):
                accumulator.add_patch(patch_start_zyx=starts, probabilities=probabilities)
            else:
                accumulator.add_patch(
                    patch_start_zyx=starts,
                    probabilities=probabilities.cpu().numpy(),
                    weights=weights,
                )
    accumulator.flush_remaining()
    assert not accumulator.buffers
    return output, plan, accumulator


def _host_accumulator(output, plan, weights):
    return native.ChunkAccumulator3D(
        output=output,
        target_chunks=set(plan.target_chunks),
        contribution_counts=plan.contribution_counts,
        chunk_shape_zyx=plan.chunk_shape_zyx,
    )


def _device_accumulator(device, max_bytes):
    def factory(output, plan, weights):
        return native.DeviceChunkAccumulator3D(
            output=output,
            target_chunks=set(plan.target_chunks),
            contribution_counts=plan.contribution_counts,
            chunk_shape_zyx=plan.chunk_shape_zyx,
            weights=weights,
            device=device,
            max_bytes=max_bytes,
        )

    return factory


# The device blend adds each patch to a voxel in the same order as the host
# blend and does the same IEEE float32 multiply, add, divide, clip and
# round-half-even steps (no fused multiply-add), so the uint8 codes must be
# identical, not merely within one code.
@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="needs CUDA"
            ),
        ),
    ],
)
def test_device_blend_matches_host_blend_with_host_fallback(device):
    expected, plan, _ = _blend_with(_host_accumulator, device=device)
    assert expected.any() and len(plan.patches) > 8

    unlimited, _, accumulator = _blend_with(
        _device_accumulator(device, 1 << 40), device=device
    )
    assert np.array_equal(unlimited, expected)
    assert accumulator.host_chunks == 0
    assert accumulator.device_chunks == len(plan.target_chunks)
    assert accumulator.device_bytes == 0

    # room for the scratch and three 4x7x5 chunks: the rest blend on the host
    chunk = 4 * 7 * 5
    budget = 4 * chunk + 3 * 2 * 4 * chunk
    mixed, _, accumulator = _blend_with(
        _device_accumulator(device, budget), device=device
    )
    assert np.array_equal(mixed, expected)
    assert accumulator.host_chunks > 0 and accumulator.device_chunks > 0
    assert accumulator.peak_device_bytes <= budget


def test_flushed_device_buffers_are_released_before_the_next_allocation(monkeypatch):
    import weakref

    flushed = []
    flush, allocate = (
        native.DeviceChunkAccumulator3D._flush,
        native.DeviceChunkAccumulator3D._allocate,
    )

    def tracking_flush(self, chunk):
        refs = [weakref.ref(tensor) for tensor in self.buffers[chunk]]
        flush(self, chunk)
        flushed.extend(refs)

    def checking_allocate(self, shape):
        # a live flushed buffer here means device use can pass the budget
        assert all(ref() is None for ref in flushed)
        return allocate(self, shape)

    monkeypatch.setattr(native.DeviceChunkAccumulator3D, "_flush", tracking_flush)
    monkeypatch.setattr(native.DeviceChunkAccumulator3D, "_allocate", checking_allocate)
    _, plan, accumulator = _blend_with(_device_accumulator("cpu", 1 << 40), device="cpu")
    assert len(flushed) == 2 * len(plan.target_chunks)
    assert accumulator.device_chunks == len(plan.target_chunks)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_device_blend_allocates_only_budgeted_buffers(monkeypatch):
    # finishing a chunk and adding to an existing chunk must not allocate on
    # the device, so an allocation failure can only hit the guarded path
    flush, accumulate = (
        native.DeviceChunkAccumulator3D._flush,
        native.DeviceChunkAccumulator3D._accumulate,
    )
    checked = {"flush": 0, "accumulate": 0}

    def assert_no_allocation(kind, call):
        torch.cuda.synchronize()
        before = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        call()
        torch.cuda.synchronize()
        assert torch.cuda.max_memory_allocated() == before
        checked[kind] += 1

    def checked_flush(self, chunk):
        if isinstance(self.buffers[chunk][0], np.ndarray):
            return flush(self, chunk)
        assert_no_allocation("flush", lambda: flush(self, chunk))

    def checked_accumulate(self, chunk, *rest):
        if chunk not in self.buffers:
            return accumulate(self, chunk, *rest)
        assert_no_allocation("accumulate", lambda: accumulate(self, chunk, *rest))

    monkeypatch.setattr(native.DeviceChunkAccumulator3D, "_flush", checked_flush)
    monkeypatch.setattr(native.DeviceChunkAccumulator3D, "_accumulate", checked_accumulate)
    _blend_with(_device_accumulator("cuda", 1 << 40), device="cuda")
    assert checked["flush"] > 0 and checked["accumulate"] > 0


def test_device_blend_rejects_mismatched_patches_and_non_cuda_models():
    target = native.ChunkId(0, 0, 0)
    accumulator = native.DeviceChunkAccumulator3D(
        output=np.zeros((2, 2, 2), dtype=np.uint8),
        target_chunks={target},
        contribution_counts={target: 1},
        chunk_shape_zyx=(2, 2, 2),
        weights=np.ones((2, 2, 2), dtype=np.float32),
        device="cpu",
        max_bytes=1024,
    )
    with pytest.raises(TypeError):
        accumulator.add_patch(
            patch_start_zyx=(0, 0, 0),
            probabilities=torch.zeros((2, 2, 2), dtype=torch.bfloat16),
        )
    with pytest.raises(ValueError):
        accumulator.add_patch(
            patch_start_zyx=(0, 0, 0), probabilities=torch.zeros((3, 2, 2))
        )
    with pytest.raises(ValueError, match="max_bytes is required"):
        native.DeviceChunkAccumulator3D(
            output=np.zeros((2, 2, 2), dtype=np.uint8),
            target_chunks={target},
            contribution_counts={target: 1},
            chunk_shape_zyx=(2, 2, 2),
            weights=np.ones((2, 2, 2), dtype=np.float32),
            device="cpu",
        ).add_patch(patch_start_zyx=(0, 0, 0), probabilities=torch.zeros((2, 2, 2)))

    with pytest.raises(ValueError, match="needs a CUDA model device"):
        native.run_native_inference(
            args=SimpleNamespace(
                batch_size=1,
                gpu_ids=(),
                num_workers=0,
                prefetch_factor=2,
                tta=False,
                tta_batch_size=None,
                blend_device="cuda",
            ),
            bundle=SimpleNamespace(
                model=None, device=torch.device("cpu"), amp_dtype=None
            ),
            dataset=[],
            plan=SimpleNamespace(
                target_chunks=frozenset(),
                contribution_counts={},
                chunk_shape_zyx=(2, 2, 2),
                patch_size_zyx=(2, 2, 2),
            ),
            output=np.zeros((2, 2, 2), dtype=np.uint8),
            importance_map=np.ones((2, 2, 2), dtype=np.float32),
        )


def test_cli_blend_device_flags():
    args = native.parse_args(["t", "c.pth", "o.zarr"])
    assert (args.blend_device, args.blend_max_gb) == ("cpu", None)
    args = native.parse_args(
        ["t", "c.pth", "o.zarr", "--blend-device", "cuda", "--blend_max_gb", "2.5"]
    )
    assert (args.blend_device, args.blend_max_gb) == ("cuda", 2.5)
    with pytest.raises(SystemExit):
        native.parse_args(["t", "c.pth", "o.zarr", "--blend-max-gb", "0"])


_TABLE_NORMALIZATIONS = [
    "percentile_minmax",
    {"mode": "percentile_minmax", "percentile_lower": 6.37, "percentile_upper": 94.77},
    {"mode": "percentile_minmax", "percentile_lower": 0.0, "percentile_upper": 100.0},
    "minmax",
    "none",
    {"mode": "divide", "divisor": 7},
    {"mode": "clip_zscore", "mean": 90.5, "std": 31.0, "clip_min": 20, "clip_max": 230},
    {"mode": "percentile_minmax", "percentile_lower": 35.01, "percentile_upper": 59.01},
]


def _uint8_cases(rng):
    """Small uint8 crops whose percentiles fall between unequal neighbours."""

    for size in (1, 2, 3, 77, 101, 705, 1420, 2762):
        yield rng.integers(0, 256, size)
        yield rng.integers(int(rng.integers(0, 250)), 256, size)
        yield np.clip(rng.normal(100, 30, size), 0, 255)
        yield rng.choice([0, 7, 200, 255], size)
        yield np.full(size, int(rng.integers(0, 256)))
    yield np.r_[np.zeros(999), 9]
    yield np.array([0, 7, 200, 255, 0, 7])  # fused C++ arithmetic would round this one differently
    yield np.r_[np.zeros(4000), rng.integers(1, 256, 1000)]


# The table must equal vc_reader's normalizer exactly (float64 percentile
# lerp and division, float32 rounding), including fractional ranks, constant
# crops and many zeros: inference inputs may not change by a single ulp.
@pytest.mark.parametrize("normalization", _TABLE_NORMALIZATIONS)
def test_uint8_normalization_table_matches_normalize_image(normalization):
    from vesuvius.ink_detection.config import NormalizationConfig
    from vesuvius.ink_detection.data.normalization import (
        HISTOGRAM_TABLE_MODES,
        normalize_image,
        uint8_normalization_table,
    )

    config = NormalizationConfig.from_value(normalization)
    rng = np.random.default_rng(3)
    checked = 0
    for values in _uint8_cases(rng):
        crop = values.astype(np.uint8)
        expected = normalize_image(crop.astype(np.float32), config)
        histogram = (
            np.bincount(crop, minlength=256)
            if config.mode in HISTOGRAM_TABLE_MODES
            else None
        )
        table = uint8_normalization_table(histogram, config)
        assert table.dtype == np.float32 and table.shape == (256,)
        assert np.array_equal(table[crop], expected), (crop.size, config.mode)
        checked += 1
    assert checked > 40


def test_uint8_normalization_table_rejects_order_modes_and_bad_histograms():
    from vesuvius.ink_detection.config import NormalizationConfig
    from vesuvius.ink_detection.data.normalization import uint8_normalization_table

    for mode in ("robust_mad", "robust_percentile_span"):
        with pytest.raises(ValueError, match="no uint8 normalization table"):
            uint8_normalization_table(
                np.ones(256, dtype=np.int64), NormalizationConfig.from_value(mode)
            )
    with pytest.raises(ValueError, match="256-bin"):
        uint8_normalization_table(
            np.ones(255, dtype=np.int64),
            NormalizationConfig.from_value("percentile_minmax"),
        )


class _ArrayVolume:
    """In-memory volume with vc_reader's single and batched read calls."""

    def __init__(self, values):
        self.values = values
        self.shape = values.shape
        self.dtype = values.dtype

    def __getitem__(self, key):
        return self.values[key]

    def read_regions(self, regions, out):
        for (start, size), destination in zip(regions, out):
            destination[...] = self.values[
                tuple(slice(a, a + n) for a, n in zip(start, size))
            ]
        return out


def _paired_datasets(monkeypatch, normalization, values, volume_path="local.zarr"):
    from vesuvius.ink_detection.config import NormalizationConfig

    volume = _ArrayVolume(values)
    monkeypatch.setattr(native, "_open_shared_volume", lambda *args, **kwargs: volume)
    config = SimpleNamespace(
        data=SimpleNamespace(
            mode="full_3d",
            normalization=(
                normalization
                if isinstance(normalization, NormalizationConfig)
                else NormalizationConfig.from_value(normalization)
            ),
        )
    )
    patches = native.build_patch_plan_for_chunks(
        target_chunks=[native.ChunkId(0, 0, 0), native.ChunkId(1, 1, 1)],
        array_shape_zyx=values.shape,
        chunk_shape_zyx=(10, 12, 14),
        patch_size_zyx=(8, 8, 8),
        stride_zyx=(4, 4, 4),
    )
    # padded at both ends, then wholly outside the volume
    patches = [
        *patches,
        native.PatchSpec(-3, 20, 25),
        native.PatchSpec(-2, -5, -7),
        native.PatchSpec(30, 0, 0),
    ]
    common = dict(
        tifxyz_dir=Path("unused"),
        volume_path=volume_path,
        resolution="0",
        patches=patches,
        patch_size_zyx=(8, 8, 8),
        config=config,
    )
    host = native.NativePatchDataset(**common)
    device = native.NativePatchDataset(**common, device_normalize=True)
    return host, device, patches


# normalize_on_device must hand the model the very float32 batch the host
# path builds, including padded and wholly outside patches, so the model
# output and every written code stay the same
@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA"),
        ),
    ],
)
@pytest.mark.parametrize("normalization", _TABLE_NORMALIZATIONS)
@pytest.mark.parametrize("volume_path", ["local.zarr", "s3://bucket/remote.zarr"])
def test_device_normalize_matches_host_normalize(
    monkeypatch, normalization, device, volume_path
):
    rng = np.random.default_rng(0)
    values = np.clip(rng.normal(90, 40, (20, 24, 28)), 0, 255).astype(np.uint8)
    values[:4] = 0  # a zero slab: in-volume zeros differ from padding
    host, raw, patches = _paired_datasets(monkeypatch, normalization, values, volume_path)
    expected = torch.stack([host[index][0] for index in range(len(patches))])
    assert expected.abs().sum() > 0

    sample, metadata = raw[len(patches) - 2]
    assert sample["raw"].dtype == torch.uint8 and sample["raw"].shape == (8, 8, 8)
    assert sample["valid_box"].tolist() == [2, 5, 7, 8, 8, 8]
    assert metadata.tolist() == [-2, -5, -7]
    assert raw[len(patches) - 1][0]["valid_box"].tolist() == [0] * 6

    loader = torch.utils.data.DataLoader(raw, batch_size=3)
    batches = [native.normalize_on_device(batch, torch.device(device)) for batch, _ in loader]
    assert all(batch.device.type == device for batch in batches)
    actual = torch.cat([batch.cpu() for batch in batches])
    assert actual.dtype == expected.dtype == torch.float32
    assert actual.shape == expected.shape
    assert torch.equal(actual, expected)


def test_device_normalize_rejects_unsupported_inputs(monkeypatch):
    from vesuvius.ink_detection.config import NormalizationConfig

    values = np.zeros((20, 24, 28), dtype=np.uint8)
    _, raw, _ = _paired_datasets(monkeypatch, "percentile_minmax", values)
    raw._volume = _ArrayVolume(values.astype(np.uint16))
    with pytest.raises(TypeError, match="uint8 volume"):
        raw[0]

    for mode, normalization in (
        ("full_3d", "robust_mad"),
        ("full_3d_single_wrap", "percentile_minmax"),
    ):
        config = SimpleNamespace(
            data=SimpleNamespace(
                mode=mode, normalization=NormalizationConfig.from_value(normalization)
            )
        )
        with pytest.raises(ValueError, match="--normalize-device cuda needs"):
            native.NativePatchDataset(
                tifxyz_dir=Path("unused"),
                volume_path="local.zarr",
                resolution="0",
                patches=(native.PatchSpec(0, 0, 0),),
                patch_size_zyx=(8, 8, 8),
                config=config,
                device_normalize=True,
            )

    bundle = native.NativeModelBundle(
        model=torch.nn.Identity(), device=torch.device("cpu"), amp_dtype=None
    )
    with pytest.raises(ValueError, match="needs a CUDA model device"):
        native.run_native_inference(
            args=SimpleNamespace(
                batch_size=1, gpu_ids=[], num_workers=0, tta=False, normalize_device="cuda"
            ),
            bundle=bundle,
            dataset=raw,
            plan=SimpleNamespace(),
            output=np.zeros((2, 2, 2), dtype=np.uint8),
            importance_map=np.ones((2, 2, 2), dtype=np.float32),
        )


def test_cli_normalize_device_flag():
    assert native.parse_args(["t", "c.pth", "o.zarr"]).normalize_device == "cpu"
    args = native.parse_args(["t", "c.pth", "o.zarr", "--normalize_device", "cuda"])
    assert args.normalize_device == "cuda"
    with pytest.raises(SystemExit):
        native.parse_args(["t", "c.pth", "o.zarr", "--normalize-device", "mps"])


def _fused_reference_normalize(crop, config):
    """vc_reader's percentile normalizer as a build with fused multiply-adds computes it."""

    values = np.sort(crop.astype(np.float64).reshape(-1))
    count = values.size

    def percentile(percent):
        q = percent / 100
        position = math.fma(float(count), q, 1 + q * -1.0) - 1
        if position >= count - 1:
            return values[-1]
        below = math.floor(position)
        fraction = position - below
        low, high = values[below], values[below + 1]
        span = high - low
        if fraction >= 0.5:
            return math.fma(-span, 1 - fraction, high)
        return math.fma(span, fraction, low)

    low = percentile(config.percentile_lower)
    high = percentile(config.percentile_upper)
    clamped = np.clip(crop.astype(np.float32), np.float32(low), np.float32(high))
    return ((clamped.astype(np.float64) - low) / (high - low)).astype(np.float32)


# A vc_reader build whose percentile arithmetic is fused must be refused, or
# device inputs would drift by an ulp from the host path's
@pytest.mark.skipif(not hasattr(math, "fma"), reason="needs math.fma")
@pytest.mark.parametrize("bounds", [(1.0, 99.0), (35.01, 59.01), (0.5, 99.5)])
def test_device_normalize_refuses_a_fused_normalizer(monkeypatch, bounds):
    from vesuvius.ink_detection.config import NormalizationConfig
    from vesuvius.ink_detection.data import normalization

    config = NormalizationConfig.from_value(
        {"mode": "percentile_minmax", "percentile_lower": bounds[0], "percentile_upper": bounds[1]}
    )
    normalization.uint8_table_matches_normalizer.cache_clear()
    assert normalization.uint8_table_matches_normalizer(config)
    crops = normalization._fma_sensitive_crops((35.01, 59.01)) + normalization._fma_sensitive_crops(bounds)
    assert crops
    assert any(
        not np.array_equal(
            _fused_reference_normalize(crop, config),
            normalization.normalize_image(crop.astype(np.float32), config),
        )
        for crop in crops
    )

    monkeypatch.setattr(normalization, "normalize_image", _fused_reference_normalize)
    normalization.uint8_table_matches_normalizer.cache_clear()
    try:
        assert not normalization.uint8_table_matches_normalizer(config)
        values = np.zeros((20, 24, 28), dtype=np.uint8)
        with pytest.raises(RuntimeError, match="fused multiply-adds"):
            _paired_datasets(monkeypatch, config, values)
    finally:
        normalization.uint8_table_matches_normalizer.cache_clear()


def _write_volume(path, values, chunks):
    from vc_reader.reader import ZarrWriter

    array = ZarrWriter.create(path, shape=values.shape, chunks=chunks, dtype="u1")
    array[:] = values
    return path


def _plan_for_patches(patches, shape, chunks, patch):
    """A NativePlan-like view: every chunk a patch touches is a target."""

    counts = {}
    for spec in patches:
        starts = (spec.z0, spec.y0, spec.x0)
        lows = [max(0, start) // size for start, size in zip(starts, chunks)]
        highs = [
            (min(extent, start + size_patch) - 1) // size
            for start, size_patch, extent, size in zip(starts, patch, shape, chunks)
        ]
        if any(
            start + size_patch <= 0 or start >= extent
            for start, size_patch, extent in zip(starts, patch, shape)
        ):
            continue
        for chunk in itertools.product(*(range(a, b + 1) for a, b in zip(lows, highs))):
            counts[native.ChunkId(*chunk)] = counts.get(native.ChunkId(*chunk), 0) + 1
    return SimpleNamespace(
        target_chunks=frozenset(counts),
        contribution_counts=counts,
        chunk_shape_zyx=chunks,
        patch_size_zyx=patch,
    )


def _settings():
    cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
    return [
        pytest.param("cpu", None, False, True, "cpu", id="cpu-tta"),
        pytest.param("cuda", torch.bfloat16, False, True, "cuda", id="cuda-amp-tta", marks=cuda),
        pytest.param("cuda", torch.bfloat16, True, False, "cuda", id="cuda-amp-compiled", marks=cuda),
        pytest.param("cuda", torch.bfloat16, True, False, "cpu", id="cuda-compiled-host-blend", marks=cuda),
    ]


# Through prediction and accumulation, with spawned workers reading a real
# zarr, a partial final batch, TTA or reduce-overhead compilation and AMP:
# the device input path must write the host path's uint8 codes exactly
@pytest.mark.parametrize("device, amp_dtype, compiled, tta, blend_device", _settings())
def test_device_normalize_inference_writes_host_output(
    tmp_path, device, amp_dtype, compiled, tta, blend_device
):
    from vesuvius.ink_detection.config import NormalizationConfig

    rng = np.random.default_rng(1)
    shape, chunks, patch = (20, 24, 28), (10, 12, 14), (8, 8, 8)
    values = np.clip(rng.normal(90, 40, shape), 0, 255).astype(np.uint8)
    values[:4] = 0
    volume = _write_volume(tmp_path / "input.zarr", values, chunks)
    patches = [
        *native.build_patch_plan_for_chunks(
            target_chunks=[native.ChunkId(0, 0, 0), native.ChunkId(1, 1, 1)],
            array_shape_zyx=shape,
            chunk_shape_zyx=chunks,
            patch_size_zyx=patch,
            stride_zyx=(4, 4, 4),
        ),
        native.PatchSpec(-3, 20, 25),
        native.PatchSpec(-2, -5, -7),
    ]
    batch_size = 3 if len(patches) % 3 else 4
    assert len(patches) % batch_size  # a partial final batch
    plan = _plan_for_patches(patches, shape, chunks, patch)
    config = SimpleNamespace(
        data=SimpleNamespace(
            mode="full_3d", normalization=NormalizationConfig.from_value("percentile_minmax")
        )
    )
    torch.manual_seed(0)
    model = torch.nn.Conv3d(1, 1, 3, padding=1).to(device).eval()
    if compiled:
        model = torch.compile(model, mode="reduce-overhead")
    bundle = native.NativeModelBundle(model=model, device=torch.device(device), amp_dtype=amp_dtype)
    outputs = []
    for device_normalize in (False, True):
        dataset = native.NativePatchDataset(
            tifxyz_dir=Path("unused"),
            volume_path=str(volume),
            resolution="0",
            patches=patches,
            patch_size_zyx=patch,
            config=config,
            device_normalize=device_normalize,
        )
        output = np.zeros(shape, dtype=np.uint8)
        native.run_native_inference(
            args=SimpleNamespace(
                batch_size=batch_size,
                gpu_ids=[],
                num_workers=2,
                prefetch_factor=2,
                tta=tta,
                tta_batch_size=None,
                blend_device=blend_device,
                normalize_device="cuda" if device_normalize and device == "cuda" else "cpu",
            ),
            bundle=bundle,
            dataset=dataset,
            plan=plan,
            output=output,
            importance_map=native.create_importance_map(patch, mode="gaussian"),
        )
        outputs.append(output)
    assert len(np.unique(outputs[0])) > 10
    assert np.array_equal(outputs[0], outputs[1])

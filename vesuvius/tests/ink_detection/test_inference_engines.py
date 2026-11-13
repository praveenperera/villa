from __future__ import annotations

import gc
import json
import sys
import threading
import weakref
from types import SimpleNamespace

import numpy as np
import pytest
import tifffile
import torch
from vc_reader.reader import ZarrWriter, create_group

from vesuvius.ink_detection.config import InkConfig
from vesuvius.ink_detection.inference import engine_parity
from vesuvius.ink_detection.inference import engines
from vesuvius.ink_detection.inference.engines import (
    EngineUnavailable,
    FlatEngineResult,
    InkrsEngine,
    WindowCache,
    select_engine,
)
from vesuvius.ink_detection.inference.infer import (
    main,
    normalize_inference_paths,
    parse_args,
    run,
)
from vesuvius.ink_detection.models.model import make_model

from .helpers import config_mapping as make_config_mapping

DEPTH = 3
SIDE = 16


def _save_checkpoint(path, *, side=SIDE):
    mapping = make_config_mapping("vesuvius_unet_2p5d", depth=DEPTH, side=side)
    mapping["image_normalization"] = "robust_mad"
    torch.manual_seed(0)
    model = make_model(InkConfig.from_mapping(mapping))
    torch.save({"config": mapping, "model": model.state_dict()}, path)
    return path


def _write_surface(path, *, layers=5, height=40, width=48):
    volume = np.random.default_rng(0).integers(1, 255, size=(layers, height, width))
    # an all-zero corner exercises the empty-tile skip and partial coverage
    volume[:, :24, :24] = 0
    array = ZarrWriter.create(
        path, shape=volume.shape, chunks=(layers, 16, 16), dtype="u1"
    )
    array[:] = volume
    return path, volume.astype(np.uint8)


COMMON = ["--workers", "0", "--no-compile", "--blend-mode", "hann", "--batch-size", "2"]


def test_stub_engine_through_villa_plumbing_matches_pytorch_exactly(tmp_path):
    surface, _ = _write_surface(tmp_path / "surface.zarr")
    checkpoint = _save_checkpoint(tmp_path / "model.pth")
    out = tmp_path / "parity"

    status = engine_parity.main(
        [
            str(surface),
            str(checkpoint),
            str(out),
            "--candidate",
            "torch-stub",
            "--layer-starts",
            "0,1",
            "--",
            *COMMON,
            "--direction",
            "both",
        ]
    )

    report = json.loads((out / "parity.json").read_text())
    assert status == 0 and report["pass"]
    assert [entry["map"] for entry in report["maps"]] == [
        "L0.tif",
        "L0_reverse.tif",
        "L1.tif",
        "L1_reverse.tif",
    ]
    assert [entry["layer_indices"] for entry in report["maps"]] == [
        [0, 1, 2],
        [2, 1, 0],
        [1, 2, 3],
        [3, 2, 1],
    ]
    for entry in report["maps"]:
        assert entry["identical"]
        # only the 16 x 16 corner lies under skipped tiles alone
        assert entry["covered_pixels"] == 40 * 48 - 16 * 16
    reference = tifffile.imread(out / "torch" / "L0_reverse.tif")
    assert reference.any()
    assert not np.array_equal(reference, tifffile.imread(out / "torch" / "L0.tif"))


def test_stand_in_keeps_villas_occupancy_skip_and_the_gate_catches_losing_it(
    tmp_path, monkeypatch
):
    _, volume = _write_surface(tmp_path / "bare.zarr")
    level3 = np.zeros((5, 6), dtype=np.uint8)
    level3[1:4, 2:5] = 1
    surface = _write_group(tmp_path / "group.zarr", volume, level3)
    checkpoint = _save_checkpoint(tmp_path / "model.pth")

    def parity(out):
        status = engine_parity.main(
            [str(surface), str(checkpoint), str(out), "--candidate", "torch-stub",
             "--layer-starts", "1", "--", *COMMON]
        )
        return status, json.loads((out / "parity.json").read_text())["maps"][0]

    status, entry = parity(tmp_path / "kept")
    assert status == 0 and entry["identical"]
    # the level-3 mask drops tiles that the bare array would run
    assert entry["covered_pixels"] < 40 * 48 - 16 * 16

    from vesuvius.ink_detection.inference import engines

    real = engines._inkrs_infer
    monkeypatch.setattr(
        engines,
        "_inkrs_infer",
        lambda module, handle, window, occupancy, *, stride: real(
            module, handle, window, None, stride=stride
        ),
    )
    status, entry = parity(tmp_path / "lost")
    assert status == 1
    assert entry["coverage_mismatch_pixels"] > 0


def _gate_fixture():
    rng = np.random.default_rng(1)
    coverage = np.zeros((64, 64), dtype=bool)
    coverage[8:56, 4:60] = True
    reference = np.where(coverage, rng.integers(1, 255, size=(64, 64)), 0).astype(np.uint8)
    return reference, coverage


def test_parity_gate_passes_small_errors_and_fails_each_mutation():
    reference, coverage = _gate_fixture()
    noisy = reference.astype(np.int16)
    noisy[coverage] += np.random.default_rng(2).integers(-3, 4, size=int(coverage.sum()))
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)

    def gate(candidate, candidate_coverage=coverage):
        return engine_parity.compare_maps(
            reference,
            candidate,
            reference_coverage=coverage,
            candidate_coverage=candidate_coverage,
        )

    assert gate(noisy)["pass"]

    nothing = np.zeros_like(coverage)
    empty = engine_parity.compare_maps(
        np.zeros_like(reference),
        np.zeros_like(reference),
        reference_coverage=nothing,
        candidate_coverage=nothing,
    )
    assert not empty["pass"]
    assert "no covered pixels to compare" in empty["failures"]

    # every pixel is within 1/255, so only the correlation can catch the flip
    alternating = np.where(coverage, 127 + (np.indices(coverage.shape).sum(0) % 2), 0).astype(np.uint8)
    flipped = np.where(coverage, 255 - alternating, 0).astype(np.uint8)
    report = engine_parity.compare_maps(
        alternating, flipped, reference_coverage=coverage, candidate_coverage=coverage
    )
    assert report["max_abs_diff"] == 1 and report["pearson"] < -0.99
    assert [failure.split()[0] for failure in report["failures"]] == ["pearson"]

    zeroed = reference.copy()
    zeroed[20:24, 20:24] = 0
    dropped = coverage.copy()
    dropped[30, 30] = False
    added = coverage.copy()
    added[0, 0] = True
    stray = reference.copy()
    stray[0, 0] = 7
    shifted = np.clip(reference.astype(np.int16) + 5, 0, 255).astype(np.uint8) * coverage

    assert not gate(zeroed)["pass"]
    assert not gate(reference, dropped)["pass"]
    assert not gate(reference, added)["pass"]
    assert not gate(stray)["pass"]
    assert not gate(shifted)["pass"]


def _inkrs_shaped_model(**overrides):
    values = dict(
        amp_dtype=torch.float16,
        preprocessing="tifxyz_robust",
        input_depth=17,
        patch_size=128,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def _engine_args(tmp_path, *extra):
    return normalize_inference_paths(
        parse_args(["in.zarr", str(tmp_path / "m.pth"), "out.tif", "--engine", "inkrs", *extra])
    )


def test_inkrs_missing_falls_back_or_stops_with_the_reason(tmp_path, monkeypatch, caplog):
    monkeypatch.setitem(sys.modules, "inkrs", None)
    device = torch.device("cuda", 0)

    with caplog.at_level("WARNING"):
        engine = select_engine(_engine_args(tmp_path), _inkrs_shaped_model(), device)
    assert engine is None
    assert "inkrs package is not installed" in caplog.text
    assert "Continuing with PyTorch" in caplog.text

    with pytest.raises(EngineUnavailable, match="not installed"):
        select_engine(
            _engine_args(tmp_path, "--engine-fallback", "error"),
            _inkrs_shaped_model(),
            device,
        )


def test_inkrs_refuses_unsupported_models_and_options_before_importing(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "inkrs", None)
    strict = _engine_args(tmp_path, "--engine-fallback", "error", "--tta-mirror")

    with pytest.raises(EngineUnavailable) as raised:
        select_engine(
            strict,
            _inkrs_shaped_model(amp_dtype=torch.bfloat16),
            torch.device("cpu"),
        )

    message = str(raised.value)
    for reason in ("CUDA device", "--tta-mirror", "fp16 autocast"):
        assert reason in message
    assert "not installed" not in message


@pytest.mark.parametrize(
    ("checkpoint", "extra", "reason"),
    [
        ("m.pth", ("--roi", "0", "8", "0", "8"), "it does not crop to --roi"),
        ("m.pth", ("--mask-path", "mask.png"), "it does not apply --mask-path"),
        ("m.pth", ("--blend-mode", "gaussian"), "it blends with Hann only, got 'gaussian'"),
        # inkrs would read any other path as its own safetensors export and fail, not decline
        ("m.pt", (), "it loads .pth checkpoints only, got m.pt"),
    ],
)
def test_inkrs_declines_whole_commands_without_loading(tmp_path, monkeypatch, caplog, checkpoint, extra, reason):
    # these hold for every input, so loading inkrs for them would only take GPU memory PyTorch needs
    def never_load(checkpoint, *, device):
        raise AssertionError("inkrs loaded for a command it declines")

    monkeypatch.setitem(sys.modules, "inkrs", _fake_inkrs(never_load))
    args = normalize_inference_paths(
        parse_args(["in.zarr", str(tmp_path / checkpoint), "out.tif", "--engine", "inkrs", *extra])
    )
    with caplog.at_level("WARNING"):
        engine = select_engine(args, _inkrs_shaped_model(), torch.device("cuda", 0))
    assert engine is None
    assert reason in caplog.text


def test_cpu_run_requesting_inkrs_writes_the_pytorch_map(tmp_path, monkeypatch, caplog):
    from vesuvius.ink_detection.inference import infer

    surface, _ = _write_surface(tmp_path / "surface.zarr")
    checkpoint = _save_checkpoint(tmp_path / "model.pth")
    # inkrs importable, so only the CPU refusal can stop the read ahead
    monkeypatch.setitem(sys.modules, "inkrs", _fake_inkrs(None))
    reads = []
    monkeypatch.setattr(infer, "read_span", lambda *a: reads.append(a))

    assert main([str(surface), str(checkpoint), str(tmp_path / "torch.tif"), *COMMON, "--device", "cpu"]) == 0
    with caplog.at_level("WARNING"):
        assert (
            main(
                [str(surface), str(checkpoint), str(tmp_path / "inkrs.tif"), *COMMON, "--device", "cpu", "--engine", "inkrs"]
            )
            == 0
        )
    assert "Continuing with PyTorch" in caplog.text
    np.testing.assert_array_equal(
        tifffile.imread(tmp_path / "inkrs.tif"), tifffile.imread(tmp_path / "torch.tif")
    )
    # a command inkrs refuses reads nothing ahead for it
    assert reads == []

    with pytest.raises(EngineUnavailable):
        main(
            [
                str(surface),
                str(checkpoint),
                str(tmp_path / "strict.tif"),
                *COMMON, "--device", "cpu",
                "--engine",
                "inkrs",
                "--engine-fallback",
                "error",
            ]
        )


class _RecordingHandle:
    """Stands in for inkrs.Engine and records what villa sends to run()."""

    def __init__(self):
        self.calls = []

    def run(self, volume, windows, *, stride, occupancy, coverage):
        self.calls.append((volume.copy(), windows, stride, occupancy, coverage))
        height, width = volume.shape[1:]
        probability = np.full((height, width), 9, dtype=np.uint8)
        return [(probability, np.ones((height, width), dtype=np.uint8))]


def _write_group(path, volume, level3):
    group = create_group(path)
    group.create_array("0", shape=volume.shape, chunks=(volume.shape[0], 16, 16), dtype="u1")[:] = volume
    group.create_array("3", shape=level3.shape, chunks=level3.shape, dtype="u1")[:] = level3
    return path


class _InkrsError(RuntimeError):
    pass


class _UnsupportedError(_InkrsError):
    pass


def _fake_inkrs(engine):
    return SimpleNamespace(
        InkrsError=_InkrsError, UnsupportedError=_UnsupportedError, Engine=engine, __version__="0"
    )


def test_inkrs_load_falls_back_only_when_inkrs_calls_it_unsupported(
    tmp_path, monkeypatch, caplog
):
    def refuse(checkpoint, *, device):
        raise _UnsupportedError("checkpoint is not the hybrid_3d2d-seed42 contract")

    monkeypatch.setitem(sys.modules, "inkrs", _fake_inkrs(refuse))
    with caplog.at_level("WARNING"):
        engine = select_engine(
            _engine_args(tmp_path), _inkrs_shaped_model(), torch.device("cuda", 0)
        )
    assert engine is None
    assert "inkrs cannot load m.pth on device 0" in caplog.text
    assert "hybrid_3d2d-seed42 contract" in caplog.text

    def broken(checkpoint, *, device):
        raise _InkrsError("no cuDNN 9 found")

    # a broken install is a real failure, not a reason to run quietly on PyTorch
    monkeypatch.setitem(sys.modules, "inkrs", _fake_inkrs(broken))
    with pytest.raises(_InkrsError, match="cuDNN"):
        select_engine(_engine_args(tmp_path), _inkrs_shaped_model(), torch.device("cuda", 0))


def test_input_inkrs_calls_unsupported_runs_on_pytorch_and_other_errors_raise(tmp_path, caplog):
    surface, _ = _write_surface(tmp_path / "surface.zarr")
    checkpoint = _save_checkpoint(tmp_path / "model.pth")
    assert main([str(surface), str(checkpoint), str(tmp_path / "torch.tif"), *COMMON]) == 0

    class _Handle:
        def __init__(self, error):
            self.error = error

        def run(self, volume, windows, *, stride, occupancy, coverage):
            raise self.error

    def run_with(error, output, *extra):
        args = normalize_inference_paths(
            parse_args([str(surface), str(checkpoint), str(tmp_path / output), *COMMON, *extra])
        )
        engine = InkrsEngine(_fake_inkrs(None), _Handle(error), device_index=0)
        return run(args, engine_factory=lambda *_: engine)

    with caplog.at_level("WARNING"):
        run_with(_UnsupportedError("a volume of 4 GiB or more"), "fallback.tif")
    assert "a volume of 4 GiB or more. Running this input with PyTorch" in caplog.text
    np.testing.assert_array_equal(
        tifffile.imread(tmp_path / "fallback.tif"), tifffile.imread(tmp_path / "torch.tif")
    )

    with pytest.raises(EngineUnavailable, match="4 GiB"):
        run_with(_UnsupportedError("a volume of 4 GiB or more"), "strict.tif", "--engine-fallback", "error")
    with pytest.raises(_InkrsError, match="out of memory"):
        run_with(_InkrsError("CUDA out of memory"), "oom.tif")
    assert not (tmp_path / "strict.tif").exists() and not (tmp_path / "oom.tif").exists()


def test_stand_in_and_adapter_match_the_installed_inkrs_signatures():
    inkrs = pytest.importorskip("inkrs")
    import inspect

    assert issubclass(inkrs.UnsupportedError, inkrs.InkrsError)
    real_init = inspect.signature(inkrs.Engine.__init__).parameters
    real_run = inspect.signature(inkrs.Engine.run).parameters
    # the adapter's calls: Engine(checkpoint, device=) and run(volume, windows, stride=, occupancy=, coverage=)
    assert {"checkpoint", "device"} <= real_init.keys()
    assert {"volume", "windows", "stride", "occupancy", "coverage"} <= real_run.keys()
    stand_in_run = inspect.signature(engine_parity.TorchStandInEngine.run).parameters
    for name in ("stride", "occupancy", "coverage", "outputs"):
        assert stand_in_run[name].default == real_run[name].default, name


@pytest.mark.parametrize("pyramid", [False, True])
def test_inkrs_adapter_sends_layers_in_model_order_and_villas_occupancy(tmp_path, pyramid):
    surface, volume = _write_surface(tmp_path / "surface.zarr")
    level3 = np.zeros((5, 6), dtype=np.uint8)
    level3[1:4, 2:5] = 1
    if pyramid:
        surface = _write_group(tmp_path / "group.zarr", volume, level3)
    checkpoint = _save_checkpoint(tmp_path / "model.pth")
    handle = _RecordingHandle()
    engine = InkrsEngine(_fake_inkrs(None), handle, device_index=0)
    args = normalize_inference_paths(
        parse_args(
            [
                str(surface),
                str(checkpoint),
                str(tmp_path / "out.tif"),
                *COMMON,
                "--layer-start",
                "1",
                "--layer-end",
                "4",
                "--direction",
                "reverse",
            ]
        )
    )

    assert run(args, engine_factory=lambda *_: engine) == 0

    [(window, windows, stride, occupancy, coverage)] = handle.calls
    np.testing.assert_array_equal(window, volume[[3, 2, 1]])
    assert window.flags.c_contiguous and window.dtype == np.uint8
    assert windows == [(0, 1)] and stride == SIDE // 2 and coverage is True
    if pyramid:
        # inkrs scales villa's own low-resolution mask by round(40 / 5) = 8
        np.testing.assert_array_equal(occupancy, level3 != 0)
    else:
        # a bare array has no pyramid, so villa scans nothing and neither may inkrs
        assert occupancy is None
    assert (tifffile.imread(tmp_path / "out.tif") == 9).all()


def test_both_directions_read_the_source_layers_once_beside_the_engine_load(
    tmp_path, monkeypatch
):
    from vesuvius.ink_detection.inference import infer

    surface, volume = _write_surface(tmp_path / "surface.zarr")
    checkpoint = _save_checkpoint(tmp_path / "model.pth")
    reads = []
    real_read = engines.read_span
    read_started, load_done = threading.Event(), threading.Event()

    def recording_read(*a):
        reads.append((threading.current_thread().name, a[2:]))
        read_started.set()
        # the read finishes only once the engine has loaded, so a run that read
        # before or after the load, not during it, times out here or below
        assert load_done.wait(10), "the engine never loaded while the read waited"
        return real_read(*a)

    def load(*_):
        assert read_started.wait(10), "the read did not start before the engine load"
        load_done.set()
        return engine

    monkeypatch.setattr(engines, "read_span", recording_read)
    monkeypatch.setattr(infer, "read_span", recording_read)
    handle = _RecordingHandle()
    engine = InkrsEngine(_fake_inkrs(None), handle, device_index=0)
    args = normalize_inference_paths(
        parse_args(
            [str(surface), str(checkpoint), str(tmp_path / "out.tif"), *COMMON,
             "--layer-start", "1", "--layer-end", "4", "--direction", "both"]
        )
    )

    assert run(args, engine_factory=load) == 0

    # one read, on a thread, spanning the engine load
    assert reads == [("engine-window-prefetch", (1, 4))]
    assert not any(t.name == "engine-window-prefetch" for t in threading.enumerate())
    [forward, reverse] = [call[0] for call in handle.calls]
    np.testing.assert_array_equal(forward, volume[[1, 2, 3]])
    np.testing.assert_array_equal(reverse, volume[[3, 2, 1]])


def test_window_read_ahead_only_for_runs_inkrs_would_take(tmp_path, monkeypatch):
    from vesuvius.ink_detection.inference.infer import engine_window_wanted

    model, cuda = _inkrs_shaped_model(), torch.device("cuda", 0)

    def wanted(*extra, device=cuda):
        return engine_window_wanted(_engine_args(tmp_path, *extra), model, device, None)

    monkeypatch.setitem(sys.modules, "inkrs", _fake_inkrs(None))
    assert wanted("--blend-mode", "hann", "--stride", "64")
    # inkrs's macOS build runs on Metal through villa's MPS device
    assert wanted("--blend-mode", "hann", "--stride", "64", device=torch.device("mps"))
    # every refusal inkrs or villa would make before a request leaves the read undone
    assert not wanted("--blend-mode", "constant", "--stride", "64")
    assert not wanted("--blend-mode", "hann", "--stride", "999")
    assert not wanted("--blend-mode", "hann", "--roi", "0", "1", "0", "1")
    assert not wanted("--blend-mode", "hann", "--tta-mirror")
    assert not wanted("--blend-mode", "hann", device=torch.device("cpu"))
    monkeypatch.setitem(sys.modules, "inkrs", None)
    assert not wanted("--blend-mode", "hann", "--stride", "64")


def test_inkrs_declines_windows_it_cannot_take_before_reading_them(tmp_path, monkeypatch, caplog):
    # inkrs checks dtype, size and the 4 GiB limit only once it has the window in memory, so villa
    # applies the same limits from metadata first; an oversized input must never be read whole
    from vesuvius.ink_detection.inference import infer
    from vesuvius.ink_detection.inference.engines import FlatEngineRequest

    reads = []
    real_read = engines.read_span

    def recording_read(*a):
        reads.append(a[2:])
        return real_read(*a)

    monkeypatch.setattr(engines, "read_span", recording_read)
    monkeypatch.setattr(infer, "read_span", recording_read)
    handle = _RecordingHandle()
    engine = InkrsEngine(_fake_inkrs(None), handle, device_index=0)
    checkpoint = _save_checkpoint(tmp_path / "model.pth")

    # a uint16 input runs on PyTorch with neither the read ahead nor the request reading it
    _, volume = _write_surface(tmp_path / "u8.zarr")
    wide = tmp_path / "u16.zarr"
    ZarrWriter.create(wide, shape=volume.shape, chunks=(volume.shape[0], 16, 16), dtype="u2")[:] = (
        volume.astype(np.uint16) * 200
    )
    assert main([str(wide), str(checkpoint), str(tmp_path / "torch.tif"), *COMMON]) == 0
    args = normalize_inference_paths(parse_args([str(wide), str(checkpoint), str(tmp_path / "engine.tif"), *COMMON]))
    with caplog.at_level("WARNING"):
        assert run(args, engine_factory=lambda *_: engine) == 0
    assert "it reads uint8 volumes only, got uint16" in caplog.text
    assert reads == [] and handle.calls == []
    np.testing.assert_array_equal(
        tifffile.imread(tmp_path / "engine.tif"), tifffile.imread(tmp_path / "torch.tif")
    )

    # 3 x 40000 x 36000 bytes is just over 4 GiB; with no chunk written it takes no disk
    big = ZarrWriter.create(tmp_path / "big.zarr", shape=(DEPTH, 40000, 36000), chunks=(DEPTH, 1024, 1024), dtype="u1")
    request = FlatEngineRequest(
        input_zarr=tmp_path / "big.zarr", resolution="0", volume=big, depth_axis_first=True,
        height=40000, width=36000, layer_indices=np.arange(DEPTH), input_depth=DEPTH,
        patch_size=SIDE, stride=SIDE // 2, blend_mode="hann", preprocessing="tifxyz_robust", blocks=(),
    )
    assert engine.unsupported_reason(request) == "its 3-layer window of 4320000000 bytes is 4 GiB or more"
    # a 2-layer input is shallower than the 3-layer model, so the engine declines it too
    _write_surface(tmp_path / "shallow.zarr", layers=2)
    model = _inkrs_shaped_model(input_depth=DEPTH, patch_size=SIDE)
    for surface in ("big.zarr", "shallow.zarr", "u8.zarr"):
        cache = WindowCache()
        args = normalize_inference_paths(
            parse_args([str(tmp_path / surface), str(checkpoint), "out.tif", "--engine", "inkrs"])
        )
        infer.prefetch_engine_window(args, model, cache)
        cache.close()
    # only the small uint8 input was read ahead
    assert reads == [(1, 4)]
    narrow = SimpleNamespace(dtype=np.dtype("u1"))
    assert engines.inkrs_window_reason(narrow, DEPTH, 40, SIDE - 1, SIDE) == (
        "the input's 40x15 layers are smaller than one 16 px tile"
    )
    assert engines.inkrs_window_reason(narrow, DEPTH, SIDE, SIDE, SIDE) is None


def test_folder_runs_load_the_engine_once_and_fall_back_per_segment(tmp_path, monkeypatch):
    # one uint8 segment runs on the engine and one uint16 segment falls back to PyTorch, in one folder run
    _, volume = _write_surface(tmp_path / "u8.zarr")
    folder = tmp_path / "segments"
    _write_surface(folder / "a" / "a.zarr")
    (folder / "b").mkdir(parents=True)
    ZarrWriter.create(folder / "b" / "b.zarr", shape=volume.shape, chunks=(volume.shape[0], 16, 16), dtype="u2")[:] = (
        volume.astype(np.uint16) * 200
    )
    checkpoint = _save_checkpoint(tmp_path / "model.pth")
    loads, handle = [], _RecordingHandle()

    def factory(*_):
        loads.append(1)
        return InkrsEngine(_fake_inkrs(None), handle, device_index=0)

    args = normalize_inference_paths(parse_args(["--folder", str(folder), str(checkpoint), *COMMON, "--direction", "both"]))
    assert run(args, engine_factory=factory) == 0
    assert main([str(folder / "b" / "b.zarr"), str(checkpoint), str(tmp_path / "torch.tif"), *COMMON]) == 0

    assert loads == [1]
    # segment a's two directions went to the engine, in model order, and b's to neither
    [forward, reverse] = [call[0] for call in handle.calls]
    np.testing.assert_array_equal(reverse, forward[::-1])
    for direction in ("forward", "reverse"):
        [a_map] = (folder / "a" / "preds").glob(f"a_model_{direction}_*.tif")
        assert (tifffile.imread(a_map) == 9).all()
    [b_forward] = (folder / "b" / "preds").glob("b_model_forward_*.tif")
    np.testing.assert_array_equal(tifffile.imread(b_forward), tifffile.imread(tmp_path / "torch.tif"))


def test_window_cache_keeps_one_span_and_never_a_failed_read():
    cache = WindowCache()
    calls = []

    def read(value):
        def inner():
            calls.append(value)
            return np.full((1, 2, 2), value, np.uint8)
        return inner

    cache.prefetch(("a", "0", True, 0, 1), read(1))
    first = cache.span(("a", "0", True, 0, 1), read(2))
    assert cache.span(("a", "0", True, 0, 1), read(2)) is first
    # the shared span is read-only, so one engine cannot change another's input
    assert not first.flags.writeable
    assert (cache.span(("b", "0", True, 0, 1), read(3)) == 3).all()
    assert (cache.span(("a", "0", True, 0, 1), read(4)) == 4).all()
    assert calls == [1, 3, 4]

    def broken():
        raise OSError("unreadable chunk")

    with pytest.raises(OSError):
        cache.span(("c", "0", True, 0, 1), broken)
    assert (cache.span(("c", "0", True, 0, 1), read(5)) == 5).all()
    # a failed read ahead is read again by the request, which then owns any error
    cache.prefetch(("d", "0", True, 0, 1), broken)
    assert (cache.span(("d", "0", True, 0, 1), read(6)) == 6).all()
    assert calls == [1, 3, 4, 5, 6]

    # close waits for a read ahead nobody asked for and keeps nothing
    release = threading.Event()

    def slow():
        release.wait(10)
        return read(7)()

    cache.prefetch(("e", "0", True, 0, 1), slow)
    threading.Timer(0.05, release.set).start()
    cache.close()
    assert calls[-1] == 7
    assert (cache.span(("e", "0", True, 0, 1), read(8)) == 8).all()


def test_engine_shape_mismatch_is_an_error_not_a_written_map(tmp_path):
    surface, _ = _write_surface(tmp_path / "surface.zarr")
    checkpoint = _save_checkpoint(tmp_path / "model.pth")

    class _WrongShape:
        name = "wrong"

        def unsupported_reason(self, request):
            return None

        def predict(self, request):
            return FlatEngineResult(
                probability=np.zeros((2, 2), dtype=np.uint8),
                coverage=np.zeros((2, 2), dtype=bool),
            )

    args = normalize_inference_paths(
        parse_args([str(surface), str(checkpoint), str(tmp_path / "out.tif"), *COMMON])
    )
    with pytest.raises(RuntimeError, match="expected \\(40, 48\\)"):
        run(args, engine_factory=lambda *_: _WrongShape())
    assert not (tmp_path / "out.tif").exists()


def test_harness_refuses_a_forwarded_engine_that_would_replace_the_reference(tmp_path):
    surface, _ = _write_surface(tmp_path / "surface.zarr")
    checkpoint = _save_checkpoint(tmp_path / "model.pth")

    with pytest.raises(ValueError, match="--candidate"):
        engine_parity.main(
            [str(surface), str(checkpoint), str(tmp_path / "out"), "--candidate", "torch-stub",
             "--", *COMMON, "--engine", "inkrs"]
        )
    assert not (tmp_path / "out" / "torch").exists()


def test_rebuilt_coverage_matches_pytorch_where_gaussian_tails_add_up(tmp_path):
    # with 128 px Gaussian tiles some edge pixels sit under 1e-6 in every tile
    # but above it in the sum, so a per-tile threshold misses them
    volume = np.random.default_rng(3).integers(1, 255, size=(DEPTH, 136, 136))
    surface = tmp_path / "surface.zarr"
    ZarrWriter.create(surface, shape=volume.shape, chunks=(DEPTH, 64, 64), dtype="u1")[:] = volume
    checkpoint = _save_checkpoint(tmp_path / "model.pth", side=128)
    requests = []

    class _Recorder:
        name = "recorder"

        def unsupported_reason(self, request):
            requests.append(request)
            return "it only records"

        def predict(self, request):
            raise AssertionError("declined requests never run")

    args = normalize_inference_paths(
        parse_args(
            [str(surface), str(checkpoint), str(tmp_path / "out.tif"), "--workers", "0",
             "--no-compile", "--blend-mode", "gaussian", "--stride", "4"]
        )
    )
    run(args, engine_factory=lambda *_: _Recorder())

    [request] = requests
    pytorch_covered = tifffile.imread(tmp_path / "out.tif") != 0
    rebuilt = engine_parity.schedule_coverage(request)
    np.testing.assert_array_equal(rebuilt, pytorch_covered)
    assert rebuilt.sum() == 18384


def test_inkrs_accepts_autocast_without_an_explicit_dtype(tmp_path, monkeypatch):
    # --amp-dtype default leaves the dtype to torch.autocast, which is fp16 on CUDA
    monkeypatch.setitem(sys.modules, "inkrs", None)

    with pytest.raises(EngineUnavailable) as raised:
        select_engine(
            _engine_args(tmp_path, "--engine-fallback", "error"),
            _inkrs_shaped_model(amp_dtype=None),
            torch.device("cuda", 0),
        )

    assert "not installed" in str(raised.value)
    assert "fp16" not in str(raised.value)


def test_engine_runs_leave_pytorch_off_the_device_until_an_input_falls_back(
    tmp_path, monkeypatch
):
    from vesuvius.ink_detection.inference import infer

    surface, _ = _write_surface(tmp_path / "surface.zarr")
    checkpoint = _save_checkpoint(tmp_path / "model.pth")
    prepared = []
    real_prepare = infer.prepare_model_for_inference

    def counting_prepare(model, **kwargs):
        prepared.append(model)
        return real_prepare(model, **kwargs)

    monkeypatch.setattr(infer, "prepare_model_for_inference", counting_prepare)
    built = []
    real_make = infer.make_model
    monkeypatch.setattr(infer, "make_model", lambda config: built.append(config) or real_make(config))

    class _Engine:
        name = "fixed"

        def __init__(self, decline):
            self.decline = decline

        def unsupported_reason(self, request):
            return "declined" if self.decline else None

        def predict(self, request):
            shape = (request.height, request.width)
            return FlatEngineResult(np.full(shape, 9, np.uint8), np.ones(shape, bool))

    def run_with(engine, output):
        args = normalize_inference_paths(
            parse_args([str(surface), str(checkpoint), str(tmp_path / output), *COMMON,
                        "--direction", "both"])
        )
        run(args, engine_factory=None if engine is None else lambda *_: engine)

    engine = _Engine(decline=False)
    released = weakref.ref(engine)
    gc.disable()
    try:
        run_with(engine, "engine.tif")
        del engine
        # nothing villa keeps may hold the engine, or its GPU memory, past run()
        assert released() is None
    finally:
        gc.enable()
    # an engine that runs every input never needs the PyTorch model built
    assert prepared == [] and built == []

    run_with(_Engine(decline=True), "fallback.tif")
    # both directions fell back, and the model was built and prepared once for both
    assert len(prepared) == 1 and len(built) == 1
    run_with(None, "torch.tif")
    assert len(built) == 2
    for name in ("", "_reverse"):
        np.testing.assert_array_equal(
            tifffile.imread(tmp_path / f"fallback{name}.tif"),
            tifffile.imread(tmp_path / f"torch{name}.tif"),
        )


def test_harness_frees_pytorch_gpu_cache_before_loading_the_candidate(tmp_path, monkeypatch):
    surface, _ = _write_surface(tmp_path / "surface.zarr")
    checkpoint = _save_checkpoint(tmp_path / "model.pth")
    events = []
    real_load = engine_parity.CANDIDATES["torch-stub"]
    monkeypatch.setattr(engine_parity.torch.cuda, "empty_cache", lambda: events.append("empty_cache"))
    monkeypatch.setitem(
        engine_parity.CANDIDATES,
        "torch-stub",
        lambda *args: events.append("load") or real_load(*args),
    )

    status = engine_parity.main(
        [str(surface), str(checkpoint), str(tmp_path / "out"), "--candidate", "torch-stub",
         "--layer-starts", "0,1", "--", *COMMON]
    )

    assert status == 0
    assert events == ["empty_cache", "load"] * 2


def test_harness_releases_the_mps_cache_on_macos(monkeypatch):
    events = []
    monkeypatch.setattr(engine_parity.torch.cuda, "empty_cache", lambda: events.append("cuda"))
    monkeypatch.setattr(engine_parity.torch.mps, "empty_cache", lambda: events.append("mps"))
    engine_parity.release_reference_memory(torch.device("mps"))
    engine_parity.release_reference_memory(torch.device("cuda", 0))
    assert events == ["mps", "cuda"]


@pytest.mark.parametrize(
    ("starts", "error"), [("1000,1001", ValueError), ("3", ValueError), ("0,0", SystemExit), ("-1", SystemExit)]
)
def test_harness_refuses_windows_it_cannot_compare_before_running(tmp_path, starts, error):
    # a window past the 5-layer input would clamp to the full depth and compare that under its own name;
    # start 3 fits only a shallower model than the checkpoint's 3 layers, so the depth is the checkpoint's
    surface, _ = _write_surface(tmp_path / "surface.zarr")
    checkpoint = _save_checkpoint(tmp_path / "model.pth")
    with pytest.raises(error):
        engine_parity.main(
            [str(surface), str(checkpoint), str(tmp_path / "out"), "--candidate", "torch-stub",
             f"--layer-starts={starts}", "--", *COMMON]
        )
    assert not (tmp_path / "out").exists()

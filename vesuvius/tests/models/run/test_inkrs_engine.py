"""vesuvius.predict --engine inkrs: what reaches inkrs, and when villa runs PyTorch instead.

inkrs is replaced by a stand-in module, so these run without a GPU or the inkrs wheel. With the real
wheel installed, one test checks the stand-in's call against inkrs.nnunet_predict's signature.
"""

from __future__ import annotations

import inspect
import sys
import types

import pytest
import torch

from vesuvius.models.run import inference, inkrs_engine

MODEL = "/models/m7"
REAL_LOCAL_ZARR_V2 = inkrs_engine.local_zarr_v2
BASE = ["--model_path", MODEL, "--input_dir", "/data/vol.zarr", "--output_dir", "/out",
        "--blend_in_memory", "/out/final.zarr", "--engine", "inkrs"]


class Unsupported(RuntimeError):
    pass


@pytest.fixture
def on_cuda(monkeypatch):
    """PyTorch sees one CUDA device, as on a GPU host, so `--device` may be left out."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)


@pytest.fixture
def local_input(monkeypatch):
    """BASE's input counts as a local zarr v2 array; it is not on disk, and test_input_formats covers the check"""
    monkeypatch.setattr(inkrs_engine, "local_zarr_v2", lambda path: True)


@pytest.fixture
def fake_inkrs(monkeypatch, on_cuda, local_input):
    calls = []

    def nnunet_predict(model, volume, output, *, device=0, bbox=None, overlap=None, tta=True, threshold=None,
                       chunk=None, max_accum_gb=12.0, algo="find", overwrite=False):
        calls.append({"model": model, "volume": volume, "output": output, "device": device, "bbox": bbox,
                      "overlap": overlap, "tta": tta, "threshold": threshold, "chunk": chunk,
                      "overwrite": overwrite})
        if fake.unsupported:
            raise Unsupported(fake.unsupported)
        return {"patches": 448, "patches_skipped": 3, "flips": 8 if tta else 1, "chunks_written": 27}

    fake = types.ModuleType("inkrs")
    fake.__version__ = "0.4.0"
    fake.nnunet_predict = nnunet_predict
    fake.UnsupportedError = Unsupported
    fake.unsupported = None
    fake.calls = calls
    monkeypatch.setitem(sys.modules, "inkrs", fake)
    monkeypatch.setattr(inkrs_engine, "local_model_dir", lambda args: args.model_path)
    return fake


def old_inkrs():
    """inkrs before 0.4.0, as an install for its ink models leaves it: it imports, with no nnU-Net engine"""
    module = types.ModuleType("inkrs")
    module.__version__ = "0.3.1"
    module.UnsupportedError = Unsupported
    return module


def fallback_warnings(caplog):
    return [r.getMessage() for r in caplog.records if r.name == inkrs_engine.__name__ and r.levelname == "WARNING"]


@pytest.fixture
def no_torch(monkeypatch):
    """Fail the test if villa goes on to build its PyTorch Inferer."""
    ran = []

    class Inferer:
        def __init__(self, **kwargs):
            ran.append(kwargs)

        def infer(self):
            return "/out/final.zarr", None

    monkeypatch.setattr(inference, "Inferer", Inferer)
    return ran


def run(monkeypatch, argv):
    monkeypatch.setattr(sys, "argv", ["vesuvius.predict", *argv])
    return inference.main()


def test_inkrs_gets_villas_resolved_options(monkeypatch, fake_inkrs, no_torch):
    code = run(monkeypatch, [*BASE, "--bbox", "12032:12544,5248:5760,4736:5248", "--overlap", "0.25",
                             "--threshold", "--threshold_value", "0.2", "--blend_chunk_size", "64,64,64",
                             "--device", "cuda:1"])
    assert code == 0 and not no_torch
    [call] = fake_inkrs.calls
    assert call == {"model": MODEL, "volume": "/data/vol.zarr", "output": "/out/final.zarr", "device": 1,
                    "bbox": "12032:12544,5248:5760,4736:5248", "overlap": 0.25, "tta": True, "threshold": 0.2,
                    "chunk": 64, "overwrite": True}


def test_bare_threshold_is_half_and_no_threshold_is_none(monkeypatch, fake_inkrs, no_torch):
    run(monkeypatch, [*BASE, "--threshold", "--disable_tta"])
    run(monkeypatch, BASE)
    assert [(c["threshold"], c["tta"]) for c in fake_inkrs.calls] == [(0.5, False), (None, True)]


@pytest.mark.parametrize("extra", [
    ["--mode", "multiclass"],
    ["--num_parts", "2", "--blend_run_id", "r"],
    ["--tta_type", "rotation"],
    ["--patch_size", "128,128,128"],
    ["--save_softmax"],
    ["--no_skip_empty_patches"],
    ["--max_patches", "4"],
    ["--device", "cpu"],
    ["--blend_sigma_scale", "4"],
    ["--blend_compression_level", "3"],
    ["--blend_chunk_size", "64,128,128"],
    ["--model_type", "train_py"],
    ["--blend_in_memory", "s3://bucket/final.zarr"],
])
def test_requests_inkrs_cannot_reproduce_run_pytorch(monkeypatch, caplog, fake_inkrs, no_torch, extra):
    assert run(monkeypatch, [*BASE, *extra]) == 0
    assert not fake_inkrs.calls and len(no_torch) == 1
    [message] = fallback_warnings(caplog)
    assert message.startswith("The inkrs engine cannot run this command: ")
    assert message.endswith(". Running it with PyTorch.")


def test_input_formats(tmp_path):
    array = tmp_path / "array.zarr"
    array.mkdir()
    (array / ".zarray").write_text("{}")
    group = tmp_path / "group.zarr"
    (group / "0").mkdir(parents=True)
    (group / ".zgroup").write_text("{}")
    (group / "0" / ".zarray").write_text("{}")
    v3 = tmp_path / "v3.zarr"
    v3.mkdir()
    (v3 / "zarr.json").write_text("{}")
    tiffs = tmp_path / "tiffs"
    tiffs.mkdir()
    (tiffs / "0000.tif").write_bytes(b"")
    found = {p.name: inkrs_engine.local_zarr_v2(p) for p in (array, group, v3, tiffs, tmp_path / "missing.zarr")}
    assert found == {"array.zarr": True, "group.zarr": True, "v3.zarr": False, "tiffs": False, "missing.zarr": False}


def test_zarr_v3_input_runs_pytorch(monkeypatch, caplog, fake_inkrs, no_torch, tmp_path):
    v3 = tmp_path / "v3.zarr"
    v3.mkdir()
    (v3 / "zarr.json").write_text("{}")
    monkeypatch.setattr(inkrs_engine, "local_zarr_v2", REAL_LOCAL_ZARR_V2)
    argv = [*BASE]
    argv[argv.index("/data/vol.zarr")] = str(v3)
    assert run(monkeypatch, argv) == 0
    assert not fake_inkrs.calls and len(no_torch) == 1
    assert f"it reads a zarr v2 array, or a zarr v2 group with array 0, got {v3}" in fallback_warnings(caplog)[0]


def test_no_cuda_device_runs_pytorch(monkeypatch, caplog, fake_inkrs, no_torch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert run(monkeypatch, BASE) == 0
    assert not fake_inkrs.calls and len(no_torch) == 1
    assert "it runs on a CUDA GPU, got --device none (no CUDA device found)" in fallback_warnings(caplog)[0]


def test_without_blend_in_memory_runs_pytorch(monkeypatch, fake_inkrs, no_torch):
    argv = [a for a in BASE if a not in ("--blend_in_memory", "/out/final.zarr")]
    monkeypatch.setattr(inference.Inferer, "infer", lambda self: (None, None))
    run(monkeypatch, argv)
    assert not fake_inkrs.calls and len(no_torch) == 1


@pytest.mark.parametrize("flag", ["--engine-fallback", "--engine_fallback"])
def test_unsupported_falls_back_and_error_mode_stops(monkeypatch, caplog, fake_inkrs, no_torch, flag):
    fake_inkrs.unsupported = "recipe fiber-tool"
    assert run(monkeypatch, BASE) == 0
    assert len(fake_inkrs.calls) == 1 and len(no_torch) == 1
    assert fallback_warnings(caplog) == [
        ("The inkrs engine cannot run this command: inkrs does not support it: recipe fiber-tool. "
         "Running it with PyTorch.")
    ]
    with pytest.raises(inkrs_engine.EngineUnavailable, match="recipe fiber-tool"):
        run(monkeypatch, [*BASE, flag, "error"])
    assert len(no_torch) == 1


def test_other_inkrs_failures_are_not_masked(monkeypatch, fake_inkrs, no_torch):
    def broken(*args, **kwargs):
        raise MemoryError("CUDA out of memory")

    fake_inkrs.nnunet_predict = broken
    with pytest.raises(MemoryError):
        run(monkeypatch, BASE)
    assert not no_torch


def test_missing_inkrs_falls_back(monkeypatch, caplog, on_cuda, local_input, no_torch):
    monkeypatch.setitem(sys.modules, "inkrs", None)
    assert run(monkeypatch, BASE) == 0
    assert len(no_torch) == 1
    [message] = fallback_warnings(caplog)
    assert "the inkrs package is not installed" in message and "inkrs 0.4.0 or later" in message


def test_inkrs_without_the_nnunet_engine_falls_back(monkeypatch, caplog, on_cuda, local_input, no_torch):
    monkeypatch.setitem(sys.modules, "inkrs", old_inkrs())
    assert run(monkeypatch, BASE) == 0
    assert len(no_torch) == 1
    assert fallback_warnings(caplog) == [
        ("The inkrs engine cannot run this command: inkrs 0.3.1 has no nnU-Net engine, which needs inkrs 0.4.0 "
         "or later. Running it with PyTorch.")
    ]
    with pytest.raises(inkrs_engine.EngineUnavailable, match="inkrs 0.3.1 has no nnU-Net engine"):
        run(monkeypatch, [*BASE, "--engine_fallback", "error"])
    assert len(no_torch) == 1


def test_default_engine_is_torch(monkeypatch, fake_inkrs, no_torch):
    run(monkeypatch, [a for a in BASE if a not in ("--engine", "inkrs")])
    assert not fake_inkrs.calls and len(no_torch) == 1


def test_stand_in_matches_the_real_signature():
    real = pytest.importorskip("inkrs")
    module, reason = inkrs_engine.import_inkrs()
    assert module is real and reason is None, reason
    params = inspect.signature(real.nnunet_predict).parameters
    for name in ["device", "bbox", "overlap", "tta", "threshold", "chunk", "overwrite"]:
        assert params[name].kind is inspect.Parameter.KEYWORD_ONLY, name
    assert list(params)[:3] == ["model", "volume", "output"]

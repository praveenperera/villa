"""vesuvius.predict --engine inkrs: what reaches inkrs, and when villa runs PyTorch instead.

inkrs is replaced by a stand-in module, so these run without a GPU or the inkrs wheel. With the real
wheel installed, one test checks the stand-in's call against inkrs.nnunet_predict's signature.
"""

from __future__ import annotations

import inspect
import sys
import types

import pytest

from vesuvius.models.run import inference, inkrs_engine

MODEL = "/models/m7"
BASE = ["--model_path", MODEL, "--input_dir", "/data/vol.zarr", "--output_dir", "/out",
        "--blend_in_memory", "/out/final.zarr", "--engine", "inkrs"]


class Unsupported(RuntimeError):
    pass


@pytest.fixture
def fake_inkrs(monkeypatch):
    calls = []

    def nnunet_predict(model, volume, output, *, device=0, bbox=None, overlap=None, tta=True, threshold=None,
                       chunk=None, max_accum_gb=12.0, algo="find", overwrite=False):
        calls.append(dict(model=model, volume=volume, output=output, device=device, bbox=bbox, overlap=overlap,
                          tta=tta, threshold=threshold, chunk=chunk, overwrite=overwrite))
        if fake.unsupported:
            raise Unsupported(fake.unsupported)
        return {"patches": 448, "patches_skipped": 3, "flips": 8 if tta else 1, "chunks_written": 27}

    fake = types.SimpleNamespace(nnunet_predict=nnunet_predict, UnsupportedError=Unsupported, unsupported=None,
                                 calls=calls)
    monkeypatch.setitem(sys.modules, "inkrs", fake)
    monkeypatch.setattr(inkrs_engine, "local_model_dir", lambda args: args.model_path)
    return fake


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
    assert call == dict(model=MODEL, volume="/data/vol.zarr", output="/out/final.zarr", device=1,
                        bbox="12032:12544,5248:5760,4736:5248", overlap=0.25, tta=True, threshold=0.2, chunk=64,
                        overwrite=True)


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
])
def test_requests_inkrs_cannot_reproduce_run_pytorch(monkeypatch, fake_inkrs, no_torch, extra):
    assert run(monkeypatch, [*BASE, *extra]) == 0
    assert not fake_inkrs.calls and len(no_torch) == 1


def test_without_blend_in_memory_runs_pytorch(monkeypatch, fake_inkrs, no_torch):
    argv = [a for a in BASE if a not in ("--blend_in_memory", "/out/final.zarr")]
    monkeypatch.setattr(inference.Inferer, "infer", lambda self: (None, None))
    run(monkeypatch, argv)
    assert not fake_inkrs.calls and len(no_torch) == 1


def test_unsupported_falls_back_and_error_mode_stops(monkeypatch, fake_inkrs, no_torch):
    fake_inkrs.unsupported = "recipe fiber-tool"
    assert run(monkeypatch, BASE) == 0
    assert len(fake_inkrs.calls) == 1 and len(no_torch) == 1
    with pytest.raises(inkrs_engine.EngineUnavailable, match="recipe fiber-tool"):
        run(monkeypatch, [*BASE, "--engine-fallback", "error"])
    assert len(no_torch) == 1


def test_other_inkrs_failures_are_not_masked(monkeypatch, fake_inkrs, no_torch):
    def broken(*args, **kwargs):
        raise MemoryError("CUDA out of memory")

    fake_inkrs.nnunet_predict = broken
    with pytest.raises(MemoryError):
        run(monkeypatch, BASE)
    assert not no_torch


def test_missing_inkrs_falls_back(monkeypatch, no_torch):
    monkeypatch.setitem(sys.modules, "inkrs", None)
    assert run(monkeypatch, BASE) == 0
    assert len(no_torch) == 1


def test_default_engine_is_torch(monkeypatch, fake_inkrs, no_torch):
    run(monkeypatch, [a for a in BASE if a not in ("--engine", "inkrs")])
    assert not fake_inkrs.calls and len(no_torch) == 1


def test_stand_in_matches_the_real_signature():
    real = pytest.importorskip("inkrs")
    params = inspect.signature(real.nnunet_predict).parameters
    for name in ["device", "bbox", "overlap", "tta", "threshold", "chunk", "overwrite"]:
        assert params[name].kind is inspect.Parameter.KEYWORD_ONLY, name
    assert list(params)[:3] == ["model", "volume", "output"]

import sys
import types

import numpy as np
import pytest
import torch
from vc_object_store import ObjectStore

from vesuvius.fiber_spline_generator import cli, engines
from vesuvius.fiber_spline_generator.predict import predict_fibers
from vesuvius.fiber_spline_generator.splines import extract_splines

from . import test_cli
from .test_cli import REQUIRED, bright_voxels_are_vertical_fibers, run, volume  # noqa: F401 (volume is a fixture)


class FakeInkrs(types.ModuleType):
    """inkrs' fiber API on the CPU: the engine runs villa's own prediction with the test network, so its fibers are villa's

    Loading prints to stdout, as inkrs' export of a model folder does on its first run; ``run`` reads every stdout
    line as a JSON event, so each test of the inkrs path also checks that the print stays off stdout
    """

    class UnsupportedError(RuntimeError):
        pass

    def __init__(self, refuse=None, fail=None):
        super().__init__("inkrs")
        self.__version__ = "0.0-test"
        self.opened, self.predicted, self.fitted = [], [], 0
        fake = self

        class NnUNet:
            def __init__(self, model, device=0):
                fake.opened.append((model, device))
                print(f"wrote an export of {model}")
                if refuse:
                    raise fake.UnsupportedError(refuse)
                if fail:
                    raise RuntimeError(fail)

            def predict_block(self, ct, *, mirror=False):
                fake.predicted.append((ct.shape, mirror))
                return predict_fibers(bright_voxels_are_vertical_fibers(), ct, (16, 16, 16), device=torch.device("cpu"), mirror=mirror)

        self.NnUNet = NnUNet

    def fiber_splines(self, probabilities, threshold, progress=None):
        self.fitted += 1
        return extract_splines(probabilities, threshold, progress)


def old_inkrs():
    """inkrs before 0.4.0, as an install for its ink models leaves it: it imports, with no fiber engine"""
    module = types.ModuleType("inkrs")
    module.__version__ = "0.3.0"
    module.UnsupportedError = FakeInkrs.UnsupportedError
    return module


@pytest.fixture
def no_env(monkeypatch):
    monkeypatch.delenv(engines.ENGINE_VARIABLE, raising=False)


def install(monkeypatch, module):
    """``import inkrs`` gives ``module``; None makes the import fail, as with inkrs not installed."""
    monkeypatch.setitem(sys.modules, "inkrs", module)


def on_cuda(monkeypatch):
    """The run believes it has a CUDA GPU; only the fake engine runs on it."""
    monkeypatch.setattr(cli, "select_device", lambda name: torch.device("cuda", 0))


def run_counting(monkeypatch, capsys, volume, output, *extra):
    """test_cli's ``run``, also counting the loads of villa's PyTorch network (its stand-in is made once per load)"""
    loads = []
    make = test_cli.bright_voxels_are_vertical_fibers
    monkeypatch.setattr(test_cli, "bright_voxels_are_vertical_fibers", lambda: loads.append(1) or make())
    code, events = run(monkeypatch, capsys, volume, output, *extra)
    return code, events, loads


def fibers(path):
    """The store's fibers by content: family and points, in the written order."""
    with ObjectStore.open(str(path), read_only=True) as db:
        rows = db.objects_by_kind("fiber")
        with db.geometry_snapshot() as snapshot:
            content = [(row["fiber"]["family"], snapshot.fiber(row["uuid"]).line_points_array().tobytes()) for row in rows]
        generator = rows[0]["meta"]["generator"]
    return content, generator


def warnings(events):
    """The engine's warnings; a run on the CPU also warns that it is slow"""
    return [e["message"] for e in events if e["event"] == "warning" and e["message"].startswith("inkrs")]


@pytest.fixture
def torch_fibers(monkeypatch, capsys, volume, tmp_path, no_env):
    code, events = run(monkeypatch, capsys, volume, tmp_path / "torch.vcobj", "--engine", "torch")
    assert code == 0
    content, generator = fibers(tmp_path / "torch.vcobj")
    assert len(content) == 1 and generator["engine"] == {"predict": "torch", "splines": "torch"}
    return content


def test_engine_defaults_to_torch(no_env):
    assert cli.build_parser().parse_args(REQUIRED).engine == "torch"
    assert cli.build_parser().parse_args(REQUIRED).engine_fallback == "torch"


def test_engine_comes_from_the_environment_unless_given(monkeypatch):
    monkeypatch.setenv(engines.ENGINE_VARIABLE, "inkrs")
    assert cli.build_parser().parse_args(REQUIRED).engine == "inkrs"
    assert cli.build_parser().parse_args([*REQUIRED, "--engine", "torch"]).engine == "torch"
    monkeypatch.setenv(engines.ENGINE_VARIABLE, "")
    assert cli.build_parser().parse_args(REQUIRED).engine == "torch"


def test_unknown_engine_flags_are_refused(no_env):
    for flag in (["--engine", "onnx"], ["--engine-fallback", "cpu"]):
        with pytest.raises(SystemExit):
            cli.build_parser().parse_args([*REQUIRED, *flag])


def test_unknown_engine_in_the_environment_fails_before_any_work(monkeypatch, capsys, volume, tmp_path):
    monkeypatch.setenv(engines.ENGINE_VARIABLE, "Inkrs")
    install(monkeypatch, None)
    output = tmp_path / "fibers.vcobj"
    code, events, loads = run_counting(monkeypatch, capsys, volume, output)
    assert code == 1 and not output.exists() and not loads
    assert events[-1]["event"] == "error" and engines.ENGINE_VARIABLE in events[-1]["message"] and "'Inkrs'" in events[-1]["message"]
    assert not [e for e in events if e["event"] == "plan"]


def test_torch_engine_never_imports_inkrs(monkeypatch, capsys, volume, tmp_path, no_env):
    fake = FakeInkrs()
    install(monkeypatch, fake)
    code, events = run(monkeypatch, capsys, volume, tmp_path / "fibers.vcobj")
    assert code == 0 and not warnings(events)
    assert (fake.opened, fake.predicted, fake.fitted) == ([], [], 0)


@pytest.mark.parametrize("mirror", [False, True])
def test_inkrs_engine_predicts_and_fits_every_block_without_pytorch(monkeypatch, capsys, volume, tmp_path, torch_fibers, mirror):
    fake = FakeInkrs()
    install(monkeypatch, fake)
    on_cuda(monkeypatch)
    output = tmp_path / "fibers.vcobj"
    extra = ["--engine", "inkrs", "--engine-fallback", "error", "--block-size", "48", *(["--mirror"] if mirror else [])]
    code, events, loads = run_counting(monkeypatch, capsys, volume, output, *extra)
    assert code == 0 and not warnings(events) and not loads
    assert fake.opened == [(volume, 0)]
    # two blocks, each read with villa's margin
    assert [m for _, m in fake.predicted] == [mirror, mirror] and fake.fitted == 2
    content, generator = fibers(output)
    assert generator["engine"] == {"predict": "inkrs", "splines": "inkrs", "inkrs": "0.0-test"}
    if not mirror:
        # the stand-in engine is villa's own prediction, so a single block gives the torch run's fibers exactly
        code, _, _ = run_counting(monkeypatch, capsys, volume, tmp_path / "one.vcobj", "--engine", "inkrs")
        assert code == 0 and fibers(tmp_path / "one.vcobj")[0] == torch_fibers


def test_inkrs_engine_uses_the_cuda_device_given(monkeypatch, capsys, volume, tmp_path, no_env):
    fake = FakeInkrs()
    install(monkeypatch, fake)
    monkeypatch.setattr(cli, "select_device", lambda name: torch.device("cuda", 3))
    code, _, _ = run_counting(monkeypatch, capsys, volume, tmp_path / "fibers.vcobj", "--engine", "inkrs")
    assert code == 0 and fake.opened == [(volume, 3)]


def test_missing_inkrs_falls_back_to_torch(monkeypatch, capsys, volume, tmp_path, torch_fibers):
    install(monkeypatch, None)
    output = tmp_path / "fibers.vcobj"
    code, events, loads = run_counting(monkeypatch, capsys, volume, output, "--engine", "inkrs")
    assert code == 0 and len(loads) == 1
    message, = warnings(events)
    assert message.startswith("inkrs cannot run, so PyTorch does: the inkrs package does not import")
    content, generator = fibers(output)
    assert content == torch_fibers and generator["engine"] == {"predict": "torch", "splines": "torch"}


def test_inkrs_without_the_fiber_engine_falls_back_to_torch(monkeypatch, capsys, volume, tmp_path, torch_fibers):
    install(monkeypatch, old_inkrs())
    on_cuda(monkeypatch)
    real = engines.torch_predict
    monkeypatch.setattr(engines, "torch_predict", lambda load, dev: real(load, torch.device("cpu")))
    output = tmp_path / "fibers.vcobj"
    code, events, loads = run_counting(monkeypatch, capsys, volume, output, "--engine", "inkrs")
    assert code == 0 and len(loads) == 1
    message, = warnings(events)
    assert message == "inkrs cannot run, so PyTorch does: inkrs 0.3.0 has no fiber engine, which needs inkrs 0.4.0 or later"
    content, generator = fibers(output)
    assert content == torch_fibers and generator["engine"] == {"predict": "torch", "splines": "torch"}


@pytest.mark.parametrize(
    "fake, device, reason",
    [
        (lambda: FakeInkrs(), "cpu", "it predicts on CUDA GPUs only, and the device is cpu"),
        (lambda: FakeInkrs(), "mps", "it predicts on CUDA GPUs only, and the device is mps"),
        (lambda: FakeInkrs(refuse="a GPU below compute capability 8.0"), "cuda:0", "a GPU below compute capability 8.0"),
    ],
)
def test_prediction_alone_falls_back_when_inkrs_cannot_predict(monkeypatch, capsys, volume, tmp_path, torch_fibers, fake, device, reason):
    fake = fake()
    install(monkeypatch, fake)
    monkeypatch.setattr(cli, "select_device", lambda name: torch.device(device))
    # villa's network runs on the CPU here, whatever device the run believes it has
    real = engines.torch_predict
    monkeypatch.setattr(engines, "torch_predict", lambda load, dev: real(load, torch.device("cpu")))
    output = tmp_path / "fibers.vcobj"
    code, events, loads = run_counting(monkeypatch, capsys, volume, output, "--engine", "inkrs")
    assert code == 0 and len(loads) == 1
    assert f"inkrs cannot predict fibers, so PyTorch does: {reason}" in warnings(events)
    # the polylines are still native
    assert fake.predicted == [] and fake.fitted == 1
    assert fake.opened == ([(volume, 0)] if device == "cuda:0" else [])
    content, generator = fibers(output)
    assert content == torch_fibers and generator["engine"] == {"predict": "torch", "splines": "inkrs", "inkrs": "0.0-test"}


@pytest.mark.parametrize(
    "fake, cuda, reason",
    [
        (lambda: None, True, "inkrs cannot run: the inkrs package does not import"),
        (old_inkrs, True, "inkrs cannot run: inkrs 0.3.0 has no fiber engine"),
        (lambda: FakeInkrs(), False, "inkrs cannot predict fibers: it predicts on CUDA GPUs only"),
        (lambda: FakeInkrs(refuse="no fiber-tool export"), True, "inkrs cannot predict fibers: no fiber-tool export"),
    ],
)
def test_fallback_error_fails_instead_of_running_pytorch(monkeypatch, capsys, volume, tmp_path, no_env, fake, cuda, reason):
    install(monkeypatch, fake())
    if cuda:
        on_cuda(monkeypatch)
    output = tmp_path / "fibers.vcobj"
    code, events, loads = run_counting(monkeypatch, capsys, volume, output, "--engine", "inkrs", "--engine-fallback", "error")
    assert code == 1 and not output.exists() and not loads
    assert events[-1]["event"] == "error" and events[-1]["message"].startswith(reason)


def test_other_inkrs_errors_fail_the_run(monkeypatch, capsys, volume, tmp_path, no_env):
    install(monkeypatch, FakeInkrs(fail="out of GPU memory"))
    on_cuda(monkeypatch)
    output = tmp_path / "fibers.vcobj"
    code, events, loads = run_counting(monkeypatch, capsys, volume, output, "--engine", "inkrs")
    assert code == 1 and not output.exists() and not loads
    assert events[-1] == {"event": "error", "message": "out of GPU memory"}


def test_a_volume_inkrs_cannot_read_is_predicted_by_pytorch(monkeypatch):
    fake = FakeInkrs()
    install(monkeypatch, fake)
    seen = []
    engine = engines.open_engine("inkrs", "torch", "model", torch.device("cuda", 0), np.dtype(np.uint16),
                                 lambda: (bright_voxels_are_vertical_fibers(), (16, 16, 16)), seen.append)
    assert seen == ["inkrs cannot predict fibers, so PyTorch does: it reads uint8 CT only, and the volume is uint16"]
    assert fake.opened == [] and engine.used["predict"] == "torch" and engine.used["splines"] == "inkrs"


def test_a_closed_engine_holds_no_network(monkeypatch):
    install(monkeypatch, FakeInkrs())
    engine = engines.open_engine("inkrs", "error", "model", torch.device("cuda", 0), np.dtype(np.uint8), lambda: None, print)
    engine.close()
    with pytest.raises(RuntimeError, match="closed"):
        engine.predict(np.zeros((4, 4, 4), np.uint8), mirror=False, progress=lambda done, total: None)


def test_native_polylines_give_villas_fibers(monkeypatch, capsys, volume, tmp_path, torch_fibers):
    """The real inkrs package, where it is installed: on the CPU PyTorch predicts and inkrs fits the polylines"""
    inkrs = pytest.importorskip("inkrs")
    if not hasattr(inkrs, "fiber_splines"):
        pytest.skip("this inkrs has no fiber_splines")
    output = tmp_path / "fibers.vcobj"
    code, events, loads = run_counting(monkeypatch, capsys, volume, output, "--engine", "inkrs", "--block-size", "48")
    assert code == 0 and len(loads) == 1
    assert warnings(events) == ["inkrs cannot predict fibers, so PyTorch does: it predicts on CUDA GPUs only, and the device is cpu"]
    content, generator = fibers(output)
    assert generator["engine"] == {"predict": "torch", "splines": "inkrs", "inkrs": inkrs.__version__}
    code, _, _ = run_counting(monkeypatch, capsys, volume, tmp_path / "torch-48.vcobj", "--engine", "torch", "--block-size", "48")
    assert code == 0 and content == fibers(tmp_path / "torch-48.vcobj")[0]

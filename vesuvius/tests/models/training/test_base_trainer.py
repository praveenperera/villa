import torch
from types import SimpleNamespace

from vesuvius.models.training.optimizers import create_optimizer
from vesuvius.models.training.train import BaseTrainer


def _make_mgr():
    return SimpleNamespace(
        gpu_ids=None,
        use_ddp=False,
        targets={"ink": {"weight": 1.0}},
        model_name="test",
    )


class _DummyTrainer(BaseTrainer):
    def __init__(self):
        super().__init__(mgr=_make_mgr(), verbose=False)


def test_base_trainer_prepare_hooks_noop():
    trainer = _DummyTrainer()
    sample = {"image": torch.zeros(1)}
    assert trainer._prepare_sample(sample, is_training=True) is sample
    assert trainer._prepare_batch(sample, is_training=True) is sample


def test_base_trainer_loss_helpers():
    trainer = _DummyTrainer()
    trainer.mgr.targets = {"ink": {"weight": 1.0}}

    pred = torch.tensor([1.0, 2.0])
    target = torch.tensor([0.0, 2.0])
    loss_fn = torch.nn.MSELoss()

    value = trainer._compute_loss_value(
        loss_fn,
        pred,
        target,
        target_name="ink",
        targets_dict={"ink": target},
        outputs={"ink": pred},
    )
    assert torch.isclose(value, torch.tensor(0.5))


def test_base_trainer_should_include_target_in_loss():
    trainer = _DummyTrainer()
    assert trainer._should_include_target_in_loss("ink") is True
    assert trainer._should_include_target_in_loss("ink_skel") is False
    assert trainer._should_include_target_in_loss("is_unlabeled") is False


def test_create_optimizer_adamw_honors_eps_override():
    model = torch.nn.Linear(4, 2)
    optimizer = create_optimizer(
        {
            "name": "adamw",
            "learning_rate": 1e-3,
            "weight_decay": 3e-5,
            "eps": 1e-4,
        },
        model,
    )

    assert isinstance(optimizer, torch.optim.AdamW)
    assert optimizer.defaults["eps"] == 1e-4


def test_create_optimizer_accepts_parameter_groups():
    model = torch.nn.Linear(4, 2)
    parameter_groups = [
        {"params": [model.weight], "lr": 2.5e-4},
        {"params": [model.bias]},
    ]

    optimizer = create_optimizer(
        {"name": "adam", "learning_rate": 1e-3},
        parameter_groups,
    )

    assert [
        [id(parameter) for parameter in group["params"]]
        for group in optimizer.param_groups
    ] == [
        [id(model.weight)],
        [id(model.bias)],
    ]
    assert [group["lr"] for group in optimizer.param_groups] == [2.5e-4, 1e-3]


def test_cudnn_benchmark_is_opt_in_on_cuda(monkeypatch):
    trainer = _DummyTrainer()

    # unset leaves the process's flag as it was, on and off
    trainer.device = torch.device("cuda")
    for before in (False, True):
        monkeypatch.setattr(torch.backends.cudnn, "benchmark", before)
        trainer._configure_cudnn()
        assert torch.backends.cudnn.benchmark is before

    monkeypatch.setattr(torch.backends.cudnn, "benchmark", False)
    trainer.mgr.cudnn_benchmark = True
    trainer.device = torch.device("cpu")
    trainer._configure_cudnn()
    assert torch.backends.cudnn.benchmark is False

    trainer.device = torch.device("cuda")
    trainer._configure_cudnn()
    assert torch.backends.cudnn.benchmark is True

    # the flag is process-wide: a later trainer configured off must clear it
    trainer.mgr.cudnn_benchmark = False
    trainer._configure_cudnn()
    assert torch.backends.cudnn.benchmark is False


def _conv3d_model():
    return torch.nn.Sequential(torch.nn.Conv3d(2, 4, 3), torch.nn.InstanceNorm3d(4, affine=True))


def test_channels_last_3d_is_opt_in_for_3d_cuda_training():
    trainer = _DummyTrainer()
    trainer.mgr.train_patch_size = (8, 8, 8)
    trainer.compile_policy = "off"

    def weight_is_channels_last(model):
        return model[0].weight.is_contiguous(memory_format=torch.channels_last_3d)

    trainer.device = torch.device("cuda")
    assert not weight_is_channels_last(trainer._configure_memory_format(_conv3d_model()))

    trainer.mgr.channels_last_3d = True
    trainer.device = torch.device("cpu")
    assert not weight_is_channels_last(trainer._configure_memory_format(_conv3d_model()))

    trainer.device = torch.device("cuda")
    model = _conv3d_model()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    converted = trainer._configure_memory_format(model)
    assert weight_is_channels_last(converted)
    # the optimizer, built before the conversion, still updates the converted weights
    assert optimizer.param_groups[0]["params"][0] is converted[0].weight
    out = converted(torch.randn(1, 2, 6, 6, 6))
    out.sum().backward()
    optimizer.step()
    assert out.shape == (1, 4, 4, 4, 4)

    # 2D models keep their layout instead of failing on channels_last_3d
    trainer.mgr.train_patch_size = (8, 8)
    conv2d = torch.nn.Conv2d(2, 4, 3)
    assert trainer._configure_memory_format(conv2d).weight.is_contiguous()

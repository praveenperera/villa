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


def test_task_losses_stay_on_device_until_read():
    trainer = _DummyTrainer()
    trainer.mgr.targets = {"ink": {"weight": 2.0}}
    trainer.guide_loss_weight = 0.0

    pred = torch.tensor([1.0, 2.0], requires_grad=True)
    target = torch.tensor([0.0, 2.0])
    loss_fns = {"ink": [(torch.nn.MSELoss(), 1.0)]}

    total_loss, task_losses = trainer._compute_train_loss({"ink": pred}, {"ink": target}, loss_fns)

    # the training step's value is a detached tensor that the loop reads after the optimizer step
    assert torch.is_tensor(task_losses["ink"]) and not task_losses["ink"].requires_grad
    assert float(task_losses["ink"]) == 1.0
    assert total_loss.requires_grad and float(total_loss) == 1.0

    raw = trainer._debug_raw_losses["ink/MSELoss"]["values"]
    assert [float(v) for v in raw] == [0.5]

    # validation reports plain floats, unweighted
    assert trainer._compute_validation_loss({"ink": pred}, {"ink": target}, loss_fns) == {"ink": 0.5}

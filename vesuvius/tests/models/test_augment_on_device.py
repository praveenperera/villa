"""`augment_on_device`: the training pipeline transforms a batch on the GPU with `Pipeline.batch`."""
import numpy as np
import pytest
import torch

from vc_reader import transforms as A

from vesuvius.models.augmentation.pipelines.training_transforms import create_training_transforms

cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")


def _batch(device, batch=2, size=48):
    rng = np.random.default_rng(5)
    image = torch.from_numpy(rng.normal(size=(batch, 1, size, size, size)).astype(np.float32)).to(device)
    label = torch.zeros(batch, 1, size, size, size, dtype=torch.float32, device=device)
    label[:, :, 12:36, 12:36, 12:36] = 1
    return {"image": image, "ink": label}


@cuda_only
def test_device_pipeline_transforms_a_batch_on_the_gpu():
    device = torch.device("cuda")
    pipeline = create_training_transforms(patch_size=(48, 48, 48), label_names=["ink"], backend="cuda")
    source = _batch(device)
    before = source["image"].clone()

    first = pipeline.batch(source, seeds=[11, 12], device=device)
    second = pipeline.batch(source, seeds=[11, 12], device=device)

    assert torch.equal(source["image"], before), "inputs are not modified"
    for key in ("image", "ink"):
        assert first[key].device.type == "cuda"
        assert first[key].shape == source[key].shape
        assert torch.isfinite(first[key]).all()
        assert torch.equal(first[key], second[key]), "the same seeds give the same batch"
    assert not torch.equal(first["image"][0], first["image"][1]), "each sample has its own random stream"
    assert not torch.equal(first["image"], before)


def _surface_sample(rng, size):
    image = rng.normal(size=(1, size, size, size)).astype(np.float32)
    # a padded border, as patches at a volume edge have
    padding_mask = np.ones((1, size, size, size), np.float32)
    padding_mask[..., size - 6:] = 0
    # a sheet to thin, with an ignored corner
    surface = np.zeros((1, size, size, size), np.float32)
    surface[:, 20:26] = 1
    surface[:, :, :8, :8] = 2
    return {"image": image, "padding_mask": padding_mask, "surface": surface}


@cuda_only
def test_device_batch_matches_the_loader_pipeline_for_each_seed():
    """`augment_on_device` gives each sample what a loader worker gives it for the same seed

    The surface recipe's full pipeline, MedialSurface with an ignore label included: labels, masks and
    skeletons match exactly, images to float32 rounding (CPU and CUDA reduce in different orders)
    """
    recipe = dict(patch_size=(48, 48, 48), label_names=["surface"], skeleton_targets=["surface"],
                  skeleton_ignore_values={"surface": 2})
    cpu = create_training_transforms(backend="cpu", **recipe)
    gpu = create_training_transforms(backend="cuda", **recipe)
    rng = np.random.default_rng(7)
    samples = [_surface_sample(rng, 48) for _ in range(16)]
    seeds = [A.seed(42, index, 0) for index in range(len(samples))]
    batch = {key: torch.from_numpy(np.stack([s[key] for s in samples])) for key in samples[0]}

    result = gpu.batch(batch, seeds=seeds, device=torch.device("cuda"))

    for b, (sample, seed) in enumerate(zip(samples, seeds)):
        expected = cpu({key: torch.from_numpy(value) for key, value in sample.items()}, seed=seed)
        assert "surface_skel" in expected
        for key, value in expected.items():
            tolerance = 1e-4 if key == "image" else 0
            torch.testing.assert_close(result[key][b].cpu(), value, atol=tolerance, rtol=tolerance)

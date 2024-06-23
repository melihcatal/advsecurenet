from unittest.mock import MagicMock, patch

import pytest
import torch

from advsecurenet.attacks.attacker.attacker import Attacker
from advsecurenet.attacks.gradient_based.fgsm import FGSM
from advsecurenet.datasets import DatasetFactory
from advsecurenet.models.model_factory import ModelFactory
from advsecurenet.shared.types.configs.attack_configs import FgsmAttackConfig
from advsecurenet.shared.types.configs.attack_configs.attacker_config import \
    AttackerConfig
from advsecurenet.shared.types.configs.dataloader_config import \
    DataLoaderConfig
from advsecurenet.shared.types.configs.device_config import DeviceConfig
from advsecurenet.shared.types.configs.model_config import CreateModelConfig


@pytest.fixture
def device(request):
    device_arg = request.config.getoption("--device")
    return torch.device(device_arg if device_arg else "cpu")


@pytest.fixture
def config(device):
    device_cfg = DeviceConfig(
        processor=device,
    )
    return AttackerConfig(
        model=ModelFactory.create_model(
            CreateModelConfig(
                model_name="CustomMnistModel",
                num_classes=10,
                num_input_channels=1,
                pretrained=False
            )
        ),
        attack=FGSM(config=FgsmAttackConfig(
            epsilon=0.3,
            device=device_cfg
        )),
        dataloader=DataLoaderConfig(
            # get the test dataset
            dataset=DatasetFactory.create_dataset(
                dataset_type="MNIST", return_loaded=True)[1]
        ),
        device=device_cfg,
        return_adversarial_images=True
    )


@pytest.fixture
def kwargs():
    return {}


@pytest.mark.advsecurenet
@pytest.mark.comprehensive
def test_execute_return_adv_images(config, kwargs):
    attacker = Attacker(config, **kwargs)
    adversarial_images = attacker.execute()
    assert isinstance(adversarial_images, list)
    for image in adversarial_images:
        assert isinstance(image, torch.Tensor)


@pytest.mark.advsecurenet
@pytest.mark.comprehensive
def test_execute_dont_return_adv_images(config, kwargs):
    config.return_adversarial_images = False
    attacker = Attacker(config, **kwargs)
    adversarial_images = attacker.execute()
    assert adversarial_images is None


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_prepare_data(config, kwargs):
    attacker = Attacker(config, **kwargs)
    data = torch.Tensor([1, 2, 3])
    prepared_data = attacker._prepare_data(data)
    assert isinstance(prepared_data, list)
    for item in prepared_data:
        assert isinstance(item, torch.Tensor)


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_get_predictions(config, kwargs):
    attacker = Attacker(config, **kwargs)
    images = torch.Tensor([[1, 2, 3], [4, 5, 6]])

    # Mock the model to return a 2D tensor
    attacker._model = MagicMock()
    attacker._model.return_value = torch.Tensor([[0.1, 0.9], [0.4, 0.6]])

    predictions = attacker._get_predictions(images)

    assert isinstance(predictions, torch.Tensor)
    assert predictions.shape == torch.Size([2])
    assert torch.equal(predictions, torch.tensor([1, 1]))


@pytest.mark.advsecurenet
@pytest.mark.comprehensive
def test_generate_adversarial_images(config, kwargs):
    attacker = Attacker(config, **kwargs)
    # random mnist images
    images = torch.rand(
        (2, 1, 28, 28), dtype=torch.float32, requires_grad=True).to(attacker._device)
    labels = torch.tensor([0, 1]).to(attacker._device)
    adversarial_images = attacker._generate_adversarial_images(images, labels)
    assert isinstance(adversarial_images, torch.Tensor)
    assert adversarial_images.shape == images.shape


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_summarize_results(config, kwargs, caplog, capsys):
    attacker = Attacker(config, **kwargs)
    results = {"accuracy": 0.85, "loss": 0.15}

    # Mock the _summarize_metric method to not perform its actual function
    attacker._summarize_metric = MagicMock()

    with caplog.at_level("INFO"):
        attacker._summarize_results(results)

    # Check log output
    assert "Results summary:" in caplog.text
    assert "Results summary: {'accuracy': 0.85, 'loss': 0.15}" in caplog.text

    # Now check the calls to _summarize_metric
    attacker._summarize_metric.assert_any_call("accuracy", 0.85)
    attacker._summarize_metric.assert_any_call("loss", 0.15)


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_create_dataloader(config, kwargs):
    attacker = Attacker(config, **kwargs)
    dataloader = attacker._create_dataloader()
    assert dataloader is not None


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_get_iterator(config, kwargs):
    attacker = Attacker(config, **kwargs)
    iterator = attacker._get_iterator()
    assert iterator is not None


@pytest.mark.advsecurenet
@pytest.mark.essential
@patch("torch.cuda.is_available", return_value=True)
@patch("advsecurenet.attacks.attacker.attacker.Attacker._setup_model")
@patch("advsecurenet.attacks.attacker.attacker.Attacker._create_dataloader")
def test_setup_device_default_cuda(mock_dataloader, mock_model, mock_cuda, config, kwargs):
    config.device.processor = None
    attacker = Attacker(config, **kwargs)
    assert attacker._device == torch.device("cuda")


@pytest.mark.advsecurenet
@pytest.mark.essential
@patch("torch.cuda.is_available", return_value=False)
@patch("advsecurenet.attacks.attacker.attacker.Attacker._setup_model")
@patch("advsecurenet.attacks.attacker.attacker.Attacker._create_dataloader")
def test_setup_device_default_cpu(mock_dataloader, mock_model, mock_cuda, config, kwargs):
    config.device.processor = None
    attacker = Attacker(config, **kwargs)
    assert attacker._device == torch.device("cpu")

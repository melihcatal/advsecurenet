import torch
import pytest
from advsecurenet.attacks.lots import LOTS
from collections import OrderedDict
import torch.nn as nn
from advsecurenet.shared.types import LotsAttackConfig, LotsAttackMode, DeviceType


class TestLOTS:
    @pytest.fixture(autouse=True)
    def setUp(self):
        self.device = DeviceType.CPU
        self.model = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)),
            ('relu1', nn.ReLU()),
            ('conv2', nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)),
            ('relu2', nn.ReLU()),
            ('maxpool', nn.MaxPool2d(kernel_size=2, stride=2)),
            ('flatten', nn.Flatten()),
            ('fc1', nn.Linear(64 * 16 * 16, 10)),])).to(self.device.value)
        self.model.eval()
        self.data = torch.randn(1, 3, 32, 32).to(self.device.value)
        self.target = torch.randn(1, 3, 32, 32).to(self.device.value)
        self.target_class = torch.tensor([0]).to(self.device.value)
        self.target_layer = "fc1"

    def test_attack_iterative(self):
        config = LotsAttackConfig(
            mode=LotsAttackMode.SINGLE, deep_feature_layer=self.target_layer)
        attack = LOTS(config)
        adversarial_data, is_successful = attack.attack(
            self.model, self.data, self.target, target_classes=self.target_class)

        # check if the perturbed image is not equal to the original image
        assert not torch.all(torch.eq(adversarial_data, self.data))

    def test_attack_single(self):
        config = LotsAttackConfig(
            mode=LotsAttackMode.SINGLE, deep_feature_layer=self.target_layer)
        attack = LOTS(config)

        adversarial_data, is_successful = attack.attack(
            self.model, self.data, self.target, target_classes=self.target_class)

        assert not torch.all(torch.eq(adversarial_data, self.data))
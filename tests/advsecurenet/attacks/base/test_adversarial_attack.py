from unittest.mock import Mock

import pytest
import torch

from advsecurenet.attacks.base.adversarial_attack import AdversarialAttack
from advsecurenet.models.base_model import BaseModel
from advsecurenet.shared.types.configs.attack_configs.attack_config import \
    AttackConfig
from advsecurenet.shared.types.configs.device_config import DeviceConfig


class TestAdversarialAttack(AdversarialAttack):
    def attack(self, model: BaseModel, x: torch.Tensor, y: torch.Tensor, *args, **kwargs):
        return x  # minimal implementation
    __test__ = False
    


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_adversarial_attack_init():
    device_config = DeviceConfig(
        processor=torch.device("cpu"),
        use_ddp=False
    )
    attack_config = AttackConfig(
        device=device_config,
        targeted=False
    )
    adversarial_attack = TestAdversarialAttack(attack_config)
    assert adversarial_attack.device_manager.initial_device == torch.device(
        "cpu")
    assert adversarial_attack.device_manager.distributed_mode == False
    assert adversarial_attack.name == "TestAdversarialAttack"
    assert adversarial_attack.targeted == False

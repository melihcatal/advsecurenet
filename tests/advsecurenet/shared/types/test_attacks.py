import pytest

from advsecurenet.attacks import (FGSM, LOTS, PGD, CWAttack, DecisionBoundary,
                                  DeepFool)
from advsecurenet.shared.types.attacks import AttackType


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_attack_type_enum_values():
    assert AttackType.LOTS.value == LOTS
    assert AttackType.FGSM.value == FGSM
    assert AttackType.PGD.value == PGD
    assert AttackType.CW.value == CWAttack
    assert AttackType.DEEPFOOL.value == DeepFool
    assert AttackType.DECISION_BOUNDARY.value == DecisionBoundary


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_attack_type_enum_names():
    assert AttackType.LOTS.name == "LOTS"
    assert AttackType.FGSM.name == "FGSM"
    assert AttackType.PGD.name == "PGD"
    assert AttackType.CW.name == "CW"
    assert AttackType.DEEPFOOL.name == "DEEPFOOL"
    assert AttackType.DECISION_BOUNDARY.name == "DECISION_BOUNDARY"

from enum import Enum

from advsecurenet.attacks import (FGSM, LOTS, PGD, CWAttack, DecisionBoundary,
                                  DeepFool)


class AttackType(Enum):
    """
    This Enum class is used to store the types of attacks.
    """
    LOTS = LOTS
    FGSM = FGSM
    PGD = PGD
    CW = CWAttack
    DEEPFOOL = DeepFool
    DECISION_BOUNDARY = DecisionBoundary

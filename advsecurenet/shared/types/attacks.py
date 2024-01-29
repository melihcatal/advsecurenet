from enum import Enum

from advsecurenet.attacks import (FGSM, LOTS, PGD, CWAttack, DecisionBoundary,
                                  DeepFool, TargetedFGSM)


class AttackType(Enum):
    """ 
    Enum class for attack types.
    """
    LOTS = LOTS
    FGSM = FGSM
    PGD = PGD
    CW = CWAttack
    DEEPFOOL = DeepFool
    TARGETED_FGSM = TargetedFGSM
    DECISION_BOUNDARY = DecisionBoundary

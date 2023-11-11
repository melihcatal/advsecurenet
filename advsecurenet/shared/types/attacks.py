from enum import Enum
from advsecurenet.attacks import FGSM, PGD, DeepFool, LOTS, CWAttack


class AttackType(Enum):
    LOTS = LOTS
    FGSM = FGSM
    PGD = PGD
    CW = CWAttack
    DEEPFOOL = DeepFool

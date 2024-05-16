# Mapping of attack types to their respective configurations
from advsecurenet.shared.types.attacks import AttackType
from advsecurenet.shared.types.configs.attack_configs import (
    CWAttackConfig, DecisionBoundaryAttackConfig, DeepFoolAttackConfig,
    FgsmAttackConfig, LotsAttackConfig, PgdAttackConfig)
from cli.shared.types.attack.attacks import (
    CwAttackCLIConfigType, DecisionBoundaryAttackCLIConfigType,
    DeepFoolAttackCLIConfigType, FgsmAttackCLIConfigType,
    LotsAttackCLIConfigType, PgdAttackCLIConfigType)

attack_cli_mapping = {
    "CW": (AttackType.CW, CwAttackCLIConfigType),
    "DEEPFOOL": (AttackType.DEEPFOOL, DeepFoolAttackCLIConfigType),
    "PGD": (AttackType.PGD, PgdAttackCLIConfigType),
    "FGSM": (AttackType.FGSM, FgsmAttackCLIConfigType),
    "DECISION_BOUNDARY": (AttackType.DECISION_BOUNDARY, DecisionBoundaryAttackCLIConfigType),
    "LOTS": (AttackType.LOTS, LotsAttackCLIConfigType),

}

attack_mapping = {
    "FGSM": FgsmAttackConfig,
    "PGD": PgdAttackConfig,
    "DEEPFOOL": DeepFoolAttackConfig,
    "CW": CWAttackConfig,
    "LOTS": LotsAttackConfig,
    "DECISION_BOUNDARY": DecisionBoundaryAttackConfig

}

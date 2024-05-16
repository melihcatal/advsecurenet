from enum import Enum


class ConfigType(Enum):
    """
    Enum for configuration types that can be loaded in the CLI.
    """
    ATTACK = "attack"
    DEFENSE = "defense"
    TRAIN = "train"
    TEST = "test"
    MODEL = "model"
    DATASET = "dataset"
    EVALUATION = "evaluation"
    ADVERSARIAL_TRAINING = "adversarial_training"
    ADVERSARIAL_EVALUATION = "adversarial_evaluation"

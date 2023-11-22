from dataclasses import dataclass
from typing import Literal


@dataclass
class ATLOTSCliConfigType:
    """ 
    This class is used as a type hint for the LOTS CLI configuration used in adversarial training i.e. at_lots
    """

    deep_feature_layer: str
    mode: str
    epsilon: float
    learning_rate: float
    max_iterations: int
    target_labels: list[int]
    target_images_dir: str
    auto_generate_target_images: bool
    maximum_generation_attempts: int


@dataclass
class LOTSCliConfigType(ATLOTSCliConfigType):
    """ 
    This class is used as a type hint for the LOTS CLI configuration.
    """

    model_name: str
    trained_on: str
    model_weights: str
    device: str
    dataset_name: str
    custom_data_dir: str
    dataset_path: Literal["train", "test", "all", "random"]
    random_samples: int

    batch_size: int
    verbose: bool
    save_result_images: bool
    result_images_dir: str
    result_images_prefix: str

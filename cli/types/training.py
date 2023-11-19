from dataclasses import dataclass


@dataclass
class TrainingCliConfigType:
    """
    This dataclass is used to store the configuration of the training CLI.
    """
    model_name: str
    dataset_name: str
    epochs: int
    batch_size: int
    num_workers_train: int
    num_workers_test: int
    shuffle_train: bool
    shuffle_test: bool
    drop_last_train: bool
    drop_last_test: bool
    lr: float
    optimizer: str
    loss: str
    use_ddp: bool
    device: str
    gpu_ids: list[int]
    pin_memory: bool
    save: bool
    save_path: str
    save_name: str
    save_checkpoint: bool
    save_checkpoint_path: str
    save_checkpoint_name: str
    checkpoint_interval: int
    load_checkpoint: bool
    load_checkpoint_path: str
    num_classes: int
    train_dataset_path: str
    test_dataset_path: str
    verbose: bool

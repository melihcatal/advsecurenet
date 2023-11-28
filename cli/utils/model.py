import os

import click
import pkg_resources

from advsecurenet.dataloader import DataLoaderFactory
from advsecurenet.datasets.dataset_factory import DatasetFactory
from advsecurenet.models.model_factory import ModelFactory
from advsecurenet.shared.types.configs import TestConfig, TrainConfig
from advsecurenet.shared.types.dataset import DatasetType
from advsecurenet.utils.model_utils import load_model, save_model
from advsecurenet.utils.tester import Tester
from advsecurenet.utils.trainer import Trainer
from cli.types.training import TrainingCliConfigType
from cli.utils.helpers import get_device_from_cfg


def prepare_model(config_data, num_classes, device):
    """Loads the model and sets its weights."""
    model = ModelFactory.create_model(
        config_data['model_name'],
        num_classes=num_classes,
        pretrained=config_data['pretrained'],
        weights=config_data['pretrained_weights'],
    )

    # set weights path to weights directory if not specified
    if not config_data['model_weights']:
        folder_path = pkg_resources.resource_filename(
            "advsecurenet", "weights")
        file_name = f"{config_data['model_name']}_{config_data['trained_on']}_weights.pth"
        config_data['model_weights'] = os.path.join(folder_path, file_name)

    # If we are using a pretrained model, we don't need to load weights
    if model and config_data['pretrained']:
        return model

    return load_model(
        model,
        config_data['model_weights'],
        device=device,
    )


def cli_train(config_data: TrainingCliConfigType):
    # set save path to weights directory if not specified
    if not config_data.save_path:
        config_data.save_path = pkg_resources.resource_filename(
            "advsecurenet", "weights")

    if not config_data.model_name or not config_data.dataset_name:
        raise ValueError("Please provide both model name and dataset name!")

    try:
        save_path_print = config_data.save_path if config_data.save_path else "weights directory"
        device = get_device_from_cfg(config_data)

        # match the dataset name to the dataset type
        dataset_name = config_data.dataset_name.upper()
        if dataset_name not in DatasetType._value2member_map_:
            raise ValueError("Unsupported dataset name! Choose from: " +
                             ", ".join([e.value for e in DatasetType]))

        dataset_type = DatasetType(dataset_name)

        dataset_obj = DatasetFactory.create_dataset(dataset_type)
        train_data = dataset_obj.load_dataset(train=True)
        test_data = dataset_obj.load_dataset(train=False)

        train_data_loader = DataLoaderFactory.create_dataloader(
            train_data, batch_size=config_data.batch_size, shuffle=True)
        test_data_loader = DataLoaderFactory.create_dataloader(
            test_data, batch_size=config_data.batch_size, shuffle=False)

        model = ModelFactory.create_model(
            config_data.model_name, num_classes=dataset_obj.num_classes)
        model.train()

        train_config = TrainConfig(
            model=model,
            train_loader=train_data_loader,
            criterion=config_data.loss,
            optimizer=config_data.optimizer,
            epochs=config_data.epochs,
            learning_rate=config_data.lr,
            device=device,
            save_checkpoint=config_data.save_checkpoint,
            save_checkpoint_path=config_data.save_checkpoint_path,
            save_checkpoint_name=config_data.save_checkpoint_name,
            checkpoint_interval=config_data.checkpoint_interval,
            load_checkpoint=config_data.load_checkpoint,
            load_checkpoint_path=config_data.load_checkpoint_path,
            use_ddp=config_data.use_ddp,
            gpu_ids=config_data.gpu_ids,
            pin_memory=config_data.pin_memory,
        )
        if config_data.use_ddp:
            _execute_ddp_training(train_config, dataset_name, train_data)
            return

        trainer = Trainer(train_config)

        trainer.train()

        save_name = config_data.save_name if config_data.save_name else f"{config_data.model_name}_{dataset_name}_weights.pth"

        if config_data.save:
            click.echo(f"Saving model to {save_path_print}")
            save_model(model, filename=save_name,
                       filepath=config_data.save_path)

        click.echo(f"Model trained on {dataset_name}!")

    except FileExistsError as e:
        print(
            f"Model {config_data.model_name} trained on {dataset_name} already exists at {save_path_print}!")
    except Exception as e:
        print(
            f"Error training model {config_data.model_name} on {dataset_name}! Details: {e}")


def cli_test(config_data: TestConfig):
    # set weights path to weights directory if not specified
    if not config_data['model_weights']:
        folder_path = pkg_resources.resource_filename(
            "advsecurenet", "weights")
        file_name = f"{config_data['model_name']}_{config_data['dataset_name']}_weights.pth"
        config_data['model_weights'] = os.path.join(folder_path, file_name)

    if not config_data['model_name'] or not config_data['dataset_name']:
        raise ValueError("Please provide both model name and dataset name!")

    try:
        device = get_device_from_cfg(config_data)

        # match the dataset name to the dataset type
        dataset_name = config_data['dataset_name'].upper()
        if dataset_name not in DatasetType._value2member_map_:
            raise ValueError("Unsupported dataset name! Choose from: " +
                             ", ".join([e.value for e in DatasetType]))

        dataset_type = DatasetType(dataset_name)

        dataset_obj = DatasetFactory.create_dataset(dataset_type)
        test_data = dataset_obj.load_dataset(train=False)

        test_data_loader = DataLoaderFactory.create_dataloader(
            test_data, batch_size=config_data['batch_size'], shuffle=False)

        model = ModelFactory.create_model(
            config_data['model_name'], num_classes=dataset_obj.num_classes)

        model = load_model(model, config_data['model_weights'], device=device)

        model.eval()
        tester = Tester(model=model, test_loader=test_data_loader,
                        device=device, criterion=config_data['loss'])
        tester.test()

    except Exception as e:
        click.echo(
            f"Error evaluating model {config_data['model_name']} on {dataset_name}! Details: {e}")


def get_models(model_type: str) -> list[str]:
    model_list_getters = {
        "all": ModelFactory.available_models,
        "custom": ModelFactory.available_custom_models,
        "standard": ModelFactory.available_standard_models
    }

    model_list = model_list_getters.get(model_type, lambda: [])()
    if not model_list:
        raise ValueError("Unsupported model type!")
    return model_list

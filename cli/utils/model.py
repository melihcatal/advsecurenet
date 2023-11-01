import os
import click
import pkg_resources
from advsecurenet.utils.config_utils import override_with_cli_args
from advsecurenet.shared.types.device import DeviceType
from advsecurenet.shared.types.dataset import DatasetType
from advsecurenet.models.model_factory import ModelFactory
from advsecurenet.datasets.dataset_factory import DatasetFactory
from advsecurenet.dataloader import DataLoaderFactory
from advsecurenet.utils.model_utils import download_weights as util_download_weights, train as util_train, test as util_test, save_model, load_model
from advsecurenet.shared.types import TrainConfig
from advsecurenet.shared.types import TestConfig

def prepare_model(config_data, num_classes, device):
    """Loads the model and sets its weights."""
    model = ModelFactory.get_model(config_data['model_name'], num_classes=num_classes)

    # set weights path to weights directory if not specified
    if not config_data['model_weights']:
        folder_path =  pkg_resources.resource_filename("advsecurenet", "weights")
        file_name = f"{config_data['model_name']}_{config_data['trained_on']}_weights.pth"
        config_data['model_weights'] = os.path.join(folder_path, file_name)

    return load_model(model, config_data['model_weights'], device=device)


def cli_train(config_data:TrainConfig):

    # set save path to weights directory if not specified
    if not config_data['save_path']:
        config_data['save_path'] = pkg_resources.resource_filename("advsecurenet", "weights")

    if not config_data['model_name'] or not config_data['dataset_name']:
        raise ValueError("Please provide both model name and dataset name!")

    try:
        save_path_print = config_data['save_path'] if config_data['save_path'] else "weights directory"
        device = DeviceType.from_string(config_data['device']) if config_data['device'] else DeviceType.CPU
        device = device.value
        

        # match the dataset name to the dataset type
        dataset_name = config_data['dataset_name'].upper()
        if dataset_name not in DatasetType._value2member_map_:
            raise ValueError("Unsupported dataset name! Choose from: " + ", ".join([e.value for e in DatasetType]))
                
        dataset_type = DatasetType(dataset_name)

        dataset_obj = DatasetFactory.load_dataset(dataset_type)
        train_data = dataset_obj.load_dataset(train=True)
        test_data = dataset_obj.load_dataset(train=False)

        train_data_loader = DataLoaderFactory.get_dataloader(train_data, batch_size=config_data['batch_size'], shuffle=True)
        test_data_loader = DataLoaderFactory.get_dataloader(test_data, batch_size=config_data['batch_size'], shuffle=False)

        model = ModelFactory.get_model(config_data['model_name'], num_classes=dataset_obj.num_classes)
        model.train()

        util_train(model, train_data_loader, epochs=config_data['epochs'], learning_rate=config_data['lr'], device=device)

        save_name = config_data['save_name'] if config_data['save_name'] else f"{config_data['model_name']}_{dataset_name}_weights.pth"

        save_model(model, save_name)

        model.eval()
        util_test(model, test_data_loader, device=device)

        click.echo(f"Trained model {config_data['model_name']} on {dataset_name} and saved to {save_path_print} as {save_name}!")

    except FileExistsError as e:
        print(f"Model {config_data['model_name']} trained on {dataset_name} already exists at {save_path_print}!")
    except Exception as e:
        print(f"Error training model {config_data['model_name']} on {dataset_name}! Details: {e}")



def cli_test(config_data:TestConfig):
   
    # set weights path to weights directory if not specified
    if not config_data['model_weights']:
        folder_path =  pkg_resources.resource_filename("advsecurenet", "weights")
        file_name = f"{config_data['model_name']}_{config_data['dataset_name']}_weights.pth"
        config_data['model_weights'] = os.path.join(folder_path, file_name)

    if not config_data['model_name'] or not config_data['dataset_name']:
        raise ValueError("Please provide both model name and dataset name!")

    try:
        device = DeviceType.from_string(config_data['device']) if config_data['device'] else DeviceType.CPU
        device = device.value

        # match the dataset name to the dataset type
        dataset_name = config_data['dataset_name'].upper()
        if dataset_name not in DatasetType._value2member_map_:
            raise ValueError("Unsupported dataset name! Choose from: " + ", ".join([e.value for e in DatasetType]))
                
        dataset_type = DatasetType(dataset_name)

        dataset_obj = DatasetFactory.load_dataset(dataset_type)
        test_data = dataset_obj.load_dataset(train=False)

        test_data_loader = DataLoaderFactory.get_dataloader(test_data, batch_size=config_data['batch_size'], shuffle=False)

        model = ModelFactory.get_model(config_data['model_name'], num_classes=dataset_obj.num_classes)
        
        model = load_model(model, config_data['model_weights'], device=device)

        model.eval()

        util_test(model, test_data_loader, device=device, criterion=loss)

    except Exception as e:
        click.echo(f"Error evaluating model {config_data['model_name']} on {dataset_name}! Details: {e}")


def get_models(model_type: str) -> list[str]:
    model_list_getters = {
            "all": ModelFactory.get_available_models,
            "custom": ModelFactory.get_available_custom_models,
            "standard": ModelFactory.get_available_standard_models
        }

    model_list = model_list_getters.get(model_type, lambda: [])()
    if not model_list:
        raise ValueError("Unsupported model type!")
    return model_list

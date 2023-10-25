# ignore warnings
import warnings
import click
import pkg_resources
import yaml
import os 
from advsecurenet.models.model_factory import ModelFactory
from advsecurenet.utils.model_utils import download_weights as util_download_weights, train as util_train, test as util_test, save_model, load_model
from advsecurenet.shared.types.model import ModelType
from advsecurenet.datasets.dataset_factory import DatasetFactory
from requests.exceptions import HTTPError
from advsecurenet.shared.types.dataset import DatasetType
from advsecurenet.shared.types.device import DeviceType
from advsecurenet.dataloader import DataLoaderFactory
from advsecurenet.utils.config_utils import override_with_cli_args, get_available_configs, generate_default_config_yaml


warnings.simplefilter(action='ignore', category=UserWarning)

@click.group()
def main():
    pass


@main.command()
@click.option('--config', type=click.Path(exists=True), default=None, help='Path to the training configuration yml file.')
@click.option('--model-name', default=None, help='Name of the model to train (e.g. "resnet18").')
@click.option('--dataset-name', default=None, help='Name of the dataset to train on (e.g. "cifar10").')
@click.option('--epochs', default=1, help='Number of epochs to train for. Defaults to 1.')
@click.option('--batch-size', default=32, help='Batch size for training.')
@click.option('--lr', default=0.001, help='Learning rate for training.')
@click.option('--optimizer', default='adam', help='Optimizer to use for training.')
@click.option('--loss', default='cross_entropy', help='Loss function to use for training.')
@click.option('--save-path', default=None, help='The directory to save the model to. If not specified, defaults to the weights directory.')
@click.option('--save-name', default=None, help='The name to save the model as. If not specified, defaults to the {model_name}_{dataset_name}_weights.pth.')
@click.option('--device', default=None, help='The device to train on. Defaults to CPU')
def train(config, model_name, dataset_name, epochs, batch_size, lr, optimizer, loss, save_path, save_name, device):
    """Command to train a model.
    
    Args:
        config (str, optional): Path to the training configuration yml file.
        model_name (str): The name of the model (e.g. "resnet18").
        dataset_name (str): The name of the dataset to train on (e.g. "cifar10").
        epochs (int, optional): The number of epochs to train for.
        batch_size (int, optional): The batch size for training.
        lr (float, optional): The learning rate for training.
        optimizer (str, optional): The optimizer to use for training.
        loss (str, optional): The loss function to use for training.
        save_path (str, optional): The directory to save the model to. If not specified, defaults to the weights directory
        save_name (str, optional): The name to save the model as. If not specified, defaults to the {model_name}_{dataset_name}_weights.pth.
        device (str, optional): The device to train on. Defaults to CPU

    Raises:
        ValueError: If the model name or dataset name is not provided.

    Notes:
        If a configuration file is provided, the CLI arguments will override the configuration file. The CLI arguments have priority.
        Configuration file attributes must match the CLI arguments. For example, if the configuration file has a "model_name" attribute, the CLI argument must be named "model_name" as well.
    """

     # If a config file is provided, load it
    config_data = {}
    if config:
        with open(config, 'r') as file:
            config_data = yaml.safe_load(file)

    # override the config data with the CLI arguments, CLI arguments have priority
    config_data = override_with_cli_args(
        config_data, 
        model_name=model_name,
        dataset_name=dataset_name,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        optimizer=optimizer,
        loss=loss,
        save_path=save_path,
        save_name=save_name,
        device=device
    )

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



@main.command()
@click.option('--config', type=click.Path(exists=True), default=None, help='Path to the evaluation configuration yml file.')
@click.option('--model-name', default=None, help='Name of the model to evaluate (e.g. "resnet18").')
@click.option('--dataset-name', default=None, help='Name of the dataset to evaluate on (e.g. "cifar10").')
@click.option('--model_weights', default=None, help='Path to the model weights to evaluate. Defaults to the weights directory.')
@click.option('--device', default=None, help='The device to evaluate on. Defaults to CPU')
@click.option('--batch-size', default=32, help='Batch size for evaluation.')
@click.option('--loss', default=None, help='Loss function to use for evaluation.')
def test(config, model_name, dataset_name, model_weights, device, batch_size, loss):
    """
    Command to evaluate a model.

    Args:
        config (str, optional): Path to the evaluation configuration yml file.
        model_name (str): The name of the model (e.g. "resnet18").
        dataset_name (str): The name of the dataset to evaluate on (e.g. "cifar10").
        model_weights (str): Path to the model weights to evaluate. Defaults to the weights directory.
        device (str, optional): The device to evaluate on. Defaults to CPU
        batch_size (int, optional): The batch size for evaluation. Defaults to 32.
        loss (str, optional): The loss function to use for evaluation. Defaults to cross entropy.

    Raises:
        ValueError: If the model name or dataset name is not provided.

    Examples:

        >>> advsecurenet test --model-name=resnet18 --dataset-name=cifar10 --model-weights=resnet18_cifar10_weights.pth
        or
        >>> advsecurenet test --config=test_config.yml

    Notes:
        If a configuration file is provided, the CLI arguments will override the configuration file. The CLI arguments have priority.
        Configuration file attributes must match the CLI arguments. For example, if the configuration file has a "model_name" attribute, the CLI argument must be named "model_name" as well.
    
    """

    # If a config file is provided, load it
    config_data = {}
    if config:
        with open(config, 'r') as file:
            config_data = yaml.safe_load(file)

    # override the config data with the CLI arguments, CLI arguments have priority
    config_data = override_with_cli_args(
        config_data, 
        model_name=model_name,
        dataset_name=dataset_name,
        model_weights=model_weights,
        device=device
    )

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

    


@main.command()
def configs():
    """
    Return the list of available configuration files.
    """
    config_list = get_available_configs()
    if len(config_list) == 0:
        click.echo("No configuration files found!")
        return
    
    click.echo("Available configuration files:")
    for config in config_list:
        click.echo(config)


@main.command()
@click.option('--config-name', default=None, help='Name of the configuration file to use. If you are unsure, use the "configs" command to list available configuration files.')
@click.option('--output-path', default=None, help='The directory to save the configuration file to. If not specified, defaults to the current working directory.')
def config_default(config_name: str, output_path:str):
    """
    Generate a default configuration file based on the name of the configuration to use.

    Args:

        config_name (str): The name of the configuration file to use.
        output_path (str): The directory to save the configuration file to. If not specified, defaults to the current working directory.

    Examples:

        >>>  advsecurenet config-default --config-name=train_config.yml
        Generated default configuration file train_config.yml!

    Notes:

        If you are unsure which configuration file to use, use the "configs" command to list available configuration files. You can discard the _config.yml suffix when specifying the configuration name.
    """

    if config_name is None:
        raise ValueError("config-name must be specified and not None!")
    
    if output_path is None:
        output_path = os.getcwd()

    try:
        generate_default_config_yaml(config_name, output_path)
        click.echo(f"Generated default configuration file {config_name}!")
    except:
        click.echo(f"Error generating default configuration file {config_name}!")
    

@main.command()
@click.option('--model-type', 
              type=click.Choice([e.value for e in ModelType] + ['all']), 
              default='all', 
              help="The type of model to list. 'custom' for custom models, 'standard' for standard models, and 'all' for all models. Default is 'all'.")
def models(model_type):
    """Command to list available models.
    
    Args:

        model_type (str, optional): The type of model to list. 'custom' for custom models, 'standard' for standard models, and 'all' for all models. Default is 'all'.

    Raises:
        ValueError: If the model_type is not supported.
    """
    model_list = _get_models(model_type)

    click.echo("Available models:")
    for model in model_list:
        click.echo(model)

@main.command()
@click.option('--model-name', default=None, help='Name of the model for which weights are to be downloaded (e.g. "resnet18").')
@click.option('--dataset-name', default=None, help='Name of the dataset the model was trained on (e.g. "cifar10").')
@click.option('--filename', default=None, help='The filename of the weights on the remote server. If provided, this will be used directly.')
@click.option('--save-path', default=None, help='The directory to save the weights to. If not specified, defaults to the weights directory.')
def download_weights(model_name, dataset_name, filename, save_path):    
    """Command to download model weights from a remote source based on the model and dataset names.
    
    Args: 
        model_name (str, optional): The name of the model (e.g. "resnet18").
        dataset_name (str, optional): The name of the dataset the model was trained on (e.g. "cifar10").
        filename (str, optional): The filename of the weights on the remote server. If provided, this will be used directly.
        save_path (str, optional): The directory to save the weights to. Defaults to weights directory.
    """
    if not model_name or not dataset_name:
        raise ValueError("Please provide both model name and dataset name!")
    try:
        save_path_print = save_path if save_path else "weights directory"
        util_download_weights(model_name, dataset_name, filename, save_path)
        click.echo(f"Downloaded weights to {save_path_print}. You can now use them for training or evaluation!")
    except FileExistsError as e:
        print(f"Model weights for {model_name} trained on {dataset_name} already exist at {save_path_print}!")
    except HTTPError as e:
        print(f"Model weights for {model_name} trained on {dataset_name} not found on remote server!")
    except Exception as e:
        print(f"Error downloading model weights for {model_name} trained on {dataset_name}!") 
    



def _get_models(model_type: str) -> list[str]:
    model_list_getters = {
            "all": ModelFactory.get_available_models,
            "custom": ModelFactory.get_available_custom_models,
            "standard": ModelFactory.get_available_standard_models
        }

    model_list = model_list_getters.get(model_type, lambda: [])()
    if not model_list:
        raise ValueError("Unsupported model type!")
    return model_list


if __name__ == "__main__":
    main()


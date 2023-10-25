# ignore warnings
import warnings
import click
from advsecurenet.models.model_factory import ModelFactory
from advsecurenet.utils.model_utils import download_weights as util_download_weights
from advsecurenet.shared.types.model import ModelType
from requests.exceptions import HTTPError

warnings.simplefilter(action='ignore', category=UserWarning)

@click.group()
def main():
    click.echo("Welcome to the Adversarial Secure Networks CLI!")

@main.command()
@click.option('--model-type', 
              type=click.Choice([e.value for e in ModelType] + ['all']), 
              default='all', 
              help="The type of model to list. 'custom' for custom models, 'standard' for standard models, and 'all' for all models. Default is 'all'.")
def models(model_type):
    """Command to list available models."""
    model_list = _get_models(model_type)

    click.echo("Available models:")
    for model in model_list:
        click.echo(model)

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

@main.command()
@click.option('--model-name', default=None, help='Name of the model for which weights are to be downloaded (e.g. "resnet18").')
@click.option('--dataset-name', default=None, help='Name of the dataset the model was trained on (e.g. "cifar10").')
@click.option('--filename', default=None, help='The filename of the weights on the remote server. If provided, this will be used directly.')
@click.option('--save-path', default=None, help='The directory to save the weights to. If not specified, defaults to the weights directory.')
def download_weights(model_name, dataset_name, filename, save_path):    
    """Command to download model weights from a remote source based on the model and dataset names."""
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
    


if __name__ == "__main__":
    main()





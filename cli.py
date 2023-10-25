# ignore warnings
import warnings
import click
warnings.simplefilter(action='ignore', category=UserWarning)
from advsecurenet.models import ModelFactory, ModelType
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

if __name__ == "__main__":
    main()





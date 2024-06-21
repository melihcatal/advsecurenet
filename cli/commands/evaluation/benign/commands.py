import click


@click.group()
def benign():
    """
    Command to evaluate models on benign examples.
    """


@benign.command()
@click.option('-c', '--config', type=click.Path(exists=True), default=None, help='Path to the evaluation configuration yml file.')
@click.option('--model-name', default=None, help='Name of the model to evaluate (e.g. "resnet18").')
@click.option('--dataset-name', default=None, help='Name of the dataset to evaluate on (e.g. "cifar10").')
@click.option('--model-weights', default=None, help='Path to the model weights to evaluate. Defaults to the weights directory.')
@click.option('-p', '--processor', default=None, help='The processor to evaluate on. Defaults to CPU')
@click.option('--batch-size', default=None, help='Batch size for evaluation.')
@click.option('--loss', default=None, help='Loss function to use for evaluation.')
@click.option('--topk', '-tk', default=None, help='Top k accuracy to calculate.', type=int)
def test(config: str, **kwargs):
    """
    Command to test a model on a dataset. This command does not evaluate the model on adversarial examples.

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

        >>> advsecurenet evaluate test --model-name=resnet18 --dataset-name=cifar10 --model-weights=resnet18_cifar10_weights.pth
        or
        >>> advsecurenet evaluate test --config=test_config.yml

    Notes:
        If a configuration file is provided, matching CLI arguments will override the configuration file. The CLI arguments have priority.
        Configuration file attributes must match the CLI arguments. For example, if the configuration file has a "model_name" attribute, the CLI argument must be named "model_name" as well.

    """
    from cli.logic.evaluation.test.test import cli_test

    cli_test(config, **kwargs)

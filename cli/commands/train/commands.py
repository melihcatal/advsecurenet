"""
Commands for training models.
"""

import click


class IntListParamType(click.ParamType):
    """
    Custom parameter type for a list of integers using the click library.
    """
    name = "intlist"

    def convert(self, value, param, ctx):
        """
        Convert the value to a list of integers.
        """
        try:
            return [int(i) for i in value.split(',')]
        except ValueError:
            self.fail(f"{value} is not a valid list of integers", param, ctx)


INT_LIST = IntListParamType()


@click.command()
@click.option('-c', '--config', type=click.Path(exists=True), default=None, help='Path to the training configuration yml file.')
@click.option('-m', '--model-name', default=None, help='Name of the model to train (e.g. "resnet18").')
@click.option('-d', '--dataset-name', default=None, help='Name of the dataset to train on (e.g. "cifar10").')
@click.option('-e', '--epochs', default=None, type=click.INT, help='Number of epochs to train for. Defaults to 1.')
@click.option('-b', '--batch-size', default=None, type=click.INT, help='Batch size for training.')
@click.option('--lr', default=None, type=click.FLOAT, help='Learning rate for training.')
@click.option('--optimizer', default=None, help='Optimizer to use for training.')
@click.option('--loss', default=None, help='Loss function to use for training.')
# @click.option('--optimizer', default=None, help='Optimizer to use for training. Available options: ' + ', '.join([e.name for e in Optimizer]))
# @click.option('--loss', default=None, help='Loss function to use for training. Available options: ' + ', '.join([e.name for e in Loss]))
@click.option('-s', '--save', type=click.BOOL, is_flag=True, default=None, help='Whether to save the model after training. Defaults to False.')
@click.option('--save-path', default=None, help='The directory to save the model to. If not specified, defaults to the weights directory.')
@click.option('--save-name', default=None, help='The name to save the model as. If not specified, defaults to the {model_name}_{dataset_name}_weights.pth.')
@click.option('-p', '--processor', default=None, help='The processor to train on. Defaults to CPU')
@click.option('--save-checkpoint', type=click.BOOL, is_flag=True, default=None, help='Whether to save model checkpoints during training. Defaults to False.')
@click.option('--checkpoint_interval', default=None, type=click.INT, help='The interval at which to save model checkpoints. Defaults to 1.')
@click.option('--save-checkpoint-path', default=None, help='The directory to save model checkpoints to. If not specified, defaults to the checkpoints directory.')
@click.option('--save-checkpoint-name', default=None, help='The name to save the model checkpoints as. If not specified, defaults to the {model_name}_{dataset_name}_checkpoint_{epoch}.pth.')
@click.option('--load-checkpoint', type=click.BOOL, is_flag=True, default=None, help='Whether to load model checkpoints before training. Defaults to False.')
@click.option('--load-checkpoint-path', default=None, help='The file path to load model checkpoint.')
@click.option('--use-ddp', type=click.BOOL, is_flag=True, default=None, help='Whether to use DistributedDataParallel for training. Defaults to False.')
@click.option('--gpu-ids', default=None, type=INT_LIST, help='Comma-separated list of GPU ids to use for training. Defaults to all available GPUs. E.g., 0,1,2,3')
@click.option('--pin-memory', type=click.BOOL, is_flag=True, default=None, help='Whether to pin memory for training. Defaults to False.')
def train(config: str, **kwargs):
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
        save (bool, optional): Whether to save the model after training. Defaults to False.
        save_path (str, optional): The directory to save the model to. If not specified, defaults to the weights directory
        save_name (str, optional): The name to save the model as. If not specified, defaults to the {model_name}_{dataset_name}_weights.pth.
        device (str, optional): The device to train on. Defaults to CPU
        save_checkpoint (bool, optional): Whether to save model checkpoints during training. Defaults to False.
        checkpoint_interval (int, optional): The interval at which to save model checkpoints. Defaults to 1.
        save_checkpoint_path (str, optional): The directory to save model checkpoints to. If not specified, defaults to the checkpoints directory.
        save_checkpoint_name (str, optional): The name to save the model checkpoints as. If not specified, defaults to the {model_name}_{dataset_name}_checkpoint_{epoch}.pth.
        load_checkpoint (bool, optional): Whether to load model checkpoints before training. Defaults to False.
        load_checkpoint_path (str, optional): The file path to load model checkpoint.
        use_ddp (bool, optional): Whether to use DistributedDataParallel for training. Defaults to False.
        gpu_ids (list[int], optional): The GPU ids to use for training. Defaults to all available GPUs.
        pin_memory (bool, optional): Whether to pin memory for training. Defaults to False.

    Examples:

            >>> advsecurenet train --model-name=resnet18 --dataset-name=cifar10
            or
            >>> advsecurenet train --config=train_config.yml

    Raises:
        ValueError: If the model name or dataset name is not provided.

    Notes:
        If a configuration file is provided, the CLI arguments will override the configuration file. The CLI arguments have priority.
        Configuration file attributes must match the CLI arguments. For example, if the configuration file has a "model_name" attribute, the CLI argument must be named "model_name" as well.
    """
    from cli.logic.train.train import cli_train

    cli_train(config, **kwargs)

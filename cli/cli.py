"""
Command line interface for AdvSecureNet.
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


@click.group()
@click.version_option(version='0.1.7')
def main():
    pass


@click.group()
def attack():
    """
    Command to execute attacks.
    """
    pass


@click.group()
def defense():
    """
    Command to execute defenses.
    """
    pass


@click.group()
def weights():
    """
    Command to model weights.
    """
    pass


main.add_command(attack)
main.add_command(defense)
main.add_command(weights)

if __name__ == "__main__":
    main()


@main.command()
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
@click.option('--device', default=None, help='The device to train on. Defaults to CPU')
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
    from cli.logic.train import cli_train

    cli_train(config, **kwargs)


@main.command()
@click.option('-c', '--config', type=click.Path(exists=True), default=None, help='Path to the evaluation configuration yml file.')
@click.option('--model-name', default=None, help='Name of the model to evaluate (e.g. "resnet18").')
@click.option('--dataset-name', default=None, help='Name of the dataset to evaluate on (e.g. "cifar10").')
@click.option('--model-weights', default=None, help='Path to the model weights to evaluate. Defaults to the weights directory.')
@click.option('--device', default=None, help='The device to evaluate on. Defaults to CPU')
@click.option('--batch-size', default=None, help='Batch size for evaluation.')
@click.option('--loss', default=None, help='Loss function to use for evaluation.')
def test(config: str, **kwargs):
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
        If a configuration file is provided, matching CLI arguments will override the configuration file. The CLI arguments have priority.
        Configuration file attributes must match the CLI arguments. For example, if the configuration file has a "model_name" attribute, the CLI argument must be named "model_name" as well.

    """
    from cli.logic.test import cli_test

    cli_test(config, **kwargs)


@weights.command()
@click.option('-m', '--model-name', default=None, help='Name of the model to evaluate (e.g. "resnet18").')
def available_weights(model_name: str):
    """
    Command to list available weights for a model.

    Args:
        model_name (str): The name of the model (e.g. "resnet18").

    Raises:
        ClickException: If the model name is not provided.,

    Examples:
        >>> advsecurenet available-weights --model-name=resnet18
            IMAGENET1K_V1

    """
    from cli.logic.model import cli_available_weights

    cli_available_weights(model_name)


@main.command()
def configs():
    """
    Return the list of available configuration files.

    Raises:
        ClickException: If no configuration file is found

    Examples:
        >>> advsecurenet configs

    """
    from cli.logic.config import cli_configs

    cli_configs()


@main.command()
@click.option('-c', '--config-name', default=None, help='Name of the configuration file to use. If you are unsure, use the "configs" command to list available configuration files.')
@click.option('-s', '--save', type=click.BOOL, is_flag=True, default=False, help='Whether to save the configuration file to the current directory. Defaults to False.')
@click.option('-p', '--print-output', 'print_output', is_flag=True, default=False, help='Whether to print the configuration file to the console. Defaults to False.')
@click.option('-o', '--output-path', default=None, help='The directory to save the configuration file to. If not specified, defaults to the current working directory.')
def config_default(config_name: str, save: bool, print_output: bool, output_path: str):
    """
    Generate a default configuration file based on the name of the configuration to use.

    Args:

        config_name (str): The name of the configuration file to use.
        output_path (str): The directory to save the configuration file to. If not specified, defaults to the current working directory. It can also be a full path including the filename.

    Examples:

        >>>  advsecurenet config-default -c train -p
        Default configuration file for train: ....
        >>> advsecurenet config-default -c train -s
        Saving default config to ... Generated default configuration file train!
        >>> advsecurenet config-default -c train -s -o ./myconfigs/mytrain_config.yml
        Saving default config to ./myconfigs/mytrain_config.yml ... Generated default configuration file train!
    Notes:

        If you are unsure which configuration file to use, use the "configs" command to list available configuration files. You can discard the _config.yml suffix when specifying the configuration name.
        You can provide a full path including the filename to the output path. If the directory does not exist, it will be created. If the file already exists, it will be overwritten.
        You can provide the relative path to the output path. Make sure it ends with a slash (e.g., ./myconfigs/).
    """
    from cli.logic.config import cli_config_default

    cli_config_default(config_name, save, print_output, output_path)


@main.command()
@click.option('-m', '--model-type',
              # type=click.Choice([e.value for e in ModelType] + ['all']),
              default='all',
              help="The type of model to list. 'custom' for custom models, 'standard' for standard models, and 'all' for all models. Default is 'all'.")
def models(model_type: str):
    """Command to list available models.

    Args:

        model_type (str, optional): The type of model to list. 'custom' for custom models, 'standard' for standard models, and 'all' for all models. Default is 'all'.

    Raises:
        ValueError: If the model_type is not supported.
    """
    from cli.logic.model import cli_models

    cli_models(model_type)


@main.command()
@click.option('-m', '--model-name', default=None, help='Name of the model to inspect (e.g. "resnet18").')
def model_layers(model_name: str):
    """Command to list the layers of a model.

    Args:

        model_name (str): The name of the model (e.g. "resnet18").

    Raises:
        ValueError: If the model name is not provided.
    """
    from cli.logic.model import cli_model_layers

    cli_model_layers(model_name)


@weights.command()
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
    from cli.logic.model import cli_download_weights

    cli_download_weights(model_name, dataset_name, filename, save_path)


################################# ATTACKS #####################################


def common_attack_options(func):
    """Decorator to define common options for attack commands."""
    for option in reversed([
        click.option('-c', '--config', type=click.Path(exists=True), default=None,
                     help='Path to the attack configuration yml file.'),
        click.option('-m', '--model-name', type=click.STRING, default=None,
                     help='Name of the model to be attacked.'),
        click.option('--trained-on', type=click.STRING, default=None,
                     help='Dataset on which the model was trained.'),
        click.option('--model-weights', type=click.Path(exists=True), default=None,
                     help='Path to model weights. If unspecified, uses the default path based on model_name and trained_on.'),
        click.option('--device', default=None, type=click.Choice(
            ['CPU', 'CUDA', 'MPS'], case_sensitive=False), help='Device for executing attacks.'),
        click.option('--dataset-name', type=click.Choice(['cifar10', 'mnist', 'custom'], case_sensitive=False),
                     default=None, help='Dataset for the attack. Choose "custom" for your own dataset.'),
        click.option('--custom-data-dir', type=click.Path(exists=True), default=None,
                     help='Path to custom dataset. Required if dataset_name is "custom".'),
        click.option('--dataset-part', type=click.Choice(['train', 'test', 'all', 'random'], case_sensitive=False),
                     default=None, help='Which part of dataset to use for attack. Ignored if dataset_name is "custom".'),
        click.option('--random-samples', type=click.INT, default=None,
                     help='Number of random samples for attack. Relevant only if dataset_part is "random" and dataset_name isn\'t "custom".'),
        click.option('--batch-size', type=click.INT, default=None,
                     help='Batch size for attack execution.'),
        click.option('--verbose', type=click.BOOL, default=None,
                     help='Whether to print progress of the attack.'),
        click.option('--save_result_images', type=click.BOOL,
                     default=None, help='Whether to save the adversarial images.'),
        click.option('--result_images_dir', type=click.Path(exists=True),
                     default=None, help='Directory to save the adversarial images.'),
        click.option('--result_images_prefix', type=click.STRING,
                     default=None, help='Prefix for the adversarial images.'),
    ]):
        func = option(func)
    return func


def execute_general_attack(attack_type, config_file: str, attack_config_class, **kwargs):
    """
    The general function to execute an attack based on the attack type.

    Args:
        attack_type (AttackType): The type of attack to execute.
        config_file (str): Path to the attack configuration yml file.
        attack_config_class (AttackConfig.AttackConfig): The attack configuration class to use.

    Returns:
        list: The adversarial images generated by the attack.

    Raises:
        ValueError: If the attack type is not supported.
    """
    from cli.logic.attack import cli_execute_general_attack

    return cli_execute_general_attack(attack_type, config_file, attack_config_class, **kwargs)


@attack.command()
@common_attack_options
@click.option('--num-classes', default=None, type=click.INT, help='Number of classes for the attack.')
@click.option('--max-iterations', default=None, type=click.INT, help='Number of iterations for the attack.')
@click.option('--overshoot', default=None, type=click.FLOAT, help='Overshoot value for the attack.')
def deepfool(config, **kwargs):
    """
    Command to execute a DeepFool attack.

    Args:

        config (str, optional): Path to the attack configuration yml file.
        model_name (str): The name of the model to be attacked.
        trained_on (str): Dataset on which the model was trained.
        model_weights (str): Path to model weights. If unspecified, uses the default path based on model_name and trained_on.
        device (str, optional): Device for executing attacks. Defaults to CPU.
        dataset_name (str, optional): Dataset for the attack. Choose "custom" for your own dataset.
        custom_data_dir (str, optional): Path to custom dataset. Required if dataset_name is "custom".
        dataset_part (str, optional): Which part of dataset to use for attack. Ignored if dataset_name is "custom".
        random_samples (int, optional): Number of random samples for attack. Relevant only if dataset_part is "random" and dataset_name isn't "custom".
        batch_size (int, optional): Batch size for attack execution. Defaults to 32.
        verbose (bool, optional): Whether to print progress of the attack. Defaults to True.
        save_result_images (bool, optional): Whether to save the adversarial images. Defaults to False.
        result_images_dir (str, optional): Directory to save the adversarial images. Defaults to None.
        num_classes (int, optional): Number of classes for the attack. Defaults to None.
        max_iterations (int, optional): Number of iterations for the attack. Defaults to None.
        overshoot (float, optional): Overshoot value for the attack. Defaults to None.

    Examples:

            >>> advsecurenet attack deepfool --model-name=resnet18 --trained-on=cifar10 --model-weights=resnet18_cifar10_weights.pth
            or
            >>> advsecurenet attack deepfool --config=deepfool_attack_config.yml

    Notes:

            If a configuration file is provided, matching CLI arguments will override the configuration file. The CLI arguments have priority.
            Configuration file attributes must match the CLI arguments. For example, if the configuration file has a "model_name" attribute, the CLI argument must be named "model_name" as well.
    """
    from cli.logic.attack import cli_attack

    cli_attack("DEEPFOOL", config, **kwargs)


@attack.command()
@common_attack_options
@click.option('--targeted', type=click.BOOL, default=None, help='Whether to perform a targeted attack. Defaults to False.')
@click.option('--c-init', type=click.FLOAT, default=None, help='The initial value of c to use for the attack. Defaults to 0.1.')
@click.option('--kappa', type=click.FLOAT, default=None, help='The confidence value to use for the attack. Defaults to 0.')
@click.option('--learning-rate', type=click.FLOAT, default=None, help='The learning rate to use for the attack. Defaults to 0.01.')
@click.option('--max-iterations', type=click.INT, default=None, help='The maximum number of iterations to use for the attack. Defaults to 10.')
@click.option('--abort-early', type=click.BOOL, default=None, help='Whether to abort the attack early if the loss stops decreasing. Defaults to False.')
@click.option('--binary-search-steps', type=click.INT, default=None, help='The number of binary search steps to use for the attack. Defaults to 10.')
@click.option('--clip-min', type=click.FLOAT, default=None, help='The minimum value for clipping pixel values. Defaults to 0.')
@click.option('--clip-max', type=click.FLOAT, default=None, help='The maximum value for clipping pixel values. Defaults to 1.')
@click.option('--c-lower', type=click.FLOAT, default=None, help='The lower bound for c. Defaults to 1e-6.')
@click.option('--c-upper', type=click.FLOAT, default=None, help='The upper bound for c. Defaults to 1.')
@click.option('--patience', type=click.INT, default=None, help='The number of iterations to wait before early stopping. Defaults to 5.')
def cw(config, **kwargs):
    """
    Command to execute a Carlini-Wagner attack.

    Args:

        config (str, optional): Path to the attack configuration yml file.
        model_name (str): The name of the model to be attacked.
        trained_on (str): Dataset on which the model was trained.
        model_weights (str): Path to model weights. If unspecified, uses the default path based on model_name and trained_on.
        device (str, optional): Device for executing attacks. Defaults to CPU.
        dataset_name (str, optional): Dataset for the attack. Choose "custom" for your own dataset.
        custom_data_dir (str, optional): Path to custom dataset. Required if dataset_name is "custom".
        dataset_part (str, optional): Which part of dataset to use for attack. Ignored if dataset_name is "custom".
        random_samples (int, optional): Number of random samples for attack. Relevant only if dataset_part is "random" and dataset_name isn't "custom".
        batch_size (int, optional): Batch size for attack execution. Defaults to 32.
        verbose (bool, optional): Whether to print progress of the attack. Defaults to True.
        save_result_images (bool, optional): Whether to save the adversarial images. Defaults to False.
        result_images_dir (str, optional): Directory to save the adversarial images. Defaults to None.
        targeted (bool, optional): Whether to perform a targeted attack. Defaults to False.
        c_init (float, optional): The initial value of c to use for the attack. Defaults to 0.1.
        kappa (float, optional): The confidence value to use for the attack. Defaults to 0.
        learning_rate (float, optional): The learning rate to use for the attack. Defaults to 0.01.
        max_iterations (int, optional): The maximum number of iterations to use for the attack. Defaults to 10.
        abort_early (bool, optional): Whether to abort the attack early if the loss stops decreasing. Defaults to False.
        binary_search_steps (int, optional): The number of binary search steps to use for the attack. Defaults to 10.
        clip_min (float, optional): The minimum value for clipping pixel values. Defaults to 0.
        clip_max (float, optional): The maximum value for clipping pixel values. Defaults to 1.
        c_lower (float, optional): The lower bound for c. Defaults to 1e-6.
        c_upper (float, optional): The upper bound for c. Defaults to 1.
        patience (int, optional): The number of iterations to wait before early stopping. Defaults to 5.

    Examples:

            >>> advsecurenet attack cw --model-name=resnet18 --trained-on=cifar10 --model-weights=resnet18_cifar10_weights.pth
            or
            >>> advsecurenet attack cw --config=cw_attack_config.yml

    Notes:

            If a configuration file is provided, matching CLI arguments will override the configuration file. The CLI arguments have priority.
            Configuration file attributes must match the CLI arguments. For example, if the configuration file has a "model_name" attribute, the CLI argument must be named "model_name" as well.

    """
    from cli.logic.attack import cli_attack

    cli_attack("CW", config, **kwargs)


@attack.command()
@common_attack_options
@click.option('--epsilon', default=None, type=click.FLOAT, help='Epsilon value for the attack.')
@click.option('--num-iter', default=None, type=click.INT, help='Number of iterations for the attack.')
@click.option('--alpha', default=None, type=click.FLOAT, help='Alpha value for the attack.')
def pgd(config, **kwargs):
    """
    Command to execute a PGD attack.

    Args:
        config (str, optional): Path to the attack configuration yml file.
        model_name (str): The name of the model to be attacked.
        trained_on (str): Dataset on which the model was trained.
        model_weights (str): Path to model weights. If unspecified, uses the default path based on model_name and trained_on.
        device (str, optional): Device for executing attacks. Defaults to CPU.
        dataset_name (str, optional): Dataset for the attack. Choose "custom" for your own dataset.
        custom_data_dir (str, optional): Path to custom dataset. Required if dataset_name is "custom".
        dataset_part (str, optional): Which part of dataset to use for attack. Ignored if dataset_name is "custom".
        random_samples (int, optional): Number of random samples for attack. Relevant only if dataset_part is "random" and dataset_name isn't "custom".
        batch_size (int, optional): Batch size for attack execution. Defaults to 32.
        verbose (bool, optional): Whether to print progress of the attack. Defaults to True.
        save_result_images (bool, optional): Whether to save the adversarial images. Defaults to False.
        result_images_dir (str, optional): Directory to save the adversarial images. Defaults to None.
        epsilon (float, optional): Epsilon value for the attack. Defaults to 0.3.
        num_iter (int, optional): Number of iterations for the attack. Defaults to 40.
        alpha (float, optional): Alpha value for the attack. Defaults to 0.01.

    Examples:

                >>> advsecurenet attack pgd --model-name=resnet18 --trained-on=cifar10 --model-weights=resnet18_cifar10_weights.pth
                or
                >>> advsecurenet attack pgd --config=pgd_attack_config.yml

    Notes:

                If a configuration file is provided, matching CLI arguments will override the configuration file. The CLI arguments have priority.
                Configuration file attributes must match the CLI arguments. For example, if the configuration file has a "model_name" attribute, the CLI argument must be named "model_name" as well.  
    """
    from cli.logic.attack import cli_attack

    cli_attack("PGD", config, **kwargs)


@attack.command()
@common_attack_options
@click.option('--epsilon', default=None, type=click.FLOAT, help='Epsilon value for the attack.')
def fgsm(config, **kwargs):
    """
    Command to execute a FGSM attack.

    Args:

        config (str, optional): Path to the attack configuration yml file.
        model_name (str): The name of the model to be attacked.
        trained_on (str): Dataset on which the model was trained.
        model_weights (str): Path to model weights. If unspecified, uses the default path based on model_name and trained_on.
        device (str, optional): Device for executing attacks. Defaults to CPU.
        dataset_name (str, optional): Dataset for the attack. Choose "custom" for your own dataset.
        custom_data_dir (str, optional): Path to custom dataset. Required if dataset_name is "custom".
        dataset_part (str, optional): Which part of dataset to use for attack. Ignored if dataset_name is "custom".
        random_samples (int, optional): Number of random samples for attack. Relevant only if dataset_part is "random" and dataset_name isn't "custom".
        batch_size (int, optional): Batch size for attack execution. Defaults to 32.
        verbose (bool, optional): Whether to print progress of the attack. Defaults to True.
        save_result_images (bool, optional): Whether to save the adversarial images. Defaults to False.
        result_images_dir (str, optional): Directory to save the adversarial images. Defaults to None.
        epsilon (float, optional): Epsilon value for the attack. Defaults to 0.3.

    Examples:

                >>> advsecurenet attack fgsm --epsilon 0.1
                or
                >>> advsecurenet attack fgsm --config=fgsm_attack_config.yml

    Notes:

                    If a configuration file is provided, matching CLI arguments will override the configuration file. The CLI arguments have priority.
                    Configuration file attributes must match the CLI arguments. For example, if the configuration file has a "model_name" attribute, the CLI argument must be named "model_name" as well.

    """
    from cli.logic.attack import cli_attack

    cli_attack("FGSM", config, **kwargs)


@attack.command()
@common_attack_options
@click.option('--mode', default=None, type=click.STRING, help='Mode for the attack.')
@click.option('--epsilon', default=None, type=click.FLOAT, help='Epsilon value for the attack.')
@click.option('--max-iterations', default=None, type=click.INT, help='Number of iterations for the attack.')
@click.option('--deep-feature-layer', default=None, type=click.STRING, help='Deep feature layer for the attack.')
@click.option('--learning-rate', default=None, type=click.FLOAT, help='Learning rate for the attack.')
@click.option('--auto_generate_target_images', default=None, type=click.BOOL, help='Whether to automatically generate target images.')
@click.option('--target_images_dir', default=None, type=click.STRING, help='Target images path.')
@click.option('--maximum_generation_attempts', default=None, type=click.INT, help='Maximum number of attempts to generate target images.')
def lots(config, **kwargs):
    """
    Command to execute a LOTS attack.

    Args:

        config (str, optional): Path to the attack configuration yml file.
        model_name (str): The name of the model to be attacked.
        trained_on (str): Dataset on which the model was trained.
        model_weights (str): Path to model weights. If unspecified, uses the default path based on model_name and trained_on.
        device (str, optional): Device for executing attacks. Defaults to CPU.
        dataset_name (str, optional): Dataset for the attack. Choose "custom" for your own dataset.
        custom_data_dir (str, optional): Path to custom dataset. Required if dataset_name is "custom".
        dataset_part (str, optional): Which part of dataset to use for attack. Ignored if dataset_name is "custom".
        random_samples (int, optional): Number of random samples for attack. Relevant only if dataset_part is "random" and dataset_name isn't "custom".
        batch_size (int, optional): Batch size for attack execution. Defaults to 32.
        verbose (bool, optional): Whether to print progress of the attack. Defaults to True.
        save_result_images (bool, optional): Whether to save the adversarial images. Defaults to False.
        result_images_dir (str, optional): Directory to save the adversarial images. Defaults to None.
        mode (str, optional): Mode for the attack. Defaults to "LOTS".
        epsilon (float, optional): Epsilon value for the attack. Defaults to 0.3.
        max_iterations (int, optional): Number of iterations for the attack. Defaults to 40.
        deep_feature_layer (str, optional): Deep feature layer for the attack. Defaults to "layer4".
        learning_rate (float, optional): Learning rate for the attack. Defaults to 0.01.
        auto_generate_target_images (bool, optional): Whether to automatically generate target images. Defaults to False.
        target_images_dir (str, optional): Target images path. Defaults to None.
        maximum_generation_attempts (int, optional): Maximum number of attempts to generate target images. Defaults to 100.

    Examples:

                    >>> advsecurenet attack lots --model-name=resnet18 --trained-on=cifar10 --model-weights=resnet18_cifar10_weights.pth
                    or
                    >>> advsecurenet attack lots --config=lots_attack_config.yml

    Notes:

                            If a configuration file is provided, matching CLI arguments will override the configuration file. The CLI arguments have priority.
                            Configuration file attributes must match the CLI arguments. For example, if the configuration file has a "model_name" attribute, the CLI argument must be named "model_name" as well.   

    """
    from cli.logic.attack import cli_attack

    cli_attack("LOTS", config, **kwargs)

################################# DEFENSES #####################################


@defense.command()
@click.option('-c', '--config', type=click.Path(exists=True), default=None, help='Path to the adversarial configuration yml file.')
def adversarial_training(config, **kwargs):
    """
    Command to execute adversarial training. It can be used to train a single model or an ensemble of models and attacks.

    Args:
        config (str, optional): Path to the adversarial training configuration yml file.

    Examples:
        >>> advsecurenet defense adversarial-training --config= ./adversarial_training_config.yml

    Notes:
        Because of the large number of arguments, it is mandatory to use a configuration file for adversarial training.

    """
    from cli.logic.defense import cli_adversarial_training

    cli_adversarial_training(config, **kwargs)

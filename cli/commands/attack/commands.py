import click


@click.group()
def attack():
    """
    Command to execute attacks.
    """


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
        click.option('-p', '--processor', default=None, type=click.Choice(
            ['cpu', 'cuda', 'mps'], case_sensitive=False), help='Processor for executing attacks.'),
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
    from cli.logic.attack.attack import cli_attack
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
    from cli.logic.attack.attack import cli_attack

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
    from cli.logic.attack.attack import cli_attack

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
    from cli.logic.attack.attack import cli_attack

    cli_attack("FGSM", config, **kwargs)


@attack.command()
@common_attack_options
@click.option('-id', '--initial-delta', default=None, type=click.FLOAT, help='Initial delta value for the attack.')
@click.option('-ie', '--initial-epsilon', default=None, type=click.FLOAT, help='Initial epsilon value for the attack.')
@click.option('-mdt', '--max-delta-trials', default=None, type=click.INT, help='Maximum number of delta trials for the attack.')
@click.option('-met', '--max-epsilon-trials', default=None, type=click.INT, help='Maximum number of epsilon trials for the attack.')
@click.option('-m', '--max-iterations', default=None, type=click.INT, help='Number of iterations for the attack.')
@click.option('-sa', '--step-adapt', default=None, type=click.FLOAT, help='Step adaptation value for the attack.')
def decision_boundary(config, **kwargs):
    """
    Command to execute a Decision Boundary attack.

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
            initial_delta (float, optional): Initial delta value for the attack. Defaults to 0.01.
            initial_epsilon (float, optional): Initial epsilon value for the attack. Defaults to 0.01.
            max_delta_trials (int, optional): Maximum number of delta trials for the attack. Defaults to 15.
            max_epsilon_trials (int, optional): Maximum number of epsilon trials for the attack. Defaults to 15.
            max_iterations (int, optional): Number of iterations for the attack. Defaults to 100.
            step_adapt (float, optional): Step adaptation value for the attack. Defaults to 0.01.
    """
    from cli.logic.attack.attack import cli_attack
    cli_attack("DECISION_BOUNDARY", config, **kwargs)


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
    from cli.logic.attack.attack import cli_attack

    cli_attack("LOTS", config, **kwargs)

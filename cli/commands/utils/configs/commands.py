import click


@click.group()
def configs():
    """
    Command to manage configuration files. Configuration files are used to run experiments with different settings. With this command, you can list available configuration files, generate a default configuration file, and save it to a directory.
    """


@configs.command()
def list():
    """
    Return the list of available configuration files.

    Raises:
        ClickException: If no configuration file is found

    Examples:
        >>> advsecurenet configs

    """
    from cli.logic.utils.config import cli_configs

    cli_configs()


@configs.command()
@click.option('-c', '--config-name', default=None, help='Name of the configuration file to use. If you are unsure, use the "configs" command to list available configuration files.')
@click.option('-s', '--save', type=click.BOOL, is_flag=True, default=False, help='Whether to save the configuration file to the current directory. Defaults to False.')
@click.option('-p', '--print-output', 'print_output', is_flag=True, default=False, help='Whether to print the configuration file to the console. Defaults to False.')
@click.option('-o', '--output-path', default=None, help='The directory to save the configuration file to. If not specified, defaults to the current working directory.')
def get(config_name: str, save: bool, print_output: bool, output_path: str):
    """
    Generate a default configuration file based on the name of the configuration to use.

    Args:

        config_name (str): The name of the configuration file to use.
        output_path (str): The directory to save the configuration file to. If not specified, defaults to the current working directory. It can also be a full path including the filename.

    Examples:

        >>>  advsecurenet utils configs get -c train -p
        Default configuration file for train: ....
        >>> advsecurenet utils configs get -c train -s
        Saving default config to ... Generated default configuration file train!
        >>> advsecurenet utils configs get -c train -s -o ./myconfigs/mytrain_config.yml
        Saving default config to ./myconfigs/mytrain_config.yml ... Generated default configuration file train!
    Notes:

        If you are unsure which configuration file to use, use the "configs" command to list available configuration files. You can discard the _config.yml suffix when specifying the configuration name.
        You can provide a full path including the filename to the output path. If the directory does not exist, it will be created. If the file already exists, it will be overwritten.
        You can provide the relative path to the output path. Make sure it ends with a slash (e.g., ./myconfigs/).
    """
    from cli.logic.utils.config import cli_config_default

    cli_config_default(config_name, save, print_output, output_path)

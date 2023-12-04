"""
Logic for the 'configs' command.
"""

import os

import click

from advsecurenet.utils.config_utils import (generate_default_config_yaml,
                                             get_available_configs)


def cli_configs():
    """
    List available configuration files.
    """

    config_list = get_available_configs()
    if len(config_list) == 0:
        click.echo("No configuration file found!")
        click.ClickException("No configuration file found!")
        return

    click.echo("Available configuration files: \n")
    for i, config in enumerate(config_list):
        click.echo(f"{i+1}. {config}")
    # add space
    click.echo("")


def cli_config_default(config_name: str, save: bool, print_output: bool, output_path: str):
    """
    Save or print default configuration file.
    """

    if config_name is None:
        raise ValueError("config-name must be specified and not None!")

    if output_path is None:
        output_path = os.getcwd()

    try:
        default_config = generate_default_config_yaml(
            config_name, output_path, save=save, config_subdir="cli")

        if print_output:
            click.echo("*"*50)
            click.echo(f"Default configuration file for {config_name}:\n")
            formatted_config = '\n'.join(
                [f"{key}: {value}" for key, value in default_config.items()])
            click.echo(formatted_config)
            click.echo("*"*50)
        if save:
            click.echo(f"Generated default configuration file {config_name}!")
    except FileNotFoundError:
        click.echo(
            f"Configuration file {config_name} not found! You can use the 'configs' command to list available configuration files.")
    except Exception:
        click.echo(
            f"Error generating default configuration file {config_name}!", e)

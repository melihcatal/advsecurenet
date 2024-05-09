"""
Logic for the 'configs' command.
"""

import os
from typing import Optional

import click

from cli.utils.config import (generate_default_config_yaml,
                              get_available_configs)


def cli_configs():
    """
    List available configuration files.
    """

    config_list = get_available_configs()
    if len(config_list) == 0:
        click.echo("No configuration file found!")
        raise click.ClickException("No configuration file found!")

    click.echo("Available configuration files: \n")
    for i, config in enumerate(config_list):
        click.echo(f"{i+1}. {config}")
    # add space
    click.echo("")


def cli_config_default(config_name: str,
                       save: Optional[bool] = False,
                       print_output: Optional[bool] = False,
                       output_path: Optional[str] = None
                       ):
    """
    Save or print default configuration file.

    Args:
        config_name (str): The name of the configuration file.
        save (bool): Whether to save the configuration file. Defaults to False.
        print_output (bool): Whether to print the configuration file. Defaults to False.
        output_path (str): The path to save the configuration file. Defaults to None.

    Raises:
        click.ClickException: If no configuration file name is provided.
    """

    if not save and not print_output:
        click.secho(
            "Please provide either the --save or --print-output flag to save or print the configuration file!", fg="red")
        return

    if config_name is None:
        raise click.ClickException("Please provide a configuration file name!")

    if output_path is None:
        output_path = os.getcwd()

    try:
        default_config = generate_default_config_yaml(
            config_name, output_path, save=save)

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
            f"Configuration file {config_name} not found! You can use the 'configs' command to list available configuration files.", err=True)
    except Exception as e:
        click.echo(
            f"Error generating default configuration file {config_name}! Details: {e}", err=True)

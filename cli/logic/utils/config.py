"""
Logic for the 'configs' command.
"""

import json
import os
from collections import defaultdict
from typing import Optional

import click

from cli.shared.utils.config import (generate_default_config_yaml,
                                     get_available_configs)


def cli_configs():
    """
    List available configuration files using Click for command line interface, grouped by title and description.
    """

    # Ensure you pass the appropriate path where configs are stored
    config_list = get_available_configs()
    if len(config_list) == 0:
        raise click.ClickException("No configuration file found!")

    # Group configurations by title and description
    grouped_configs = defaultdict(list)
    for config in config_list:
        grouped_configs[(config['title'], config['description'])
                        ].append(config['config_file'])
    # Add an additional space at the beginning for better readability
    click.echo("")
    click.secho("Available configuration files:\n", bold=True, fg="magenta")
    for idx, ((title, description), config_files) in enumerate(grouped_configs.items(), start=1):
        click.secho(f"{idx}. Title: {title} ", bold=True, fg="blue")
        click.secho(
            f"   Description: {description if description else 'No description available.'}", fg="green")
        click.secho("   Config Files:", fg="red", bold=True)
        for file_idx, file in enumerate(config_files, start=1):
            click.secho(f"       {file_idx}. {file}")
        # Separator for better visual distinction between categories
        click.secho("\n" + "="*60 + "\n", bold=True, fg="cyan")

    # Adds an additional space at the end for better readability
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
            click.secho("*"*50, fg='blue', bold=True)
            click.secho(
                f"Default configuration file for the {config_name}:\n", fg='red', bold=True)
            formatted_config = json.dumps(default_config, indent=4)

            click.secho(formatted_config, fg='green')
            click.secho("*"*50, fg='blue', bold=True)
        if save:
            click.echo(f"Generated default configuration file {config_name}!")
    except FileNotFoundError:
        click.echo(
            f"Configuration file {config_name} not found! You can use the 'configs' command to list available configuration files.", err=True)
    except Exception as e:
        click.echo(
            f"Error generating default configuration file {config_name}! Details: {e}", err=True)

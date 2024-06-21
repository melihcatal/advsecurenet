import os
from unittest.mock import patch

import click
import pytest

from cli.logic.utils.config import cli_config_default, cli_configs


@pytest.mark.cli
@pytest.mark.essential
@patch("click.secho")
@patch("click.echo")
@patch("cli.logic.utils.config.get_available_configs", return_value=[])
def test_cli_configs_no_configs(mock_get_available_configs, mock_echo, mock_secho):
    with pytest.raises(click.ClickException):
        cli_configs()

    mock_get_available_configs.assert_called_once()
    mock_echo.assert_not_called()
    mock_secho.assert_not_called()


@pytest.mark.cli
@pytest.mark.essential
@patch("click.secho")
@patch("click.echo")
@patch("cli.logic.utils.config.get_available_configs")
def test_cli_configs_with_configs(mock_get_available_configs, mock_echo, mock_secho):
    mock_get_available_configs.return_value = [
        {"title": "Title1", "description": "Description1",
            "config_file": "config1.yaml"},
        {"title": "Title2", "description": "Description2",
            "config_file": "config2.yaml"},
        {"title": "Title1", "description": "Description1",
            "config_file": "config3.yaml"},
    ]

    cli_configs()

    mock_get_available_configs.assert_called_once()
    mock_echo.assert_any_call("")
    mock_secho.assert_any_call(
        "Available configuration files:\n", bold=True, fg="magenta")
    mock_secho.assert_any_call("1. Title: Title1 ", bold=True, fg="blue")
    mock_secho.assert_any_call("   Description: Description1", fg="green")
    mock_secho.assert_any_call("   Config Files:", fg="red", bold=True)
    mock_secho.assert_any_call("       1. config1.yaml")
    mock_secho.assert_any_call("       2. config3.yaml")
    mock_secho.assert_any_call("2. Title: Title2 ", bold=True, fg="blue")
    mock_secho.assert_any_call("   Description: Description2", fg="green")
    mock_secho.assert_any_call("   Config Files:", fg="red", bold=True)
    mock_secho.assert_any_call("       1. config2.yaml")
    mock_secho.assert_any_call("\n" + "="*60 + "\n", bold=True, fg="cyan")
    mock_echo.assert_any_call("")


@pytest.mark.cli
@pytest.mark.essential
@patch("click.secho")
@patch("click.echo")
@patch("cli.logic.utils.config.generate_default_config_yaml", side_effect=FileNotFoundError)
def test_cli_config_default_file_not_found(mock_generate_default_config_yaml, mock_echo, mock_secho):
    with patch("builtins.print") as mock_print:
        cli_config_default("config1", save=True)

        mock_generate_default_config_yaml.assert_called_once_with(
            "config1", os.getcwd(), save=True)
        mock_echo.assert_called_once_with(
            "Configuration file config1 not found! You can use the 'configs' command to list available configuration files.", err=True
        )


@pytest.mark.cli
@pytest.mark.essential
@patch("click.secho")
@patch("click.echo")
@patch("cli.logic.utils.config.generate_default_config_yaml")
def test_cli_config_default_save(mock_generate_default_config_yaml, mock_echo, mock_secho):
    cli_config_default("config1", save=True)

    mock_generate_default_config_yaml.assert_called_once_with(
        "config1", os.getcwd(), save=True)
    mock_echo.assert_called_once_with(
        "Generated default configuration file config1!")


@pytest.mark.cli
@pytest.mark.essential
@patch("click.secho")
@patch("click.echo")
@patch("cli.logic.utils.config.generate_default_config_yaml")
def test_cli_config_default_print(mock_generate_default_config_yaml, mock_echo, mock_secho):
    mock_generate_default_config_yaml.return_value = {"key": "value"}

    cli_config_default("config1", print_output=True)

    mock_generate_default_config_yaml.assert_called_once_with(
        "config1", os.getcwd(), save=False)
    mock_secho.assert_any_call("*" * 50, fg='blue', bold=True)
    mock_secho.assert_any_call(
        "Default configuration file for the config1:\n", fg='red', bold=True)
    mock_secho.assert_any_call('{\n    "key": "value"\n}', fg='green')
    mock_secho.assert_any_call("*" * 50, fg='blue', bold=True)


@pytest.mark.cli
@pytest.mark.essential
@patch("click.secho")
@patch("click.echo")
@patch("cli.logic.utils.config.generate_default_config_yaml", side_effect=Exception("General error"))
def test_cli_config_default_general_error(mock_generate_default_config_yaml, mock_echo, mock_secho):
    with patch("builtins.print") as mock_print:
        cli_config_default("config1", save=True)

        mock_generate_default_config_yaml.assert_called_once_with(
            "config1", os.getcwd(), save=True)
        mock_echo.assert_called_once_with(
            "Error generating default configuration file config1! Details: General error", err=True
        )


@pytest.mark.cli
@pytest.mark.essential
def test_cli_config_default_no_name():
    with pytest.raises(click.ClickException):
        cli_config_default(None, save=True)


@pytest.mark.cli
@pytest.mark.essential
def test_cli_config_default_no_save_no_print():
    with patch("click.secho") as mock_secho:
        cli_config_default("config1")
        mock_secho.assert_called_once_with(
            "Please provide either the --save or --print-output flag to save or print the configuration file!", fg="red"
        )

from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from cli.commands.evaluation.commands import evaluate


@pytest.fixture
def runner():
    return CliRunner()


@pytest.mark.cli
@pytest.mark.essential
def test_evaluate_help(runner):
    result = runner.invoke(evaluate, ['--help'])
    assert result.exit_code == 0
    assert "Usage:" in result.output
    assert "Command to evaluate models." in result.output

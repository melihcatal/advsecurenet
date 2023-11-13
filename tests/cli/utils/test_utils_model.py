from unittest.mock import MagicMock, patch
import pytest
from cli.utils.model import prepare_model, cli_train, cli_test, get_models
from advsecurenet.shared.types.configs import TrainConfig, TestConfig


@pytest.mark.parametrize("model_type", ["all", "custom", "standard"])
def test_get_models(model_type):
    models = get_models(model_type)
    assert isinstance(models, list)

# TODO: Add unit tests for cli_train and cli_test functions

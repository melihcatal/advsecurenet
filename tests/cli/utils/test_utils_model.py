from unittest.mock import MagicMock, patch

import pytest

from advsecurenet.shared.types.configs import TestConfig, TrainConfig
from cli.shared.utils.model import (cli_test, cli_train, get_models,
                                    prepare_model)


@pytest.mark.parametrize("model_type", ["all", "custom", "standard"])
def test_get_models(model_type):
    models = get_models(model_type)
    assert isinstance(models, list)

# TODO: Add unit tests for cli_train and cli_test functions

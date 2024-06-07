
import click
import torch

from advsecurenet.models import BaseModel
from advsecurenet.models.model_factory import ModelFactory
from advsecurenet.shared.types.configs.model_config import CreateModelConfig
from advsecurenet.utils.dataclass import (filter_for_dataclass,
                                          flatten_dataclass)
from advsecurenet.utils.normalization_layer import NormalizationLayer
from cli.shared.messages.errors import CLIErrorMessages
from cli.shared.types.utils.model import ModelCliConfigType


def create_model(config: ModelCliConfigType) -> BaseModel:
    """
    Creates a model based on the provided configuration.
    Args:
        config (ModelCliConfigType): The model configuration.

    Returns:
        BaseModel: The created model.
    """
    # Flatten the dataclass and filter for the CreateModelConfig dataclass
    flat_config = flatten_dataclass(config)
    filtered_config = filter_for_dataclass(flat_config, CreateModelConfig)
    create_model_config = CreateModelConfig(**filtered_config)
    # create the model
    model = ModelFactory.create_model(create_model_config)

    # if we are using a normalization layer, add it to the model
    if config.norm_config.add_norm_layer:
        _validate_norm_layer(config)

        norm_layer = NormalizationLayer(
            mean=config.norm_config.norm_mean,
            std=config.norm_config.norm_std,
        )

        model.add_layer(new_layer=norm_layer, position=0, inplace=True)

    if not config.is_external and config.path_configs.model_weights_path is not None and config.pretrained:
        click.secho(
            "Trying to load the model weights from the provided path...", fg="yellow")

        # load the model weights if provided
        model.load_state_dict(torch.load(
            config.path_configs.model_weights_path, map_location=torch.device('cpu')))

    return model


def _validate_norm_layer(config: ModelCliConfigType) -> None:
    """
    Validate the normalization layer.
    """
    if config.norm_config.add_norm_layer and (config.norm_config.norm_mean is None or config.norm_config.norm_std is None):
        raise ValueError(
            CLIErrorMessages.TRAINER.value.NORM_LAYER_MISSING_MEAN_OR_STD.value)
    if config.norm_config.add_norm_layer and (not isinstance(config.norm_config.norm_mean, list) or not isinstance(config.norm_config.norm_std, list)):
        raise ValueError(
            CLIErrorMessages.TRAINER.value.NORM_LAYER_MEAN_OR_STD_NOT_LIST.value)
    if config.norm_config.add_norm_layer and len(config.norm_config.norm_mean) != config.num_input_channels:
        raise ValueError(
            CLIErrorMessages.TRAINER.value.NORM_LAYER_LENGTH_MISMATCH_MEAN_AND_NUM_INPUT_CHANNELS.value)
    if config.norm_config.add_norm_layer and len(config.norm_config.norm_std) != config.num_input_channels:
        raise ValueError(
            CLIErrorMessages.TRAINER.value.NORM_LAYER_LENGTH_MISMATCH_STD_AND_NUM_INPUT_CHANNELS.value)
    if len(config.norm_config.norm_mean) != len(config.norm_config.norm_std):
        raise ValueError(
            CLIErrorMessages.TRAINER.value.NORM_LAYER_LENGTH_MISMATCH_MEAN_AND_STD.value)

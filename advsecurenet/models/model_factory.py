import logging
from enum import EnumMeta
from typing import Optional

from torch import nn

from advsecurenet.models.base_model import BaseModel
from advsecurenet.models.custom_model import CustomModel
from advsecurenet.models.external_model import ExternalModel
from advsecurenet.models.standard_model import StandardModel
from advsecurenet.shared.types.configs.model_config import (
    CreateModelConfig, CustomModelConfig, ExternalModelConfig,
    StandardModelConfig)
from advsecurenet.shared.types.model import ModelType
from advsecurenet.utils.reproducibility_utils import set_seed

logger = logging.getLogger(__name__)


class ModelFactory:
    """
    This class is a factory class for creating models. It provides a single interface for creating models. It supports both standard models and custom models.
    """

    @staticmethod
    def infer_model_type(model_name: str) -> ModelType:
        """
        This function infers the model type based on the model_name.

        Parameters
        ----------
        model_name: str
            The name of the model to be loaded. For example, 'resnet18' or 'CustomMnistModel'.

        Returns
        -------
        ModelType
            The model type of the model_name.

        Raises
        ------
        ValueError
            If the model_name is not supported by torchvision or is not a custom model.
        """
        if model_name in StandardModel.models():
            return ModelType.STANDARD

        if model_name in CustomModel.models():
            return ModelType.CUSTOM

        raise ValueError(
            "Unsupported model. If you are trying to load an external model, please set is_external=True in the CreateModelConfig.")

    @staticmethod
    def create_model(config: Optional[CreateModelConfig] = None,
                     **kwargs) -> BaseModel:
        """ 
        This function creates a model based on the CreateModelConfig.

        Args:
            config (Optional[CreateModelConfig]): The configuration for creating the model. If not provided, the model will be created with the passed keyword arguments.
            CreateModelConfig contains the following fields:
                - model_name: str
                - num_classes: Optional[int] = 1000
                - num_input_channels: Optional[int] = 3
                - pretrained: Optional[bool] = True
                - weights: Optional[str] = "IMAGENET1K_V1"
                - custom_models_path: Optional[str] = "CustomModels"
                - model_arch_path: Optional[str] = None
                - model_weights_path: Optional[str] = None
                - is_external: bool = False
                - random_seed: Optional[int] = None

            **kwargs: Additional keyword arguments to be passed to the model constructor.

        Returns:
            BaseModel: The created model.

        Note:
            If the model is a custom model, the model_name should be the name of the custom model class. For example, 'CustomMnistModel'.
            You can use your external model by setting is_external=True in the CreateModelConfig and providing the model_arch_path and model_weights_path.
        """
        try:

            if config is None or not isinstance(config, CreateModelConfig):
                config = CreateModelConfig(**kwargs)
            if config.is_external:
                cfg = ExternalModelConfig(
                    model_name=config.model_name,
                    num_classes=config.num_classes,
                    model_arch_path=config.model_arch_path,
                    pretrained=config.pretrained,
                    model_weights_path=config.model_weights_path
                )
                return ExternalModel(cfg, **kwargs)

            inferred_type: ModelType = ModelFactory.infer_model_type(
                config.model_name)

            ModelFactory._validate_create_model_config(inferred_type, config)

            if config.random_seed is not None:
                set_seed(config.random_seed)
            if inferred_type == ModelType.STANDARD:
                cfg = StandardModelConfig(
                    model_name=config.model_name,
                    num_classes=config.num_classes,
                    pretrained=config.pretrained,
                    weights=config.weights
                )
                return StandardModel(cfg, **kwargs)

            if inferred_type == ModelType.CUSTOM:
                # The custom model name would typically be without the 'Custom' prefix for the filename.
                # For example: 'MnistModel' for 'CustomMnistModel.py'. Adjust as necessary.
                cfg = CustomModelConfig(
                    model_name=config.model_name,
                    num_classes=config.num_classes,
                    num_input_channels=config.num_input_channels,
                    custom_models_path=config.custom_models_path,
                    pretrained=config.pretrained
                )
                return CustomModel(cfg, **kwargs)
        except Exception as e:
            err = f"Error creating model. Please check the model_name and other arguments. Error: {str(e)}"
            logger.error(err)
            raise ValueError(err) from e

    @staticmethod
    def _validate_create_model_config(
        inferred_type: ModelType, config: CreateModelConfig
    ):
        """ 
        This function validates the CreateModelConfig based on the inferred model type.
        """
        if inferred_type == ModelType.CUSTOM and config.pretrained:
            raise ValueError(
                "Custom models do not support pretrained weights. Instead, you can load the weights after loading the model.")

        if inferred_type == ModelType.STANDARD and config.pretrained and config.random_seed is not None:
            raise ValueError(
                "Pretrained standard models do not support random seed. They already have a fixed set of weights :)")

    @staticmethod
    def available_models() -> list[str]:
        """
        Returns a list of all available models.
        """
        return StandardModel.models() + CustomModel.models()

    @staticmethod
    def available_standard_models() -> list[str]:
        """
        Returns a list of all available standard models that are supported by torchvision.
        """
        return StandardModel.models()

    @staticmethod
    def available_custom_models() -> list[str]:
        """
        Returns a list of all available custom models that are created by the user. These models are stored in the 'advsecurenet/models/CustomModels' directory.
        """
        return CustomModel.models()

    @staticmethod
    def available_weights(model_name: str) -> EnumMeta:
        """
        Returns a list of available weights for the given model_name.

        Args:
            model_name (str): The name of the model. You can get the list of available models using StandardModel.models().

        Returns:
            EnumMeta: A EnumMeta object containing the available weights for the given model_name.

        Raises:
            ValueError: If the model_name is not supported.

        Note:
            You can get the list of available weights for a model using list(StandardModel.available_weights(model_name)).
            This is only applicable for standard models.

        Raises:
            ValueError: If the model_name is not supported.
            ValueError: If the model_name is a custom model.

        Examples:
            >>> ModelFactory.available_weights("resnet18")
            <enum 'ResNet18Weights'>
            >>> list(ModelFactory.available_weights("resnet18"))
            [ResNet18_Weights.IMAGENET1K_V1]

        """
        inferred_type: ModelType = ModelFactory.infer_model_type(model_name)
        if inferred_type == ModelType.CUSTOM:
            raise ValueError(
                "Custom models do not support pretrained weights. Instead, you can load the weights after loading the model.")
        return StandardModel.available_weights(model_name)

    @staticmethod
    def add_layer(model: nn.Module, new_layer: nn.Module, position: int = -1) -> nn.Module:
        """
        Inserts a new layer into an existing PyTorch model at the specified position. If the model is not a Sequential model,
        it will be converted into one.

        Args:
            model (nn.Module): The original model to which the new layer will be added.
            new_layer (nn.Module): The layer to be inserted into the model.
            position (int): The position at which to insert the new layer. If set to -1, the layer is added at the end.
                            Positions are zero-indexed.

        Returns:
            nn.Module: A new model with the layer added at the specified position.

        Raises:
            ValueError: If the specified position is out of bounds.
        """
        # Convert non-Sequential models to Sequential if necessary
        if not isinstance(model, nn.Sequential):
            model = nn.Sequential(model)

        # Prepare the list of existing layers
        layers = list(model.children())

        # Check position validity
        if position < -1 or position > len(layers):
            raise ValueError("Position out of bounds.")

        # Insert the new layer at the specified position or append at the end
        if position == -1 or position == len(layers):
            layers.append(new_layer)
        else:
            layers.insert(position, new_layer)

        # Create a new Sequential model with the updated list of layers
        updated_model = nn.Sequential(*layers)

        return updated_model

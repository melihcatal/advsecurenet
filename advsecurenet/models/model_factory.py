from enum import EnumMeta
from typing import Optional

import torch
import torch.nn as nn

from advsecurenet.models import *
from advsecurenet.models.base_model import BaseModel
from advsecurenet.shared.types.model import ModelType
from advsecurenet.utils.reproducibility_utils import set_seed


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
        elif model_name in CustomModel.models():
            return ModelType.CUSTOM
        else:
            raise ValueError("Unsupported model")

    @staticmethod
    def create_model(model_name: str, num_classes: int, num_input_channels: int = 3, pretrained: bool = False, weights: Optional[str] = "IMAGENET1K_V1", random_seed: Optional[int] = None, **kwargs) -> BaseModel:
        """
        This function returns a model based on the model_name. If the model_name is a standard model, it will be loaded from torchvision.models. If the model_name is a custom model, it will be loaded from advsecurenet.models.CustomModels.

        Parameters
        ----------
        model_name: str
            The name of the model to be loaded. For example, 'resnet18' or 'CustomMnistModel'.
        num_classes: int
            The number of classes in the dataset.
        num_input_channels: int
            The number of input channels in the dataset. Default is 3 for RGB images.
        pretrained: bool
            Whether to load pretrained weights or not. Default is False. This is only applicable for standard models.
        weights: str
            The weights for the pretrained standard model. Default is IMAGENET1K_V1. This is only applicable for standard models.
        random_seed: int
            The random seed to use for the model. Default is None. If provided, the model will be initialized with the given random seed. This helps in reproducibility.
        **kwargs
            Additional keyword arguments that will be passed to the model.

        Raises
        ------
        ValueError
            If the model_name is not supported by torchvision or is not a custom model.
        ValueError
            If the model_name is a standard model and pretrained is True and random_seed is not None.
        ValueError
            If the model_name is a custom model and weights is not None.

        Returns
        -------
        BaseModel
            The model for the given model_name. It will be of type StandardModel or CustomModel.

        """
        try:
            inferred_type: ModelType = ModelFactory.infer_model_type(
                model_name)

            if inferred_type == ModelType.CUSTOM and pretrained:
                raise ValueError(
                    "Custom models do not support pretrained weights. Instead, you can load the weights after loading the model.")

            if inferred_type == ModelType.STANDARD and pretrained and random_seed is not None:
                raise ValueError(
                    "Pretrained standard models do not support random seed. They already have a fixed set of weights :)")

            if random_seed is not None:
                set_seed(random_seed)

            if inferred_type == ModelType.STANDARD:
                return StandardModel(model_name=model_name, num_classes=num_classes, num_input_channels=num_input_channels, pretrained=pretrained, weights=weights, **kwargs)
            elif inferred_type == ModelType.CUSTOM:
                # The custom model name would typically be without the 'Custom' prefix for the filename.
                # For example: 'MnistModel' for 'CustomMnistModel.py'. Adjust as necessary.
                return CustomModel(model_name=model_name, num_classes=num_classes, num_input_channels=num_input_channels, pretrained=pretrained, **kwargs)
        except ValueError as e:
            raise ValueError(
                f"Unsupported model type: {model_name}. If you are trying to load a custom model, please ensure that the model file is in the 'advsecurenet/models/custom' directory.") from e

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

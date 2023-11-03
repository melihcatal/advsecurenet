from advsecurenet.models import *
from advsecurenet.models.base_model import BaseModel
from advsecurenet.shared.types.model import ModelType


class ModelFactory:

    @staticmethod
    def infer_model_type(model_variant: str) -> ModelType:
        """
        This function infers the model type based on the model_variant.

        Parameters
        ----------
        model_variant: str
            The name of the model to be loaded. For example, 'resnet18' or 'CustomMnistModel'.

        Returns
        -------
        ModelType
            The model type of the model_variant.

        Raises
        ------
        ValueError
            If the model_variant is not supported by torchvision or is not a custom model.
        """
        if model_variant in StandardModel.models():
            return ModelType.STANDARD
        elif model_variant in CustomModel.models():
            return ModelType.CUSTOM
        else:
            raise ValueError("Unsupported model")

    @staticmethod
    def get_model(model_variant: str, num_classes: int, num_input_channels: int = 3, pretrained: bool = False, **kwargs) -> BaseModel:
        """
        This function returns a model based on the model_variant. If the model_variant is a standard model, it will be loaded from torchvision.models. If the model_variant is a custom model, it will be loaded from advsecurenet.models.CustomModels.

        Parameters
        ----------
        model_variant: str
            The name of the model to be loaded. For example, 'resnet18' or 'CustomMnistModel'.
        num_classes: int
            The number of classes in the dataset.
        num_input_channels: int
            The number of input channels in the dataset. Default is 3 for RGB images.
        pretrained: bool
            Whether to load pretrained weights or not. Default is False. This is only applicable for standard models.
        **kwargs
            Additional keyword arguments that will be passed to the model.
        """
        try:
            inferred_type: ModelType = ModelFactory.infer_model_type(
                model_variant)

            if inferred_type == ModelType.STANDARD:
                return StandardModel(model_variant=model_variant, num_classes=num_classes, num_input_channels=num_input_channels, pretrained=pretrained, **kwargs)
            elif inferred_type == ModelType.CUSTOM:
                # The custom model name would typically be without the 'Custom' prefix for the filename.
                # For example: 'MnistModel' for 'CustomMnistModel.py'. Adjust as necessary.
                return CustomModel(model_name=model_variant, num_classes=num_classes, num_input_channels=num_input_channels, pretrained=pretrained, **kwargs)
        except ValueError as e:
            raise ValueError(
                f"Unsupported model type: {model_variant}. If you are trying to load a custom model, please ensure that the model file is in the 'advsecurenet/models/custom' directory.") from e

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

import torch.nn as nn
import torchvision.models as models
from enum import EnumMeta
from torchvision.models._api import get_model_weights
from advsecurenet.models.base_model import BaseModel
class StandardModel(BaseModel):
    """
    This class is used to load standard models from torchvision.models. It supports loading pretrained models and
    modifying the model after loading.

    Attributes
    ----------
    model_variant: str
        The name of the model to load. For example, 'resnet18'.
    num_input_channels: int
        The number of input channels in the model. For example, 3 for RGB images.
    weights: str
        The weights for the pretrained model. Default is IMAGENET1K_V1. Weight li

    """

    def __init__(self, 
                 model_variant: str,
                 num_input_channels: int = 3,
                 weights: str = "IMAGENET1K_V1",
                **kwargs):
        self.model_variant = model_variant
        self.num_input_channels = num_input_channels
        self.weights = weights

        # Initialize the BaseModel
        super().__init__(**kwargs)

    def load_model(self) -> models:
        """
        Load the model. This method is called by the BaseModel constructor. It loads the model from torchvision.models based on the model_variant attribute.

        Raises
        ------
        ValueError
            If the model_variant is not supported.

        """

        if not hasattr(models, self.model_variant):
            raise ValueError(f"Unsupported model type: {self.model_variant}")

        if self.pretrained:
            weights = self.weights
        else:
            # this is equivalent to pretrained=False
            weights = None

        self.model = getattr(models, self.model_variant)(
            weights=weights, num_classes=self.num_classes)

        # Perform necessary modifications after model load
        self.modify_model()

    def modify_model(self):
        pass

    @staticmethod
    def models():
        return models.list_models()
    
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

        Examples:
            >>> StandardModel.available_weights("resnet18")
            <enum 'ResNet18Weights'>
            >>> list(StandardModel.available_weights("resnet18"))
            [ResNet18_Weights.IMAGENET1K_V1]

        """
        return get_model_weights(model_name)

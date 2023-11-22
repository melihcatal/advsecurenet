import torch.nn as nn
import torchvision.models as models
from enum import EnumMeta
from torchvision.models._api import get_model_weights
from advsecurenet.models.base_model import BaseModel


class StandardModel(BaseModel):
    """
    This class is used to load standard models from torchvision.models. It supports loading pretrained models and
    modifying the model after loading.

    Args: 
        model_name: str
            The name of the model to load. For example, 'resnet18'.
        num_classes: int
            The number of classes in the dataset.
        pretrained: bool
            Whether to load pretrained weights or not. Default is False.
        weights: str
            The weights for the pretrained model. Default is IMAGENET1K_V1. Weight list can be obtained using StandardModel.available_weights(model_name).

    """

    def __init__(self,
                 model_name: str,
                 num_classes: int,
                 pretrained: bool = False,
                 weights: str = "IMAGENET1K_V1",
                 **kwargs):
        self.model_name = model_name
        self.pretrained = pretrained
        self.weights = weights
        self.num_classes = num_classes
        self.model: nn.Module = None

        # Initialize the BaseModel
        super().__init__()

    def load_model(self) -> None:
        """
        Load the model. This method is called by the BaseModel constructor. It loads the model from torchvision.models based on the model_name attribute and sets the model attribute.

        Raises:
            ValueError: If the model_name is not supported by torchvision.

        """
        if not hasattr(models, self.model_name):
            raise ValueError(f"Unsupported model type: {self.model_name}")

        model_fn = getattr(models, self.model_name)
        if self.pretrained:
            self.model = model_fn(weights=self.weights)
            if self.num_classes != 1000:  # ImageNet has 1000 classes, pretrained models are trained on ImageNet
                self.modify_model()
        else:
            # if not pretrained, load the model without weights and with the specified number of classes
            self.model = model_fn(num_classes=self.num_classes, weights=None)

    def modify_model(self):
        """
        Modifies the model after loading. It updates the number of output classes of the pretrained model to the number of classes in the dataset.
        """
        # Adjust for number of output classes
        named_children_list = list(self.model.named_children())
        for name, module in reversed(named_children_list):
            if isinstance(module, nn.Linear):
                setattr(self.model, name, nn.Linear(
                    module.in_features, self.num_classes))
                break

    @staticmethod
    def models() -> list:
        """
        Returns a list of available standard models from torchvision.models.
        """
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

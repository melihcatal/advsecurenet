from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

import torch
import torch.nn as nn


class BaseModel(ABC, nn.Module):
    """
    Abstract class for models.

    Attributes:
        num_classes (int): The number of classes in the dataset.
        pretrained (bool): Whether to load the pretrained weights or not.
        target_layer (str): The name of the layer to be used as the target layer.
    """

    def __init__(self):
        super().__init__()
        self.model: Optional[nn.Module] = None
        self.load_model()

    @abstractmethod
    def load_model(self, *args, **kwargs) -> None:
        """
        Abstract method to load the model. This method should be implemented
        in derived classes (e.g., StandardModel, CustomModel).
        """
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        if self.model is None:
            raise ValueError("Model is not loaded.")
        return self.model(x)

    def predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predicts the class of the input tensor.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 
                - The predicted class index.
                - The probability of the predicted class.
        """
        logits = self.forward(x)
        probabilities = nn.Softmax(dim=1)(logits)
        predicted_classes = logits.argmax(dim=1)
        max_probabilities = probabilities.max(dim=1)[0]
        return predicted_classes, max_probabilities

    @abstractmethod
    def models(self):
        """
        Return a list of available models.
        """
        pass

    def get_layer_names(self) -> List[str]:
        """
        Return a list of layer names in the model.
        """
        if self.model is None:
            raise ValueError("Model is not loaded.")
        return [name for name, _ in self.model.named_modules() if name != '']

    def get_layer(self, layer_name: str) -> nn.Module:
        """
        Retrieve a specific layer module based on its name.

        Examples:
            >>> model = StandardModel(model_name='resnet18', num_classes=10)
            >>> model.get_layer('layer1.0.conv1')
        """
        if self.model is None:
            raise ValueError("Model is not loaded.")
        return dict(self.model.named_modules()).get(layer_name, None)

    def set_layer(self, layer_name: str, new_layer: nn.Module):
        """
        Replace a specific layer module based on its name with a new module.

        Examples:
            >>> model = StandardModel(model_name='resnet18', num_classes=10)
            >>> model.set_layer('layer1.0.conv1', nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False))
        """
        if self.model is None:
            raise ValueError("The model has not been loaded.")

        # Obtain the parent module and the attribute name of the layer
        parent, name = self._get_parent_module_and_name(layer_name)
        # Set the new layer
        setattr(parent, name, new_layer)

    def _get_parent_module_and_name(self, layer_name: str) -> Tuple[nn.Module, str]:
        """
        Helper method to get the parent module and the attribute name of a layer.
        """
        if self.model is None:
            raise ValueError("The model has not been loaded.")

        if '.' in layer_name:
            parent_name, child_name = layer_name.rsplit('.', 1)
            parent = dict(self.model.named_modules()).get(
                parent_name, self.model)
        else:
            parent = self.model
            child_name = layer_name
        return parent, child_name

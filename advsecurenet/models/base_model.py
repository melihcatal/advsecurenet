from abc import ABC, abstractmethod
from functools import wraps
from typing import List, Optional, Tuple

import torch
from torch import nn
from torchvision.models.feature_extraction import get_graph_node_names


def check_model_loaded(func):
    """
    Wrapper function to check if the model is loaded before calling the decorated function.
    """
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if self.model is None:
            raise ValueError("Model is not loaded.")
        return func(self, *args, **kwargs)
    return wrapper


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
    def load_model(self) -> None:
        """
        Abstract method to load the model. This method should be implemented
        in derived classes (e.g., StandardModel, CustomModel).
        """

    @check_model_loaded
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        return self.model(x)

    @check_model_loaded
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

    @check_model_loaded
    def save_model(self, path: str) -> None:
        """
        Save the model to the specified path.

        Args:
            path (str): The path to save the model.
        """
        torch.save(self.model.state_dict(), path)

    @abstractmethod
    def models(self):
        """
        Return a list of available models.
        """
        pass

    @check_model_loaded
    def get_layer_names(self) -> List[str]:
        """
        Return a list of layer names in the model.
        """

        _, eval_nodes = get_graph_node_names(self.model)
        return eval_nodes

    @check_model_loaded
    def get_layer(self, layer_name: str) -> nn.Module:
        """
        Retrieve a specific layer module based on its name.

        Examples:
            >>> model = StandardModel(model_name='resnet18', num_classes=10)
            >>> model.get_layer('layer1.0.conv1')
        """
        return dict(self.model.named_modules()).get(layer_name, None)

    @check_model_loaded
    def set_layer(self, layer_name: str, new_layer: nn.Module):
        """
        Replace a specific layer module based on its name with a new module.

        Examples:
            >>> model = StandardModel(model_name='resnet18', num_classes=10)
            >>> model.set_layer('layer1.0.conv1', nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False))
        """

        # Obtain the parent module and the attribute name of the layer
        parent, name = self._get_parent_module_and_name(layer_name)
        # Set the new layer
        setattr(parent, name, new_layer)

    @check_model_loaded
    def add_layer(self, new_layer: nn.Module, position: int = -1, inplace: bool = True) -> Optional[nn.Module]:
        """
        Inserts a new layer into the model at the specified position.

        Args:
            new_layer (nn.Module): The new layer to be added.
            position (int): The position at which the new layer should be added. By default, it is added at the end.
            inplace (bool): Whether to add the layer in-place or not. If set to False, a new model is created.

        Returns:
            Optional[nn.Module]: The new model if inplace is set to False.

        """

        if not isinstance(self.model, nn.Sequential):
            # convert the model to a Sequential model
            self.model = nn.Sequential(self.model)

        layers = list(self.model.children())

        # Check if the position is out of bounds
        if position < -1 or position > len(layers):
            raise ValueError(f"Invalid position: {position}")

        if position == -1 or position == len(layers):
            layers.append(new_layer)

        else:
            layers.insert(position, new_layer)

        if inplace:
            self.model = nn.Sequential(*layers)
        else:
            return nn.Sequential(*layers)

    @check_model_loaded
    def _get_parent_module_and_name(self, layer_name: str) -> Tuple[nn.Module, str]:
        """
        Helper method to get the parent module and the attribute name of a layer.
        """

        if '.' in layer_name:
            parent_name, child_name = layer_name.rsplit('.', 1)
            parent = dict(self.model.named_modules()).get(
                parent_name, self.model)
        else:
            parent = self.model
            child_name = layer_name
        return parent, child_name

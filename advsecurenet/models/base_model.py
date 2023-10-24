from abc import ABC, abstractmethod
import torch.nn as nn
import torch


class BaseModel(ABC, nn.Module):
    """
    Abstract class for models.

    Attributes:
        num_classes (int): The number of classes in the dataset.
        pretrained (bool): Whether to load the pretrained weights or not.
        target_layer (str): The name of the layer to be used as the target layer.
    """

    def __init__(self, num_classes=1000, pretrained=False, target_layer=None, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.pretrained = pretrained
        self.target_layer = target_layer

        self.load_model()

    @abstractmethod
    def load_model(self, *args, **kwargs):
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
        return self.model(x)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predicts the class of the input tensor.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The predicted class.
        """
        # return result and the probability
        return self.forward(x).argmax(dim=1), nn.Softmax(dim=1)(self.forward(x)).max(dim=1)[0]

    @abstractmethod
    def models(self):
        """
        Return a list of available models.
        """
        pass

    def get_layer_names(self):
        """
        Return a list of layer names in the model.
        """
        return [name for name, _ in self.model.named_modules()]

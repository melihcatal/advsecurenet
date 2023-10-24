import torchvision.models as models
from advsecurenet.models.base_model import BaseModel
import torch.nn as nn


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

    """

    def __init__(self, model_variant, num_input_channels=3, **kwargs):
        self.model_variant = model_variant
        self.num_input_channels = num_input_channels

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
            # TODO: Add support for other pretrained weights
            weights = 'IMAGENET1K_V1'
        else:
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

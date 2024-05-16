import importlib.util
import os

import torch

from advsecurenet.models.base_model import BaseModel
from advsecurenet.shared.types.configs.model_config import ExternalModelConfig


class ExternalModel(BaseModel):
    """
    This class is used to load external models that are not provided by the package. These models are loaded from external Python files.
    """

    def __init__(self,
                 config: ExternalModelConfig,
                 **kwargs):

        self.model_name = config.model_name
        self.model_arch_path = config.model_arch_path
        self.pretrained = config.pretrained
        self.model_weights_path = config.model_weights_path

        self.model = None
        super().__init__()

    def load_model(self):
        """
        Loads the external model from the specified path.
        """
        if not os.path.exists(self.model_arch_path):
            raise FileNotFoundError(
                f"Model architecture file not found at {self.model_arch_path}")

        # Dynamically import the external model based on its path
        spec = importlib.util.spec_from_file_location(
            self.model_name, self.model_arch_path)
        custom_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(custom_module)

        # Assume the model class inside the external model file has the same name as the file
        if not hasattr(custom_module, self.model_name):
            raise ValueError(
                f"Model class {self.model_name} not found in module {self.model_arch_path}")

        model_class = getattr(custom_module, self.model_name)

        self.model = model_class()
        if self.pretrained:
            try:
                self.model.load_state_dict(torch.load(
                    self.model_weights_path))
            except Exception as e:
                raise ValueError(
                    f"Error loading model weights! Details: {e}") from e

    def models(self):
        """
        Returns a list of available external models.
        """
        raise NotImplementedError(
            "This method is not applicable for external models.")

import importlib.util
import os
from types import ModuleType

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

        self._model_name = config.model_name
        self._model_arch_path = config.model_arch_path
        self._pretrained = config.pretrained
        self._model_weights_path = config.model_weights_path
        self._kwargs = kwargs

        self.model = None
        super().__init__()

    def load_model(self):
        """
        Loads the external model from the specified path.
        """
        if not os.path.exists(self._model_arch_path):
            raise FileNotFoundError(
                f"Model architecture file not found at {self._model_arch_path}")

        # Dynamically import the external model based on its path
        spec = importlib.util.spec_from_file_location(
            self._model_name, self._model_arch_path)
        custom_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(custom_module)

        # Assume the model class inside the external model file has the same name as the file
        self._ensure_model_class_exists(custom_module)

        model_class = getattr(custom_module, self._model_name)

        self.model = model_class()
        if self._pretrained:
            try:
                self.model.load_state_dict(torch.load(
                    self._model_weights_path))
            except Exception as e:
                raise ValueError(
                    f"Error loading model weights! Details: {e}") from e

    def models(self):
        """
        Returns a list of available external models.
        """
        raise NotImplementedError(
            "This method is not applicable for external models.")

    def _ensure_model_class_exists(self, module: ModuleType) -> None:
        """
        Ensures that the model class exists within the given module.

        Args:
            module (ModuleType): The module to check for the model class.

        Raises:
            ValueError: If the model class is not found within the module.
        """
        if not hasattr(module, self._model_name):
            raise ValueError(
                f"Model class {self._model_name} not found in module {self._model_arch_path}")

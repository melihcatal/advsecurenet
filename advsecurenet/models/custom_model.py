import importlib
import os

from advsecurenet.models.base_model import BaseModel


class CustomModel(BaseModel):
    """
    This class is used to load a custom model. It is a subclass of BaseModel. 

    Parameters
    ----------
    model_name : str
        The name of the custom model. This should be the same as the name of the file in the CustomModels directory.
    custom_models_path : str, optional
        The path to the CustomModels directory, by default "CustomModels"

    Raises
    ------
    ValueError
        If the model class is not found in the custom model file.

    Examples
    --------
    >>> from advsecurenet.models.custom_model import CustomModel
    >>> model = CustomModel("CustomMnistModel")
    >>> model
    CustomMnistModel(
        (model): Sequential(
            (0): Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1))
            (1): ReLU()
            (2): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1))
            (3): ReLU()
            (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
            (5): Dropout(p=0.25, inplace=False)
            (6): Flatten(start_dim=1, end_dim=-1)
            (7): Linear(in_features=9216, out_features=128, bias=True)
            (8): ReLU()
            (9): Dropout(p=0.5, inplace=False)
            (10): Linear(in_features=128, out_features=10, bias=True)
        )
    )
    """

    def __init__(self, model_name, custom_models_path="CustomModels", **kwargs):
        self.custom_models_path = custom_models_path
        self.model_name = model_name

        # Initialize the BaseModel
        super().__init__()

    def load_model(self):
        """
        Load the custom model. This method is called by the BaseModel constructor.
        It dynamically imports the custom model based on its name. That is, it assumes that the model class inside the custom model file has the same name as the file
        i.e., for 'CustomMnistModel.py', there should be a class 'CustomMnistModel'.

        Raises
        ------
        ValueError
            If the model class is not found in the custom model file.
        """

        # Dynamically import the custom model based on its name
        custom_module_name = f"advsecurenet.models.{self.custom_models_path}.{self.model_name}"
        custom_module = importlib.import_module(custom_module_name)

        # Assume the model class inside the custom model file has the same name as the file
        if not hasattr(custom_module, self.model_name):
            raise ValueError(
                f"Model class {self.model_name} not found in module {custom_module_name}")

        model_class = getattr(custom_module, self.model_name)

        self.model = model_class()

        # Perform necessary modifications after model load
        self.modify_model()

    def modify_model(self):
        # Usually, for custom models, you might not need modifications after loading.
        # But if you do, you can specify them here.
        pass

    @staticmethod
    def models():
        """
        Returns a list of available custom models from the CustomModels directory.

        Returns
        -------
        List[str]
            A list of available custom models.
        """

        # Path to the CustomModels directory
        custom_models_dir = os.path.join(os.path.dirname(
            os.path.abspath(__file__)), "CustomModels")

        # List all python files in the CustomModels directory
        all_files = [f for f in os.listdir(custom_models_dir) if os.path.isfile(
            os.path.join(custom_models_dir, f)) and f.endswith('.py')]

        # Extract the model names by stripping the '.py' from filenames
        model_names = [os.path.splitext(f)[0] for f in all_files]

        # remove __init__ from model names
        model_names = [
            model_name for model_name in model_names if model_name != "__init__"]

        return model_names

    @staticmethod
    def available_weights(model_name: str) -> list[str]:
        """
        Not applicable for custom models.
        """
        raise NotImplementedError(
            "This method is not applicable for custom models.")

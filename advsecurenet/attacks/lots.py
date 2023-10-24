from typing import Optional, Tuple, Union
import torch
from tqdm.auto import trange
from torchvision.models.feature_extraction import create_feature_extractor

from advsecurenet.attacks.adversarial_attack import AdversarialAttack
from advsecurenet.shared.types import LotsAttackConfig, LotsAttackMode
from advsecurenet.shared.colors import red, yellow, reset
from advsecurenet.utils import get_device
from advsecurenet.models.base_model import BaseModel


class LOTS(AdversarialAttack):
    """
    LOTS attack

    Args:
        deep_feature_layer (str): The name of the layer to use for the attack.
        mode (LotsAttackMode): The mode to use for the attack. Defaults to LotsAttackMode.ITERATIVE.
        epsilon (float): The epsilon value to use for the attack. Defaults to 0.1.
        learning_rate (float): The learning rate to use for the attack. Defaults to 1./255.
        max_iterations (int): The maximum number of iterations to use for the attack. Defaults to 1000.
        verbose (bool): Whether to print progress of the attack. Defaults to True.
        device (DeviceType): Device to use for the attack. Defaults to DeviceType.CPU.


    References:
           [1] Rozsa, A., Güunther, M., and Boult, T. E. (2017). LOTS about attacking deep features. In International Joint Conference on Biometrics (IJCB), pages 168{176. IEEE. https://arxiv.org/abs/1611.06179

    """

    def __init__(self, config: LotsAttackConfig) -> None:

        self.validate_config(config)

        self.deep_feature_layer: str = config.deep_feature_layer
        self.mode: LotsAttackMode = config.mode
        self.epsilon: float = config.epsilon
        self.learning_rate: float = config.learning_rate
        self.max_iterations: int = config.max_iterations
        self.verbose: bool = config.verbose
        self.device: str = config.device.value if config.device is not None else get_device()

    @staticmethod
    def validate_config(config: LotsAttackConfig) -> None:
        """
        Validate the provided configuration settings.

        :param config: An instance of LotsAttackConfig containing the configuration settings.
        :raises ValueError: If any of the configuration settings are invalid.
        """
        # Validate config type
        if not isinstance(config, LotsAttackConfig):
            raise ValueError(
                "Invalid config type provided. Expected LotsAttackConfig. But got: " + str(type(config)))

        # Validate mode type
        if not isinstance(config.mode, LotsAttackMode):
            allowed_modes = ", ".join([mode.value for mode in LotsAttackMode])
            raise ValueError(
                f"Invalid mode type provided. Allowed modes are: {allowed_modes}")

        # Validate epsilon, learning rate, and max_iterations
        for attribute, name in [("epsilon", "Epsilon"),
                                ("learning_rate", "Learning rate"),
                                ("max_iterations", "Max iterations")]:
            value = getattr(config, attribute)
            if value is not None and value < 0:
                raise ValueError(
                    f"Invalid {name.lower()} value provided. {name} must be greater than 0.")

        # Validate deep_feature_layer
        if config.deep_feature_layer is None:
            raise ValueError(
                "Deep feature layer that you want to use for the attack must be provided.")

    def attack(self, model: BaseModel, data: torch.Tensor, target: torch.Tensor, target_classes: Optional[Union[torch.Tensor, int]] = None) -> Tuple[torch.Tensor, bool]:
        """
        Generates adversarial examples using the LOTS attack. Based on the provided mode, either the iterative or single attack will be used. If the iterative attack is used, the attack will be run for the specified number of iterations. If the single attack is used, the attack will be run for a single iteration.

        Args:
            model (BaseModel): The model to attack.
            data (torch.tensor): The original input tensor. Expected shape is (batch_size, channels, height, width).
            target (torch.tensor): The target tensor. Expected shape is (batch_size, channels, height, width).
            target_classes (torch.tensor): The target classes tensor. Expected shape is (batch_size,).

        Returns:
            torch.tensor: The adversarial example tensor.
        """
        data = data.clone().detach().to(self.device)
        target = target.clone().detach().to(self.device)

        if self.mode == LotsAttackMode.ITERATIVE:
            return self._lots_iterative(model, data, target, target_classes)
        if self.mode == LotsAttackMode.SINGLE:
            return self._lots_single(model, data, target, target_classes)

        # if we reach here, the mode is invalid
        raise ValueError("Invalid mode provided.")

    def _lots_iterative(self, network: BaseModel, data: torch.Tensor, target: torch.Tensor, target_classes: torch.Tensor) -> Tuple[torch.Tensor, bool]:

        feature_extractor_model = create_feature_extractor(
            network, {self.deep_feature_layer: "deep_feature_layer"})
        target = feature_extractor_model.forward(target)["deep_feature_layer"]

        if target_classes is not None:
            if not torch.is_tensor(target_classes):
                target_classes = torch.tensor(
                    [target_classes] * data.size(0)).to(data.device)
        else:
            target_classes = torch.tensor([-1] * data.size(0)).to(data.device)

        # Make data a Parameter so it can be updated by optimizer
        data = torch.nn.Parameter(data)

        # Create an optimizer for the data
        optimizer = torch.optim.Adam([data], lr=self.learning_rate)

        for _ in trange(self.max_iterations, desc=f"{red}Running LOTS{reset}", bar_format="{l_bar}%s{bar}%s{r_bar}" % (yellow, reset), leave=False, position=0, disable=not self.verbose):
            optimizer.zero_grad()

            logits = network.forward(data)
            features = feature_extractor_model.forward(
                data)["deep_feature_layer"]

            with torch.no_grad():
                pred_classes = torch.argmax(logits, dim=-1)
                success_indices = pred_classes == target_classes
                if self.epsilon is not None:
                    distances = torch.norm(features - target, dim=1)
                    success_distances = distances < self.epsilon
                    if success_indices.all() or success_distances.all():
                        data = torch.clamp(data, 0, 1)
                        return data.detach(), True

            loss = torch.nn.functional.mse_loss(
                features, target, reduction="sum")
            loss.backward(retain_graph=True)

            optimizer.step()

            # Clipping data to ensure it remains in [0, 1]
            data.data.clamp_(0, 1)

        return data.detach(), False

    def _lots_single(self, network: BaseModel, data: torch.Tensor, target: torch.Tensor, target_classes: torch.Tensor) -> Tuple[torch.Tensor, bool]:

        feature_extractor_model = create_feature_extractor(
            network, {self.deep_feature_layer: "deep_feature_layer"})
        target = feature_extractor_model.forward(target)["deep_feature_layer"]

        # Convert data into a Parameter so it can be updated by optimizer
        data = torch.nn.Parameter(data)

        # Create an optimizer for the data
        optimizer = torch.optim.Adam([data], lr=self.learning_rate)

        network.zero_grad()
        features = feature_extractor_model.forward(data)["deep_feature_layer"]
        loss = torch.nn.functional.mse_loss(features, target, reduction="mean")
        loss.backward(retain_graph=True)

        optimizer.step()

        data = torch.clamp(data, 0, 1)
        logits = network.forward(data)

        if target_classes is not None:
            if not torch.is_tensor(target_classes):
                target_classes = torch.tensor(
                    [target_classes] * data.size(0)).to(data.device)
        else:
            target_classes = torch.tensor([-1] * data.size(0)).to(data.device)

        pred_classes = torch.argmax(logits, dim=-1)
        success_indices = pred_classes == target_classes

        if self.epsilon is not None:
            distances = torch.norm(features - target, dim=1)
            success_distances = distances < self.epsilon

            is_sucess = success_indices.all() or success_distances.all()
            return data.detach(), is_sucess
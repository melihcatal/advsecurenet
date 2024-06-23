import typing
import warnings
from typing import Optional

import torch
from torchvision.models.feature_extraction import create_feature_extractor

from advsecurenet.attacks.base.adversarial_attack import AdversarialAttack
from advsecurenet.models.base_model import BaseModel
from advsecurenet.shared.types.configs.attack_configs import (LotsAttackConfig,
                                                              LotsAttackMode)

# create_feature_extractor raises a warning that is not relevant to the user
warnings.filterwarnings("ignore", message="'has_cuda' is deprecated")
warnings.filterwarnings("ignore", message="'has_cudnn' is deprecated")
warnings.filterwarnings("ignore", message="'has_mps' is deprecated")
warnings.filterwarnings("ignore", message="'has_mkldnn' is deprecated")


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
        device (torch.device): Device to use for the attack. Defaults to "cpu".


    References:
           [1] Rozsa, A., Güunther, M., and Boult, T. E. (2017). LOTS about attacking deep features. In International Joint Conference on Biometrics (IJCB), pages 168{176. IEEE. https://arxiv.org/abs/1611.06179

    """

    def __init__(self, config: LotsAttackConfig) -> None:
        self._validate_config(config)
        self._deep_feature_layer: str = config.deep_feature_layer
        self._mode: LotsAttackMode = config.mode
        self._epsilon: float = config.epsilon
        self._learning_rate: float = config.learning_rate
        self._max_iterations: int = config.max_iterations
        self._verbose: bool = config.verbose
        super().__init__(config)

    @typing.no_type_check
    def attack(self,
               model: BaseModel,
               x: torch.Tensor,
               y: Optional[torch.Tensor] = None,
               x_target: torch.Tensor = None) -> torch.Tensor:
        """
        Generates adversarial examples using the LOTS attack. Based on the provided mode, either the iterative or single attack will be used.

        Args:
            model (BaseModel): The model to attack.
            x (torch.Tensor): The original input tensor. Shape: (batch_size, channels, height, width).
            x_target (torch.Tensor): The x_target tensor. Shape: (batch_size, channels, height, width).
            y (torch.Tensor, optional): The target classes tensor. Shape: (batch_size,).

        Returns:
            torch.Tensor: The adversarial example tensor.
        """
        if self.device_manager.distributed_mode:
            model = model.module

        self._validate_layer(model)

        x, x_target = self._prepare_inputs(x, x_target)
        feature_extractor_model = self._create_feature_extractor(model)

        x = torch.nn.Parameter(x)
        optimizer = torch.optim.Adam([x], lr=self._learning_rate)

        if self._mode == LotsAttackMode.ITERATIVE:
            return self._lots_iterative(model, x, x_target, feature_extractor_model, optimizer, y)
        elif self._mode == LotsAttackMode.SINGLE:
            return self._lots_single(x, x_target, feature_extractor_model, optimizer)
        else:
            raise ValueError("Invalid mode provided.")

    def _lots_iterative(self,
                        model: torch.nn.Module,
                        x: torch.Tensor,
                        x_target: torch.Tensor,
                        feature_extractor_model: torch.nn.Module,
                        optimizer: torch.optim.Optimizer,
                        y: Optional[torch.Tensor] = None
                        ) -> torch.Tensor:
        """
        Performs the LOTS iterative attack.

        Args:
            model (torch.nn.Module): The target model to be attacked.
            x (torch.Tensor): The input image to be perturbed.
            x_target (torch.Tensor): The target image to be matched.
            feature_extractor_model (torch.nn.Module): The feature extractor model.
            optimizer (torch.optim.Optimizer): The optimizer used to update the input image.
            y (Optional[Union[torch.Tensor, int]], optional): The target classes for the attack. Defaults to None.

        Returns:
            torch.Tensor: The perturbed input image.
        """
        x_target_deep_feature = feature_extractor_model(x_target)[
            "deep_feature_layer"]
        successes = torch.zeros(x.size(0), dtype=torch.bool, device=x.device)

        for _ in range(self._max_iterations):
            x.requires_grad = True
            model.zero_grad()
            optimizer.zero_grad()
            x_deep_features = feature_extractor_model(x)["deep_feature_layer"]

            if y is not None:
                with torch.no_grad():
                    successes |= self._evaluate_adversarial_success(
                        model, x, x_deep_features, x_target_deep_feature, y)
                    if all(successes):
                        return x.clamp(0, 1).detach()

            loss = torch.nn.functional.mse_loss(
                x_deep_features, x_target_deep_feature, reduction="sum")

            loss.backward(retain_graph=True)
            optimizer.step()
        return x.clamp(0, 1).detach()

    def _lots_single(self,
                     x: torch.Tensor,
                     x_target: torch.Tensor,
                     feature_extractor_model: torch.nn.Module,
                     optimizer: torch.optim.Optimizer
                     ) -> torch.Tensor:
        """
        Performs a single iteration of the LOTS attack.

        Args:
            model (torch.nn.Module): The target model to be attacked.
            x (torch.Tensor): The input image tensor.
            x_target (torch.Tensor): The target image tensor.
            feature_extractor_model (torch.nn.Module): The feature extractor model.
            optimizer (torch.optim.Optimizer): The optimizer used for gradient descent.

        Returns:
            torch.Tensor: The perturbed image tensor after the attack.
        """
        x_target_deep_feature = feature_extractor_model(x_target)[
            "deep_feature_layer"]
        x_deep_features = feature_extractor_model(x)["deep_feature_layer"]

        loss = torch.nn.functional.mse_loss(
            x_deep_features, x_target_deep_feature, reduction="mean")
        loss.backward(retain_graph=True)
        optimizer.step()

        return x.clamp(0, 1).detach()

    def _evaluate_adversarial_success(self,
                                      model: torch.nn.Module,
                                      x: torch.Tensor,
                                      x_deep_features: torch.Tensor,
                                      x_target_deep_features: torch.Tensor,
                                      y: Optional[torch.Tensor] = None
                                      ) -> torch.Tensor:
        """
        Evaluates the success of adversarial attacks. 
        If the epsilon value is provided, the distance between the deep features of the input and target tensors is calculated and compared with the epsilon value.
        If the epsilon value is not provided, the predicted classes are compared with the target classes.

        Args:
            model (torch.nn.Module): The model used for evaluation.
            x (torch.Tensor): The input tensor.
            x_deep_features (torch.Tensor): The deep features of the input tensor.
            x_target_deep_features (torch.Tensor): The deep features of the target tensor.
            y (torch.Tensor): The target classes.

        Returns:
            torch.Tensor: A tensor indicating the success of adversarial attacks.
        """
        success_indices = torch.zeros(
            x.size(0), dtype=torch.bool, device=x.device)
        logits = model(x)
        pred_classes = torch.argmax(logits, dim=-1)
        pred_classes = self.device_manager.to_device(pred_classes)

        if y is not None:
            y = self.device_manager.to_device(y)
            success_indices = pred_classes == y

        if self._epsilon is not None:
            distances = torch.norm(
                x_deep_features - x_target_deep_features, dim=1)
            print(distances)

            success_distances = distances < self._epsilon
            return success_indices | success_distances

        return success_indices

    def _validate_config(self, config: LotsAttackConfig) -> None:
        """
        Validate the provided configuration settings.

        Args:
            config (LotsAttackConfig): The configuration settings to validate.

        Raises:
            ValueError: If the provided configuration settings are invalid.
        """
        # Validate config type
        if not isinstance(config, LotsAttackConfig):
            raise ValueError(
                "Invalid config type provided. Expected LotsAttackConfig. But got: " + str(type(config)))
        # Validate mode type
        if isinstance(config.mode, str):
            try:
                config.mode = LotsAttackMode[config.mode.upper()]
            except KeyError:
                pass  # Will be handled by the next check

        if not isinstance(config.mode, LotsAttackMode):
            allowed_modes = ", ".join(mode.value for mode in LotsAttackMode)
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

    def _validate_layer(self, model: BaseModel) -> None:
        """
        Validate the provided layer name.

        Parameters:
            model (BaseModel): The model to validate the layer against.

        Raises:
            ValueError: If the provided layer name is not found in the model.

        Note:
            The layer name should be prefixed with "model.". However, while checking, we need to temporarily remove the prefix before checking.

        """
        # the layer name should be prefixed with "model.". However, while checking, we need to temporarily remove the prefix before checking
        layer = self._deep_feature_layer

        if layer.replace("model.", "") not in model.get_layer_names():
            raise ValueError(
                f"Layer '{layer}' not found in the model. Please provide a valid layer name.")

        # if the layer name is valid but not prefixed with "model.", we add the prefix
        if not layer.startswith("model."):
            self._deep_feature_layer = f"model.{layer}"

    def _prepare_inputs(self, x: torch.Tensor, x_target: torch.Tensor) -> tuple:
        """
        Prepare the input tensors for the attack. Original and target images are moved to the device and detached from the computation graph.

        Args:
            x (torch.Tensor): The original input tensor.
            x_target (torch.Tensor): The target input tensor.

        Returns:
            tuple: A tuple containing the prepared input tensors (x, x_target).
        """
        x = x.clone().detach().requires_grad_(True)
        x_target = x_target.clone().detach().requires_grad_(False)
        x = self.device_manager.to_device(x)
        x_target = self.device_manager.to_device(x_target)
        return x, x_target

    def _create_feature_extractor(self, model: BaseModel) -> torch.nn.Module:
        """
        Creates a feature extractor module based on the given model. Feature extractor is used to extract deep features from the input images.

        Args:
            model (BaseModel): The base model to create the feature extractor from.

        Returns:
            torch.nn.Module: The feature extractor module.

        """
        feature_extractor_model = create_feature_extractor(
            model,
            {self._deep_feature_layer: "deep_feature_layer"})
        return self.device_manager.to_device(feature_extractor_model)

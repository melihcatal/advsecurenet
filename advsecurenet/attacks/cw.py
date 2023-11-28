import torch
from torch import optim

from advsecurenet.attacks.adversarial_attack import AdversarialAttack
from advsecurenet.models.base_model import BaseModel
from advsecurenet.shared.types.configs.attack_configs import CWAttackConfig


class CWAttack(AdversarialAttack):
    """
    Carlini-Wagner L2 attack

    Args:

        targeted (bool): If True, targets the attack to the specified label. Defaults to False.
        c_init (float): Initial value of c. Defaults to 0.1.
        kappa (float): Confidence parameter for CW loss. Defaults to 0.
        learning_rate (float): Learning rate for the Adam optimizer. Defaults to 0.01.
        max_iterations (int): Maximum number of iterations for the Adam optimizer. Defaults to 10.
        abort_early (bool): If True, aborts the attack early if the loss stops decreasing. Defaults to False.
        binary_search_steps (int): Number of binary search steps. Defaults to 10.
        device (torch.device): Device to use for the attack. Defaults to "cpu".
        clip_min (float): Minimum value of the input. Defaults to 0.
        clip_max (float): Maximum value of the input. Defaults to 1.
        c_lower (float): Lower bound for c. Defaults to 1e-6.
        c_upper (float): Upper bound for c. Defaults to 1.
        patience (int): Number of iterations to wait before aborting early. Defaults to 5.
        verbose (bool): If True, prints progress of the attack. Defaults to True.


    References:
            [1] Carlini, Nicholas, and David Wagner. "Towards evaluating the robustness of neural networks." 2017 IEEE Symposium on Security and Privacy (SP). IEEE, 2017.

    """

    def __init__(self, config: CWAttackConfig) -> None:
        self.c_init: float = config.c_init
        self.kappa: float = config.kappa
        self.learning_rate: float = config.learning_rate
        self.max_iterations: int = config.max_iterations
        self.abort_early: bool = config.abort_early
        self.targeted: bool = config.targeted
        self.binary_search_steps: int = config.binary_search_steps
        self.clip_min: float = config.clip_min
        self.clip_max: float = config.clip_max
        self.c_lower: float = config.c_lower
        self.c_upper: float = config.c_upper
        self.patience: int = config.patience
        self.verbose: bool = config.verbose
        super().__init__(config)

    def attack(self,
               model: BaseModel,
               x: torch.Tensor,
               y: torch.Tensor,
               *args, **kwargs
               ) -> torch.Tensor:
        """
        Performs the Carlini-Wagner L2 attack on the specified model and input.

        Args:
            model (BaseModel): Model to attack.
            x (torch.Tensor): Batch of inputs to attack.
            y (torch.Tensor): Label of the input. If targeted is True, the attack will try to make the model predict this label. Otherwise, the attack will try to make the model predict any other label than this one.

        Returns:
            torch.Tensor: Adversarial example.
        """
        batch_size = x.shape[0]
        image = x.clone().detach()
        label = y.clone().detach()
        # image = self.device_manager.to_device(image)
        # label = self.device_manager.to_device(label)
        # model = self.device_manager.to_device(model)

        image = self._initialize_x(image)

        c_lower = torch.full((batch_size,), self.c_lower,
                             device=self.device_manager.get_current_device(), dtype=torch.float32)
        c_upper = torch.full((batch_size,), self.c_upper,
                             dtype=torch.float32, device=self.device_manager.get_current_device())
        self.c = torch.full((batch_size,), self.c_init,
                            device=self.device_manager.get_current_device(), dtype=torch.float32)

        best_adv_images = image.clone()
        best_perturbations = torch.full(
            (batch_size,), float("inf"), device=self.device_manager.get_current_device())

        for _ in range(self.binary_search_steps):
            # for _ in trange(self.binary_search_steps, desc="Binary Search Steps", disable=not self.verbose):
            adv_images = self._run_attack(model, image, label)
            successful_mask = self._is_successful(model, adv_images, label)

            # Calculate L2 perturbations
            perturbations = torch.norm(adv_images - image, dim=[1, 2, 3], p=2)

            # Update based on smaller perturbation and successful attack
            improved_mask = (
                perturbations < best_perturbations) & successful_mask
            best_adv_images[improved_mask] = adv_images[improved_mask]
            best_perturbations[improved_mask] = perturbations[improved_mask]

            # Update c values - if attack was successful, make c smaller, else make c larger
            c_upper[successful_mask] = torch.min(
                c_upper[successful_mask], self.c[successful_mask])
            c_lower[~successful_mask] = torch.max(
                c_lower[~successful_mask], self.c[~successful_mask])
            self.c = (c_lower + c_upper) / 2

        return best_adv_images.detach()

    def _initialize_x(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.clip_min) / (self.clip_max - self.clip_min)
        x = torch.clamp(x, 0, 1)
        x = x * 2 - 1
        x = torch.arctanh(x * 0.999999)
        return x

    def _run_attack(self, model: BaseModel, image: torch.Tensor, label: torch.Tensor) -> torch.Tensor:

        perturbation = torch.zeros_like(
            image, requires_grad=True)
        # perturbation = self.device_manager.to_device(perturbation)
        optimizer = optim.Adam([perturbation], lr=self.learning_rate)
        patience_counter = 0
        min_loss = float('inf')
        for iteration in range(self.max_iterations):

            adv_image_tanh = torch.tanh(image + perturbation)
            adv_image = (adv_image_tanh + 1) / 2
            loss = self._cw_loss(model, adv_image, label, perturbation)
            loss = loss.mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Early stopping based on patience
            if loss.item() < min_loss:
                min_loss = loss.item()
                patience_counter = 0
            else:
                patience_counter += 1

            # TODO: Better early stopping
            if self.abort_early and patience_counter >= self.patience:
                print("Early stopping at iteration: ", iteration)
                break
        return adv_image.detach()

    def _cw_loss(self, model: BaseModel, x_adv: torch.Tensor, label: torch.Tensor, perturbation: torch.Tensor) -> torch.Tensor:
        f_val = self._f(model, x_adv, label)
        return torch.norm(perturbation, p=2) + self.c * f_val

    def _f(self, model: BaseModel, x: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        outputs = model(x)
        eye_matrix = torch.eye(len(outputs[0]))
        eye_matrix = self.device_manager.to_device(eye_matrix)
        one_hot_labels = eye_matrix[label]
        # maximum value of all classes except the target class
        i, _ = torch.max((1 - one_hot_labels) * outputs, dim=1)
        # prediction score of the target class
        j = torch.masked_select(outputs, one_hot_labels.bool())
        if self.targeted:
            # return i - j + self.kappa or 0 max of these two
            return torch.clamp(i - j + self.kappa, min=0)
            # return torch.clamp(i - j, min=-self.kappa)
        else:
            return torch.clamp(j - i + self.kappa, min=0)

    def _is_successful(self, model: BaseModel, adv_image: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        outputs = model(adv_image)
        predicted = torch.argmax(outputs, 1)
        if self.targeted:
            return predicted == label
        return predicted != label

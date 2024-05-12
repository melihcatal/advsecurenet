import torch

from advsecurenet.evaluation.base_evaluator import BaseEvaluator


class PerturbationDistanceEvaluator(BaseEvaluator):
    """
    Evaluator for the perturbation distance. Supported metrics are:
    - L0
    - L2
    - Linf
    """

    def __init__(self):
        self.total_l0_distance = 0
        self.total_l2_distance = 0
        self.total_l_inf_distance = 0
        self.batch_size = 0

    def reset(self):
        """
        Resets the evaluator for a new streaming session.
        """
        self.total_l0_distance = 0
        self.total_l2_distance = 0
        self.total_l_inf_distance = 0
        self.batch_size = 0

    def update(self, original_images: torch.Tensor, adversarial_images: torch.Tensor):
        """
        Updates the evaluator with new data for streaming mode. Expects the unnormalized, original distribution of the data.

        Args:
            original_images (torch.Tensor): The original images.
            adversarial_images (torch.Tensor): The adversarial images.
        """
        l0_distance, l2_distance, l_inf_distance = self.calculate_perturbation_distances(
            original_images, adversarial_images)

        self.total_l0_distance += l0_distance
        self.total_l2_distance += l2_distance
        self.total_l_inf_distance += l_inf_distance
        self.batch_size += 1

    def get_results(self) -> dict[str, float]:
        """
        Calculates the mean perturbation distances for the streaming session.

        Returns:
            dict[str, float]: The mean perturbation distances for the adversarial examples in the streaming session.
        """
        if self.batch_size > 0:
            return {
                "L0": self.total_l0_distance / self.batch_size,
                "L2": self.total_l2_distance / self.batch_size,
                "Linf": self.total_l_inf_distance / self.batch_size
            }
        return {
            "L0": 0.0,
            "L2": 0.0,
            "Linf": 0.0
        }

    def get_perturbation_distance(self, distance_type: str) -> float:
        """
        Calculates the mean perturbation distance for the streaming session for the specified distance type.

        Args:
            distance_type (str): The distance type. Valid values are: L0, L2, Linf.

        Returns:
            float: The mean perturbation distance for the adversarial examples in the streaming session.
        """
        if self.batch_size > 0:
            if distance_type == "L0":
                return self.total_l0_distance
            elif distance_type == "L2":
                return self.total_l2_distance
            elif distance_type == "Linf":
                return self.total_l_inf_distance
            else:
                raise ValueError(
                    "Invalid distance type. Valid values are: L0, L2, Linf.")
        else:
            return 0.0

    def calculate_l0_distance(self, original_images: torch.Tensor, adversarial_images: torch.Tensor) -> float:
        """
        Calculates the L0 distance between the original and adversarial images. L0 distance is the count of pixels that are different between the two images (i.e. the number of pixels that have been changed in the adversarial image compared to the original image).

        Args:
            original_images (torch.Tensor): The original images.
            adversarial_images (torch.Tensor): The adversarial images.

        Returns:
            float: The mean L0 distance between the original and adversarial images.
        """

        # Calculating the L0 distance
        l0_distance = (original_images !=
                       adversarial_images).sum(dim=(1, 2, 3))

        # Convert to floating point before taking the mean
        l0_distance = l0_distance.float().mean()

        return l0_distance.item()

    def calculate_l2_distance(self, original_images: torch.Tensor, adversarial_images: torch.Tensor) -> float:
        """ 
        Calculates the L2 distance between the original and adversarial images. L2 distance is the Euclidean distance between the two images.

        Args:
            original_images (torch.Tensor): The original images.
            adversarial_images (torch.Tensor): The adversarial images.

        Returns:
            float: The mean L2 distance between the original and adversarial images.
        """
        l2_distance = torch.norm(
            (original_images - adversarial_images).view(original_images.shape[0], -1), p=2, dim=1).mean()
        return l2_distance.item()

    def calculate_l_inf_distance(self, original_images: torch.Tensor, adversarial_images: torch.Tensor) -> float:
        """
        Calculates the L∞ distance between the original and adversarial images. L∞ distance is the maximum absolute difference between the two images in any pixel.

        Args:
            original_images (torch.Tensor): The original images.
            adversarial_images (torch.Tensor): The adversarial images.

        Returns:
            float: The mean L∞ distance between the original and adversarial images.
        """
        l_inf_distance = (original_images - adversarial_images).view(
            original_images.shape[0], -1).abs().max(dim=1)[0].mean()
        return l_inf_distance.item()

    def calculate_perturbation_distances(self, original_images: torch.Tensor, adversarial_images: torch.Tensor) -> tuple[float, float, float]:
        """
        Calculates the L0, L2, and L∞ distances between the original and adversarial images. 

        Args:
            original_images (torch.Tensor): The original images.
            adversarial_images (torch.Tensor): The adversarial images.

        Returns:
            Tuple[float, float, float]: The mean L0, L2, and L∞ distances between the original and adversarial images.
        """
        l0_distance = self.calculate_l0_distance(
            original_images, adversarial_images)
        l2_distance = self.calculate_l2_distance(
            original_images, adversarial_images)
        l_inf_distance = self.calculate_l_inf_distance(
            original_images, adversarial_images)
        return l0_distance, l2_distance, l_inf_distance

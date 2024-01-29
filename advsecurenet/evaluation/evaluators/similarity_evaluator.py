import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from advsecurenet.evaluation.base_evaluator import BaseEvaluator


class SimilarityEvaluator(BaseEvaluator):

    """
    Calculates the structural similarity index (SSIM) and peak signal-to-noise ratio (PSNR) between the original and adversarial images.
    This evaluator supports both streaming and non-streaming modes. In streaming mode, the evaluator can be updated with new data and the results are calculated on the fly.
    In non-streaming mode, the evaluator returns the results for the provided data only.

    For streaming mode, the evaluator can be updated with new data using the `update` method. The results can be obtained using the `get_results` method. 

    Note:
        The SSIM and PSNR metrics expect the images to be in the original range. If the images are normalized, they need to be denormalized before calculating the metrics. 

    Example:
        >>> from advsecurenet.evaluation.evaluators.similarity_evaluator import SimilarityEvaluator
        >>> with SimilarityEvaluator() as evaluator:
        >>>    for batch in data_loader:
        >>>        # Generate adversarial images
        >>>        ...
        >>>        # Update evaluator with new data
        >>>        evaluator.update(original_images, adversarial_images)
        >>>    # Get results
        >>>    ssim_score, psnr_score = evaluator.get_results()

    For non-streaming mode, the evaluator can be used as a normal function.

    Example:
        >>> from advsecurenet.evaluation.evaluators.similarity_evaluator import SimilarityEvaluator
        >>> similarity_evaluator = SimilarityEvaluator()
        >>> ssim_score, psnr_score = similarity_evaluator.calculate_similarity_scores(original_images, adversarial_images)
        >>> print(f"SSIM: {ssim_score}, PSNR: {psnr_score}")

    """

    def __init__(self):
        self.ssim_score = 0
        self.psnr_score = 0
        self.total_images = 0
        self.total_batches = 0

    def reset(self):
        """
        Resets the evaluator.
        """
        self.ssim_score = 0
        self.psnr_score = 0
        self.total_images = 0
        self.total_batches = 0

    def update_ssim(self, original_images: torch.Tensor, adversarial_images: torch.Tensor, update_total_images: bool = True):
        """
        Updates the evaluator with new data.

        Args:
            original_images (torch.Tensor): The original images. Expected shape is (batch_size, channels, height, width).
            adversarial_images (torch.Tensor): The adversarial images.
        """
        ssim_score = self.calculate_ssim(original_images, adversarial_images)
        self.ssim_score += ssim_score
        if update_total_images:
            self.total_images += original_images.shape[0]

    def update_psnr(self, original_images: torch.Tensor, adversarial_images: torch.Tensor, update_total_images: bool = True):
        """
        Updates the evaluator with new data.

        Args:
            original_images (torch.Tensor): The original images. Expected shape is (batch_size, channels, height, width).
            adversarial_images (torch.Tensor): The adversarial images.
        """
        psnr_score = self.calculate_psnr(original_images, adversarial_images)
        self.psnr_score += psnr_score
        if update_total_images:
            self.total_images += original_images.shape[0]

    def update(self, original_images: torch.Tensor, adversarial_images: torch.Tensor):
        """
        Updates the evaluator with new data for both SSIM and PSNR.

        Args:
            original_images (torch.Tensor): The original images. Expected shape is (batch_size, channels, height, width).
            adversarial_images (torch.Tensor): The adversarial images.
        """
        self.update_ssim(original_images, adversarial_images, False)
        self.update_psnr(original_images, adversarial_images, False)
        self.total_batches += 1

    def get_ssim(self) -> float:
        """
        Calculates the mean SSIM between the original and adversarial images for all the data seen so far.

        Returns:
            float: The mean SSIM between the original and adversarial images. [-1, 1] range. 1 means the images are identical.
        """
        return self.ssim_score

    def get_psnr(self) -> float:
        """
        Calculates the mean PSNR between the original and adversarial images for all the data seen so far.

        Returns:
            float: The mean PSNR between the original and adversarial images. [0, inf) range. Higher values (e.g. 30 dB or more) indicate better quality. Infinite if the images are identical.
        """
        return self.psnr_score

    def get_results(self) -> dict[str, float]:
        """
        Calculates the mean SSIM and PSNR between the original and adversarial images for all the data seen so far.

        Returns:
            dict[str, float]: A dictionary containing the mean SSIM and PSNR between the original and adversarial images.
        """
        return {
            "SSIM": self.get_ssim() / self.total_batches,
            "PSNR": self.get_psnr() / self.total_batches
        }

    def calculate_ssim(self, original_images: torch.Tensor, adversarial_images: torch.Tensor) -> float:
        """
        Calculates the mean structural similarity index (SSIM) between the original and adversarial images. SSIM is a metric that measures the similarity between two images. 
        The higher the SSIM, the more similar the images are. 

        Args:
            original_images (torch.Tensor): The original images. Expected shape is (batch_size, channels, height, width).
            adversarial_images (torch.Tensor): The adversarial images.

        Returns:
            float: The mean SSIM between the original and adversarial images. [-1, 1] range. 1 means the images are identical.
        """
        # Convert tensors to numpy arrays and ensure they are float32
        original_images_np = original_images.cpu().detach().numpy().astype(np.float32)
        adversarial_images_np = adversarial_images.cpu().detach().numpy().astype(np.float32)
        data_range = original_images_np.max() - original_images_np.min()
        ssim_score = ssim(original_images_np,
                          adversarial_images_np,
                          channel_axis=1,
                          data_range=data_range,
                          )

        return ssim_score.mean().item()

    def calculate_psnr(self, original_images: torch.Tensor, adversarial_images: torch.Tensor) -> float:
        """
        Calculates the mean peak signal-to-noise ratio (PSNR) between the original and adversarial images. PSNR is a metric that measures the similarity between two images. 
        The higher the PSNR, the more similar the images are and the lower the distortion between them. A high PSNR could indicate that the perturbations introduced are
        subtle but may not necessarily reflect the perceptual similarity between the images.

        Args:
            original_images (torch.Tensor): The original images. Expected shape is (batch_size, channels, height, width).
            adversarial_images (torch.Tensor): The adversarial images.

        Returns:
            float: The mean PSNR between the original and adversarial images. [0, inf) range. Higher values (e.g. 30 dB or more) indicate better quality. Infinite if the images are identical.
        """
        original_images_np = original_images.cpu().detach().numpy().astype(np.float32)
        adversarial_images_np = adversarial_images.cpu().detach().numpy().astype(np.float32)
        data_range = original_images_np.max() - original_images_np.min()
        psnr_score = psnr(original_images_np,
                          adversarial_images_np, data_range=data_range)
        return psnr_score.mean().item()

    def calculate_similarity_scores(self, original_images: torch.Tensor, adversarial_images: torch.Tensor) -> tuple[float, float]:
        """
        Calculates the SSIM and PSNR between the original and adversarial images. 

        Args:
            original_images (torch.Tensor): The original images. Expected shape is (batch_size, channels, height, width).
            adversarial_images (torch.Tensor): The adversarial images.

        Returns:
            Tuple[float, float]: The mean SSIM and PSNR between the original and adversarial images.
        """
        ssim_score = self.calculate_ssim(original_images, adversarial_images)
        psnr_score = self.calculate_psnr(original_images, adversarial_images)
        return ssim_score, psnr_score

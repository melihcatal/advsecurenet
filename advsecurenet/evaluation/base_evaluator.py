import csv
import os
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional

import torch

from advsecurenet.models.base_model import BaseModel


class BaseEvaluator(ABC):
    """
    Base class for all evaluators.
    """

    def __init__(self, *args, **kwargs):
        """
        Base constructor for all evaluators.
        """

    def __enter__(self):
        self.reset()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """
        Exits the context manager.
        """

    @abstractmethod
    def reset(self):
        """
        Resets the evaluator for a new streaming session.
        """

    @abstractmethod
    def update(self,
               model: BaseModel,
               original_images: torch.Tensor,
               true_labels: torch.Tensor,
               adversarial_images: torch.Tensor,
               is_targeted: Optional[bool] = False,
               target_labels: Optional[torch.Tensor] = None
               ) -> None:
        """
        Updates the evaluator with new data for streaming mode.

        Args:
            model (BaseModel): The model being evaluated.
            original_images (torch.Tensor): The original images.
            original_labels (torch.Tensor): The true labels for the original images.
            adversarial_images (torch.Tensor): The adversarial images.
            is_targeted (bool, optional): Whether the attack is targeted.
            target_labels (Optional[torch.Tensor], optional): Target labels for the adversarial images if the attack is targeted.
        """

    @abstractmethod
    def get_results(self):
        """
        Calculates the results for the streaming session.
        """

    def save_results_to_csv(self,
                            evaluation_results: dict,
                            experiment_info: Optional[dict] = None,
                            path: Optional[str] = None,
                            file_name: Optional[str] = None
                            ) -> None:
        """
        Saves the evaluation results to a CSV file in a structured format.

        Args:
            evaluation_results (dict): The evaluation results.
            experiment_info (dict, optional): The experiment info.
            path (str, optional): The path where the file will be saved.
            file_name (str, optional): The name of the CSV file.

        """

        # Create file name with timestamp if not provided
        if file_name is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            file_name = f"{timestamp}_experiment.csv"

        # Create path if provided and does not exist
        if path:
            os.makedirs(path, exist_ok=True)
            file_path = os.path.join(path, file_name)
        else:
            file_path = file_name

        file_exists = os.path.exists(file_path)

        with open(file_path, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)

            # Write the experiment info and headers if file doesn't exist
            if not file_exists and experiment_info is not None:
                writer.writerow(
                    [f"Experiment conducted on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"])
                for key, value in experiment_info.items():
                    writer.writerow([key, value])

                # Write a separator
                writer.writerow(["-" * 10, "-" * 10])

                # Write headers
                headers = list(evaluation_results.keys())
                writer.writerow(headers)

            # Write the actual results
            values = [str(value)
                      for value in evaluation_results.values()]
            writer.writerow(values)

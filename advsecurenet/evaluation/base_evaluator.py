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

    @abstractmethod
    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        self.reset()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass

    @abstractmethod
    def reset(self):
        """
        Resets the evaluator for a new streaming session.
        """
        pass

    @abstractmethod
    def update(self, *args, **kwargs):
        """
        Updates the evaluator with new data for streaming mode.
        """
        pass

    @abstractmethod
    def get_results(self):
        """
        Calculates the results for the streaming session.
        """
        pass

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

        with open(file_path, mode='a', newline='') as file:
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

    def _calculate_accuracy(self, model: BaseModel, images: torch.Tensor, labels: torch.Tensor) -> float:
        """
        Calculates the accuracy of the model on the given images.
        """
        model.eval()
        predictions = model(images)
        predicted_labels = torch.argmax(predictions, dim=1)
        correct_predictions = torch.sum(predicted_labels == labels)
        accuracy = correct_predictions.item() / len(labels)
        return accuracy

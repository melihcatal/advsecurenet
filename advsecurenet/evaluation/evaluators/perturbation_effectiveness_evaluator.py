from advsecurenet.evaluation.base_evaluator import BaseEvaluator


class PerturbationEffectivenessEvaluator(BaseEvaluator):
    """
    Evaluator for the perturbation effectiveness. The effectiveness score is the attack success rate divided by the perturbation distance. The higher the score, the more effective the attack.
    """

    def __init__(self):
        self.total_attack_success_rate = 0
        self.total_perturbation_distance = 0

    def reset(self):
        """
        Resets the evaluator for a new streaming session.
        """
        self.total_attack_success_rate = 0
        self.total_perturbation_distance = 0

    def update(self, attack_success_rate: float, perturbation_distance: float):
        """
        Updates the evaluator with new data for streaming mode.

        Args:
            attack_success_rate (float): The attack success rate.
            perturbation_distance (float): The perturbation distance.
        """
        self.total_attack_success_rate += attack_success_rate
        self.total_perturbation_distance += perturbation_distance

    def get_results(self) -> float:
        """
        Calculates the mean perturbation effectiveness score for the streaming session.

        Returns:
            float: The mean perturbation effectiveness score for the adversarial examples in the streaming session.
        """
        try:
            return self.total_attack_success_rate / self.total_perturbation_distance
        except ZeroDivisionError:
            return 0

    def calculate_perturbation_effectiveness_score(self, attack_success_rate: float, perturbation_distance: float) -> float:
        """ 
        Calculates the perturbation effectiveness score for the attack. The effectiveness score is the attack success rate divided by the perturbation distance. The higher the score, the more effective the attack. 
        The purpose of this metric is to distinguish between attacks that have a high success rate but require a large perturbation magnitude, 
        and attacks that have a lower success rate but require a smaller perturbation magnitude.
        Args:
            attack_success_rate (float): The attack success rate.
            perturbation_distance (float): The perturbation distance.

        Returns:
            float: The effectiveness score.
        """

        return attack_success_rate / perturbation_distance

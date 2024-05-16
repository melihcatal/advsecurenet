from advsecurenet.evaluation.evaluators import (
    AttackSuccessRateEvaluator, PerturbationDistanceEvaluator,
    PerturbationEffectivenessEvaluator, RobustnessGapEvaluator,
    SimilarityEvaluator, TransferabilityEvaluator)

adversarial_evaluators = {
    "similarity": SimilarityEvaluator(),
    "robustness_gap": RobustnessGapEvaluator(),
    "attack_success_rate": AttackSuccessRateEvaluator(),
    "perturbation_effectiveness": PerturbationEffectivenessEvaluator(),
    "perturbation_distance": PerturbationDistanceEvaluator(),
    "transferability": TransferabilityEvaluator([])
}

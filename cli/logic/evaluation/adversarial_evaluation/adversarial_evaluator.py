import logging
from typing import List

from advsecurenet.models.base_model import BaseModel
from advsecurenet.shared.types.configs.configs import ConfigType
from cli.logic.attack.attack import cli_attack
from cli.shared.types.evaluation import AdversarialEvaluationCliConfigType
from cli.shared.types.utils.model import ModelCliConfigType
from cli.shared.utils.config import load_and_instantiate_config
from cli.shared.utils.model import create_model

logger = logging.getLogger(__name__)


class CLIAdversarialEvaluator():
    """
    Base class for the adversarial evaluation CLI.
    """

    def __init__(self, config: AdversarialEvaluationCliConfigType, **kwargs):
        self.config = config
        self.kwargs = kwargs

    def run(self):
        """
        Run the evaluation.


        try:
            logger.info("Starting adversarial evaluation with the following evaluators: %s",
                         self.config.evaluation.evaluators)
            model, data_loader, attack, target_models = self._prepare_evaluation_env()
            self._execute_evaluation(
                model, data_loader, attack, target_models)
            logger.info("Adversarial evaluation completed successfully.")
        except Exception as e:
            click.secho("Evaluation failed.", fg="red")
            logger.error("Failed to evaluate adversarial examples: %s", e)
            raise e
        """
        attack_name = self.config.evaluation_config.attack.name.upper()
        attack_config = self.config.evaluation_config.attack.config
        target_models = self._prepare_target_models()
        cli_attack(attack_name,
                   attack_config,
                   target_models=target_models,
                   evaluators=self.config.evaluation_config.evaluators)

    def _prepare_target_models(self) -> List[BaseModel]:
        """ 
        Prepare the models that will be used as the target models for the transferability evaluation.

        Returns:
            List[BaseModel]: List of initialized target models based on the configuration provided.

        """

        if not self.config.evaluation_config.target_models or not any(self.config.evaluation_config.target_models):
            return []

        # Load and initialize each model based on its configuration
        models = [
            create_model(
                load_and_instantiate_config(
                    config=model_config.get('config'),
                    default_config_file="model_config.yml",
                    config_type=ConfigType.MODEL,
                    config_class=ModelCliConfigType
                )
            )
            for model_config in self.config.evaluation_config.target_models
        ]

        return models

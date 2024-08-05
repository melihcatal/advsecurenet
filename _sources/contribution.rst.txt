How to Contribute?
==================

Thank you for considering contributing to ``AdvSecureNet``. We welcome
all contributions, including new features, bug fixes, and documentation
improvements. To make the process seamless, we have prepared a guide for
different types of contributions.

Reporting Bugs
--------------

If you find a bug in the project, please open an issue on the GitHub
repository. When reporting a bug, include the following details:

-  A detailed description of the issue.
-  Steps to reproduce the issue.
-  The version of the project you are using.
-  The operating system you are running.
-  The Python version.
-  If applicable:

   -  The CUDA version.

-  Any other relevant information.

An example of a good bug report would be:

::

   Description: [Detailed description of the issue]

   Steps to Reproduce:

   1.  [Step one]
   2.  [Step two]
   3.  [Step three]

   Version: [Project version]
   OS: [Operating system]
   Python: [Python version ]
   CUDA: [CUDA version if applicable]
   Other: [Any other relevant information]

Contributing Code
-----------------

``AdvSecureNet`` has two main components: the API and the CLI. You can
contribute to either component. The high-level process is the same:

1. Fork the repository.
2. Create a new branch for your changes.
3. Make your changes.
4. Write tests for your changes.
5. Run the tests.
6. Document your changes.
7. Submit a pull request.
8. Stay tuned for feedback.

Code Quality Standards
~~~~~~~~~~~~~~~~~~~~~~

To ensure code quality, follow these standards:

-  Format code using ``black``.
-  Lint code using ``pylint``.
-  Test code using ``pytest``.
-  Document code using ``sphinx``.
-  Type-check code using ``mypy``.
-  Have your code reviewed by at least one other contributor before
   merging.

``AdvSecureNet`` uses ``Gitflow`` as its branching model:

-  Develop new features in a ``feature`` branch.
-  Develop bug fixes in a ``hotfix`` branch.
-  The ``main`` branch is reserved for stable releases.
-  The ``develop`` branch is used for development.

When submitting a pull request, target the ``develop`` branch. For more
information on ``Gitflow``, refer to `this
guide <https://www.atlassian.com/git/tutorials/comparing-workflows/gitflow-workflow>`__.

Creating a New Feature
~~~~~~~~~~~~~~~~~~~~~~

You have three options for creating a new feature:

-  A new attack
-  A new defense
-  A new evaluation metric / evaluator

Depending on the feature type and the target component (API or CLI), the
process will differ. Refer to the following sections for more
information.

Creating a New Attack
^^^^^^^^^^^^^^^^^^^^^

API
'''

The ``advsecurenet`` package contains an ``attacks`` module with various
submodules based on attack types (e.g., gradient-based, decision-based).
If your attack does not fit into any existing submodule, feel free to
create a new one. Currently, ``AdvSecureNet`` supports evasion attacks
on computer vision models only. All attacks should inherit from the
``AdversarialAttack`` class, an abstract base class that defines the
interface for all attacks. The ``AdversarialAttack`` class is defined in
the ``attacks.base`` module. Additionally, each attack should have its
own configuration class, which defines the parameters of the attack.
This approach keeps the attack class clean and makes it easier to use
the attack in the CLI. The configuration class should be defined in the
``shared.types.configs.attack_configs`` folder and should inherit from
the ``AttackConfig`` class. The ``AttackConfig`` class, also defined in
the ``shared.types.configs.attack_configs`` folder, contains the
``device`` attribute, specifying the device on which the attack should
run, and a flag indicating whether the attack is targeted or untargeted.

Follow these steps to create a new attack:

1. Create a new submodule in the ``attacks`` module.
2. Create a class for your attack that inherits from the
   ``AdversarialAttack`` class.
3. Create a configuration class for your attack that inherits from the
   ``AttackConfig`` class.
4. Implement the ``__init__`` method, accepting the configuration as an
   argument.
5. Implement the ``attack`` method, taking the model, input, and target
   as arguments and returning the adversarial example.

**Example**:

.. code:: python

   from advsecurenet.attacks.base import AdversarialAttack
   from advsecurenet.shared.types.configs.attack_configs import AttackConfig
   from advsecurenet.models.base_model import BaseModel
   from dataclasses import dataclass, field

   @dataclass(kw_only=True)
   class RandomNoiseAttackConfig(AttackConfig):
       epsilon: float = field(default=0.1)

   class RandomNoiseAttack(AdversarialAttack):
       def __init__(self, config: RandomNoiseAttackConfig):
           self.epsilon = config.epsilon

       def attack(self, model: BaseModel, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
           noise = torch.randn_like(input) * self.epsilon
           return input + noise

Using a dataclass for the configuration class makes it easy to create
instances of the class with default values. It also prevents users from
passing invalid arguments to the attack. Using ``kw_only=True`` ensures
that users have to pass the arguments by keyword, which makes the code
more readable and less error-prone. Additionally, it facilitates the
future extension of the configuration class with new parameters without
breaking the existing code.

CLI
'''

To use your attack in the CLI, follow these additional steps:

1. Create Default YAML Configuration Files

1.1. For Adversarial Training Command:

::

   •   Create a configuration file for the attack parameters in the `cli/configs/attacks/base` folder.
   •   Create another configuration file for the attack itself (including other necessary parameters) in the cli/configs/attacks folder.

This separation makes it easier to use the attack both in the
adversarial training command and as a standalone attack.

Example of a configuration file for the attack parameters:

.. code:: yaml

   # Description: Base configuration file for the attack parameters. Located in the cli/configs/attacks/base folder
   target_parameters: !include ../shared/attack_target_config.yml # Targeted attack configs
   attack_parameters:
   epsilon: 0.3 # The epsilon value to be used for the FGSM attack. The higher the value, the more the perturbation

Example of a configuration file for the attack:

.. code:: yaml

   # Description: Configuration file for the attack. Located in the cli/configs/attacks folder

   model: !include ../shared/model_config.yml
   dataset: !include ./shared/attack_dataset_config.yml
   dataloader: !include ../shared/dataloader_config.yml
   device: !include ../shared/device_config.yml
   attack_procedure: !include ./shared/attack_procedure_config.yml

   # Attack Specific Configuration
   attack_config: !include ./base/attack_base_config.yml

The first file contains the parameters specific to the attack, while the
second file contains the parameters common to all attacks. The first
file is used when another command wants to use the attack as a
parameter, which means that command takes care of the essential
parameters like the model, dataset, etc. The second file is used when
the attack is used as a standalone command, which means that the attack
command needs to configure the necessary parameters to prepare the
environment for the attack and then run the attack.

2. Create a configuration dataclass for your attack in the
   ``cli/shared/types/attack/attacks`` folder.
3. Update the attack mapping in the
   ``cli/shared/utils/attack_mappings.py`` file to include your attack.
4. Finally, update the ``cli/commands/attack/commands.py`` module to
   include your attack as a subcommand.

Creating a New Defense
^^^^^^^^^^^^^^^^^^^^^^

.. _api-1:

API
'''

There is no base class for defenses in ``AdvSecureNet``. However, each
defense should have its own configuration class similar to attacks.
Follow these steps:

1. Create a new submodule in the ``defenses`` module.
2. Create a configuration class for your defense.
3. Create a class for your defense and use the configuration class you
   created to initialize the defense in the ``__init__`` method.

.. _cli-1:

CLI
'''

Since there isn't a ``defender`` module to run the defenses, unlike the
attacks, you need to create the logic for the defense in the
``cli/logic/defense`` folder. To add your defense to the CLI:

1. Create a default YAML configuration file for your defense in the
   ``cli/configs/defenses`` folder.
2. Create a configuration dataclass for your defense in the
   ``cli/shared/types/defense/defenses`` folder.
3. Implement the defense logic in the ``cli/logic/defense`` folder.
4. Add the defense to the ``cli/commands/defense/commands.py`` module.

Creating a New Evaluation Metric
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. _api-2:

API
'''

Evaluation metrics are used to assess the performance of attacks,
defenses or models. The ``advsecurenet`` package includes an
``evaluation`` module that contains all the evaluation metrics. Each
evaluation metric should inherit from the ``BaseEvaluator`` class, an
abstract base class that defines the interface for all evaluation
metrics. The evaluators are context managers, allowing them to be used
in a ``with`` statement to automatically clean up any resources they
use. The ``BaseEvaluator`` class is defined in the
``evaluation.base_evaluator`` module.

1. Create a new class for your evaluation metric that inherits from the
   ``BaseEvaluator`` class in the ``evaluation.evaluators`` folder.
2. Implement the ``update`` method of your evaluation metric class. This
   method defines how the evaluation metric should be updated when a new
   sample is evaluated.
3. Implement the ``get_results`` method of your evaluation metric class.
   This method should return the final result of the evaluation metric.
4. If the evaluator is an adversarial evaluator, update the
   ``advsecurenet.shared.adversarial_evaluators`` module to include your
   evaluator.

.. _cli-2:

CLI
'''

Evaluation metrics are automatically available in the CLI once the API
is updated. No additional steps are needed. This is because the CLI uses
the attacker to run evaluation metrics, and they are not run
independently.

Documentation
--------------

Improving documentation is always appreciated. If you find any part of
the codebase that is not well-documented or could be improved, please
open a pull request with your changes. We value any help in making the
documentation more comprehensive and easier to understand.

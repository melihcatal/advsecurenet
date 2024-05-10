# Title of the module
MODULE_TITLE = "Shared Configurations"

# Description of the module
MODULE_DESCRIPTION = """
    This module contains the shared submodules that are used in the other modules. The shared submodules include the
    configurations for the dataloader, dataset, device, model, and training. 
    These configurations are used in the other modules to create the configurations for the CLI. 
    
    Some of these configurations can be used as standalone configurations, such as the model configuration, in the adversarial training
    as part of the ensemble of models.
"""

# Flag to include this module in CLI configurations
INCLUDE_IN_CLI_CONFIGS = True

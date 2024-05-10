
import click

from advsecurenet.shared.types.attacks import AttackType
from advsecurenet.shared.types.configs import ConfigType
from cli.attacks.lots import CLILOTSAttack
from cli.shared.attack_mappings import attack_mapping
from cli.types.attacks.attack_base import BaseAttackCLIConfigType
from cli.utils.attack import execute_attack
from cli.utils.config import load_and_instantiate_config
from cli.utils.data import load_and_prepare_data
from cli.utils.dataloader import get_dataloader
from cli.utils.helpers import save_images
from cli.utils.model import create_model


def cli_attack(attack_name: str, config: str, **kwargs) -> None:
    """
    Entry point for the attack CLI. This function is called when the user wants to execute an attack. It determines the
    attack type and maps it to the respective configuration class. It then loads the configuration and executes the
    attack.

    Args:
        attack_name (str): The name of the attack to execute.
        config (str): The path to the configuration file.
        **kwargs: Additional keyword arguments to pass to the configuration class.

    Raises:
        ValueError: If the attack type is unknown.
    """
    if attack_name not in attack_mapping:
        raise ValueError(f"Unknown attack type: {attack_name}")

    attack_type, attack_config_class = attack_mapping[attack_name]
    config_data = load_and_instantiate_config(
        config=config,
        default_config_file=f"{attack_name.lower()}_attack_config.yml",
        config_type=ConfigType.ATTACK,
        config_class=attack_config_class,
        **kwargs
    )
    cli_execute_general_attack(config_data, attack_type)


def cli_execute_general_attack(config_data: BaseAttackCLIConfigType, attack_type: AttackType) -> None:
    """ 
    The logic function to execute an attack based on the attack type.

    Args:
        config_data (BaseAttackCLIConfigType): The configuration data for the attack.
        attack_type (AttackType): The type of the attack to execute.
    """

    model = create_model(config_data.model)

    dataset = load_and_prepare_data(config_data.dataset)
    data_loader = get_dataloader(
        config=config_data.dataloader,
        dataset=dataset,
        dataset_type='default',
        use_ddp=config_data.device.use_ddp)

    attack_config = config_data.attack_config

    # Set the device for the attack. This is a workaround for now until we refactor the device handling
    attack_config.device = config_data.device

    # if the attack is LOTS, we need to use the custom lots wrapper
    if attack_type == AttackType.LOTS:
        lots = CLILOTSAttack(config=attack_config,
                             dataset=dataset,
                             data_loader=data_loader,
                             model=model)
        adv_images = lots.execute_attack()
    else:
        attack_class = attack_type.value
        attack = attack_class(attack_config)

        adv_images = execute_attack(model=model,
                                    data_loader=data_loader,
                                    attack=attack,
                                    device=config_data.device.processor,
                                    )
    if config_data.attack_procedure.save_result_images:
        click.secho("Saving adversarial images...", fg="green")
        save_images(adv_images[0],
                    config_data.attack_procedure.result_images_dir,
                    prefix=config_data.attack_procedure.result_images_prefix)

    click.secho("Attack completed!", fg="green")

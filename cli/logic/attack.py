import click

from advsecurenet.dataloader import DataLoaderFactory
from advsecurenet.shared.types.attacks import AttackType
from advsecurenet.shared.types.configs import ConfigType, attack_configs
from advsecurenet.utils.config_utils import get_default_config_yml
from cli.attacks.lots import CLILOTSAttack
from cli.utils.attack import execute_attack, save_adversarial_images
from cli.utils.config import build_config, load_configuration
from cli.utils.data import load_and_prepare_data
from cli.utils.model import prepare_model

# Mapping of attack types to their respective configurations
attack_mapping = {
    "CW": (AttackType.CW, attack_configs.CWAttackConfig),
    "DEEPFOOL": (AttackType.DEEPFOOL, attack_configs.DeepFoolAttackConfig),
    "PGD": (AttackType.PGD, attack_configs.PgdAttackConfig),
    "FGSM": (AttackType.FGSM, attack_configs.FgsmAttackConfig),
    "TARGETED_FGSM": (AttackType.TARGETED_FGSM, attack_configs.FgsmAttackConfig),
    "LOTS": (AttackType.LOTS, attack_configs.LotsAttackConfig),
    "DECISION_BOUNDARY": (AttackType.DECISION_BOUNDARY, attack_configs.DecisionBoundaryAttackConfig),
}


def cli_attack(attack_name: str, config: str, **kwargs):
    """Execute a specified attack."""
    if attack_name not in attack_mapping:
        raise ValueError(f"Unknown attack type: {attack_name}")

    attack_type, attack_config_class = attack_mapping[attack_name]
    cli_execute_general_attack(
        attack_type, config, attack_config_class, **kwargs)


def cli_execute_general_attack(attack_type, config_file: str, attack_config_class, **kwargs) -> None:
    """
    The logic function to execute an attack based on the attack type.
    """
    attack_name = attack_type.name.lower()
    if not config_file:
        click.secho(
            f"No config file provided! Using default {attack_name} attack config.", fg="yellow")

        file_name = f'{attack_name}_attack_config.yml'
        config_file = get_default_config_yml(file_name, "cli")

    config_data = load_configuration(
        config_type=ConfigType.ATTACK, config_file=config_file, **kwargs)

    data, num_classes, device = load_and_prepare_data(config_data)
    model = prepare_model(config_data, num_classes, device)
    # if the attack is LOTS, we need to use a different logic as LOTS expects target layers
    if attack_type == AttackType.LOTS:
        click.echo(f"Executing {attack_name} attack...")
        attack = CLILOTSAttack(
            config_data=config_data, model=model, device=device, dataset=data
        )
        adversarial_images = attack.execute_attack()

    else:
        if attack_type == AttackType.CW and config_data['targeted']:
            click.secho(
                "CW attack is not supported in targeted mode!", fg="yellow")
            return

        attack_config = build_config(config_data, attack_config_class)

        attack_class = attack_type.value  # Get the class from the Enum
        attack = attack_class(attack_config)
        click.echo(f"Executing {attack_name} attack...")

        data_loader = DataLoaderFactory.create_dataloader(
            data, batch_size=config_data['batch_size'])

        adversarial_images = execute_attack(
            model=model, data_loader=data_loader, attack=attack, device=device, verbose=config_data['verbose'])

    if config_data['save_result_images']:
        save_adversarial_images(
            data,
            adversarial_images,
            attack_name,
            config_data['save_limit'],
            config_data['result_images_dir']
        )

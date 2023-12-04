import click

from advsecurenet.shared.types.attacks import AttackType
from advsecurenet.shared.types.configs import ConfigType, attack_configs
from advsecurenet.utils.config_utils import get_default_config_yml
from cli.attacks.lots import CLILOTSAttack
from cli.utils.attack import execute_attack
from cli.utils.config import build_config, load_configuration
from cli.utils.data import load_and_prepare_data
from cli.utils.model import prepare_model

# Mapping of attack types to their respective configurations
attack_mapping = {
    "CW": (AttackType.CW, attack_configs.CWAttackConfig),
    "DEEPFOOL": (AttackType.DEEPFOOL, attack_configs.DeepFoolAttackConfig),
    "PGD": (AttackType.PGD, attack_configs.PgdAttackConfig),
    "FGSM": (AttackType.FGSM, attack_configs.FgsmAttackConfig),
    "LOTS": (AttackType.LOTS, attack_configs.LotsAttackConfig),

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
        click.echo(
            f"No configuration file provided for {attack_name} attack! Using default configuration...")
        file_name = f'{attack_name}_attack_config.yml'
        config_file = get_default_config_yml(file_name, "cli")

    config_data = load_configuration(
        config_type=ConfigType.ATTACK, config_file=config_file, **kwargs)
    if attack_type == AttackType.LOTS:
        click.echo(f"Executing {attack_name} attack...")
        attack = CLILOTSAttack(config_data)
        adversarial_images = attack.execute_attack()
    else:
        if attack_type == AttackType.CW and config_data['targeted']:
            click.UsageError(
                "Targeted CW attack through CLI not supported yet! Please set targeted to False or use API.")
        data, num_classes, device = load_and_prepare_data(config_data)
        attack_config = build_config(config_data, attack_config_class)
        print(f"config data is {config_data}")
        model = prepare_model(config_data, num_classes, device)

        attack_class = attack_type.value  # Get the class from the Enum
        print(f"config {attack_config}")
        attack = attack_class(attack_config)
        click.echo(f"Executing {attack_name} attack...")
        adversarial_images = execute_attack(
            model=model, data=data, batch_size=config_data['batch_size'], attack=attack, device=device, verbose=config_data['verbose'])

    click.echo(f"Attack completed! Saving adversarial images...")
    # save adversarial images
    # TODO: save adversarial images for given number of images

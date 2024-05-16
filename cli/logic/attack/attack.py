
import click
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from advsecurenet.attacks import AdversarialAttack
from advsecurenet.models.base_model import BaseModel
from advsecurenet.shared.types.attacks import AttackType
from advsecurenet.shared.types.configs import ConfigType
from cli.logic.attack.attacks.lots import CLILOTSAttack
from cli.shared.types.attack import BaseAttackCLIConfigType
from cli.shared.utils.attack_mappings import attack_cli_mapping
from cli.shared.utils.config import load_and_instantiate_config
from cli.shared.utils.data import load_and_prepare_data
from cli.shared.utils.dataloader import get_dataloader
from cli.shared.utils.helpers import save_images
from cli.shared.utils.model import create_model


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
    if attack_name not in attack_cli_mapping:
        raise ValueError(f"Unknown attack type: {attack_name}")

    attack_type, attack_config_class = attack_cli_mapping[attack_name]
    config_data = load_and_instantiate_config(
        config=config,
        default_config_file=f"{attack_name.lower()}_attack_config.yml",
        config_type=ConfigType.ATTACK,
        config_class=attack_config_class,
        **kwargs
    )
    cli_execute_general_attack(config_data, attack_type)


def execute_attack(model: BaseModel,
                   data_loader: DataLoader,
                   attack: AdversarialAttack,
                   device: torch.device = torch.device("cpu"),
                   verbose: bool = False
                   ) -> list[torch.Tensor]:
    """
    Execute the specified attack on the model using the data loader.

    Args:
        model (BaseModel): The model to attack.
        data_loader (DataLoader): The data loader to use for generating adversarial samples.
        attack (AdversarialAttack): The attack to execute.
        device (torch.device): The device to use for the attack.
        verbose (bool): Whether to print verbose logs.

    Returns:
        list[torch.Tensor]: A list of adversarial images.

    """
    try:
        model = model.to(device)
        model.eval()
        adversarial_images = []

        successful_attacks = 0  # Track number of successful attacks
        total_samples = 0       # Track total number of samples processed

        for images, labels in tqdm(data_loader, desc="Generating adversarial samples"):
            images = images.to(device)
            labels = labels.to(device)

            # Get predictions for the original images
            original_preds = torch.argmax(model(images), dim=1)

            # Generate adversarial images
            adv_images = attack.attack(model, images, labels)
            adversarial_images.append(adv_images)

            # Get predictions for the adversarial images
            adversarial_preds = torch.argmax(model(adv_images), dim=1)

            # Check how many attacks were successful
            successful_attacks += (adversarial_preds !=
                                   original_preds).sum().item()
            total_samples += images.size(0)
            if verbose:
                click.echo(
                    f"Attack success rate: {successful_attacks / total_samples * 100:.2f}%")

        success_rate = (successful_attacks / total_samples) * 100
        click.secho(
            f"Succesfully generated adversarial samples! Attack success rate: {success_rate:.2f}%", fg="green")

        return adversarial_images
    except Exception as e:
        raise click.ClickException(
            f"Error executing attack! Details: {e}")


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

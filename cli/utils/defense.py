
import os
import click
import pkg_resources
from advsecurenet.shared.types.configs.attack_configs.attack_config import AttackConfig
from advsecurenet.shared.types.configs.configs import ConfigType
from advsecurenet.shared.types.dataset import DatasetType
from advsecurenet.models.model_factory import ModelFactory
from advsecurenet.datasets.dataset_factory import DatasetFactory
from advsecurenet.dataloader import DataLoaderFactory
from advsecurenet.shared.types.configs.defense_configs.adversarial_training_config import AdversarialTrainingConfig
from advsecurenet.shared.types.configs import TestConfig
from advsecurenet.utils.model_utils import train as util_train, test as util_test, save_model, load_model
from advsecurenet.shared.types.attacks import AttackType as AdversarialAttackType
from advsecurenet.defenses.adversarial_training import AdversarialTraining
from cli.utils.config import build_config, load_configuration


def _get_attack_configs(config_files: list, **overrides) -> list[AttackConfig]:
    """
    Get the attack configurations from the provided configuration files.
    """
    attack_configs = []
    for config_file in config_files:
        config_data = load_configuration(
            ConfigType.ATTACK, config_file, **overrides)
        attack_configs.append(build_config(config_data, AttackConfig))
    return attack_configs


def cli_adversarial_training(config_data):
    print(config_data)
    # # match the dataset name to the dataset type
    # dataset_name = config_data['dataset_type'].upper()
    # if dataset_name not in DatasetType._value2member_map_:
    #     raise ValueError("Unsupported dataset name! Choose from: " +
    #                      ", ".join([e.value for e in DatasetType]))

    # target_model = ModelFactory.get_model(
    #     config_data['model'], num_classes=config_data['num_classes'])

    # models = []
    # attacks = []
    # for model_name in config_data['models']:
    #     models.append(ModelFactory.get_model(
    #         model_name, num_classes=config_data['num_classes']))

    # for attack_name in config_data['attacks']:
    #     attacks.append(AdversarialAttackType[attack_name.upper()])
    # attacks = [attack.value for attack in attacks]

    # attack_configs = _get_attack_configs(config_data["attack_configs"])

    # # instantiate the attacks with the attack configs
    # attack_objs = []
    # for attack, attack_config in zip(attacks, attack_configs):
    #     attack_objs.append(attack(attack_config))

    # # create objects from the attack classes
    # print(attack_objs)

    # dataset_type = DatasetType(dataset_name.upper())
    # dataset_obj = DatasetFactory.load_dataset(dataset_type)
    # train_data = dataset_obj.load_dataset(train=True)
    # test_data = dataset_obj.load_dataset(train=False)

    # train_data_loader = DataLoaderFactory.get_dataloader(
    #     train_data, batch_size=config_data['batch_size'], shuffle=True)
    # test_data_loader = DataLoaderFactory.get_dataloader(
    #     test_data, batch_size=config_data['batch_size'], shuffle=False)

    # config = AdversarialTrainingConfig(
    #     model=target_model,
    #     models=models,
    #     attacks=attacks,
    #     train_loader=train_data_loader,
    #     epochs=config_data['epochs'],
    #     learning_rate=config_data['learning_rate'],
    #     save_checkpoint=config_data['save_checkpoint'],
    #     save_checkpoint_path=config_data['save_checkpoint_path'],
    #     save_checkpoint_name=config_data['save_checkpoint_name'],
    #     load_checkpoint=config_data['load_checkpoint'],
    #     load_checkpoint_path=config_data['load_checkpoint_path'],
    #     verbose=config_data['verbose'],
    #     adv_coeff=config_data['adv_coeff'],
    #     optimizer=config_data['optimizer'],
    #     criterion=config_data['criterion'])

    # adversarial_training = AdversarialTraining(config)
    # adversarial_training.adversarial_training()

    # click.echo(
    #     f"Model trained on {dataset_name} with attacks {config_data['attacks']}!")

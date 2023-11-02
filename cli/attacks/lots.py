import random
import click
import torch
from tqdm import tqdm
from advsecurenet.attacks.lots import LOTS
from advsecurenet.dataloader.data_loader_factory import DataLoaderFactory
from advsecurenet.datasets.dataset_factory import DatasetFactory
from advsecurenet.shared.types.configs import attack_configs
from advsecurenet.shared.types.device import DeviceType
from cli.utils.config import build_config

from cli.utils.data import get_custom_data, load_and_prepare_data
from cli.utils.model import prepare_model

def validate_config(config_data):
    # check if deep feature layer is provided
    if not config_data['deep_feature_layer']:
        raise ValueError("Please provide deep feature layer for the attack!")
    
    # check if auto generate target images is false and target images are not provided
    if not config_data['auto_generate_target_images'] and not config_data['target_images_dir']:
        raise ValueError("Please provide target images for the attack or set auto_generate_target_images to True!")
    
def _generate_target_images(config_data, all_data, labels):

    if config_data['target_images_dir']:
        try:
            target_images, target_labels = get_custom_data(config_data['target_images_dir'])
        except Exception as e:
            raise ValueError("Error loading target images! Details: {e}")
        
    elif config_data['auto_generate_target_images']:
        target_images, target_labels = generate_target_images(all_data, labels, config_data['maximum_generation_attempts']) 

    return target_images, target_labels


def execute_lots_attack(config_data):
    validate_config(config_data)
     
    # if target images are provided, load them 
    target_images = None
    target_labels = None

    # get data
    data, num_classes, device = load_and_prepare_data(config_data)
    labels = data.tensors[1]

    # generate dataset 
    dataset_obj = DatasetFactory.load_dataset(config_data['dataset_type'])
    train_data = dataset_obj.load_dataset(train=True)
    test_data = dataset_obj.load_dataset(train=False)
    all_data = train_data + test_data

    # generate target images
    target_images, target_labels = _generate_target_images(config_data, all_data, labels)

    # Adjust the mode for the LOTS attack
    mode_string = config_data.get("mode")
    config_data["mode"] = attack_configs.LotsAttackMode[mode_string.upper()]
    
    attack_config = build_config(config_data, attack_configs.LotsAttackConfig)

    if isinstance(device, DeviceType):
        device = device.value
    
    model = prepare_model(config_data, num_classes, device)
    attack = LOTS(attack_config)

    data_loader = DataLoaderFactory.get_dataloader(data, batch_size=config_data['batch_size'], shuffle=False)

    adversarial_images = []
    successful_attacks = 0
    total_samples = 0
    verbose = config_data['verbose']

    for images, labels in tqdm(data_loader, desc="Generating adversarial samples"):
        # get current portion of target images
        target_images_batch = target_images[total_samples:total_samples + images.size(0)]
        target_labels_batch = target_labels[total_samples:total_samples + images.size(0)]

        # move to device
        images = images.to(device)
        labels = labels.to(device)
        target_images_batch = target_images_batch.to(device)
        target_labels_batch = target_labels_batch.to(device)

        adversarial_images_batch, is_found = attack.attack(model=model, data=images, target=target_images_batch, target_classes=target_labels_batch)
        adversarial_preds = torch.argmax(model(adversarial_images_batch), dim=1)

        # attack is successful if the prediction is target label
        successful_attacks += (adversarial_preds == target_labels_batch).sum().item()

        adversarial_images.append(adversarial_images_batch)
        total_samples += images.size(0)
        if verbose:
            click.echo(f"Attack success rate: {successful_attacks / total_samples * 100:.2f}%")

    success_rate = (successful_attacks / total_samples) * 100
    print (f"Succesfully generated adversarial samples! Attack success rate: {success_rate:.2f}%")

    return adversarial_images



def generate_target_images(all_images, clean_labels, max_attempts=100):
    target_images = []
    target_labels = []

    for label in clean_labels:
        attempts = 0
        found_mismatch = False
        while attempts < max_attempts:
            random_idx = random.randint(0, len(all_images) - 1)
            if all_images[random_idx][1] != label:
                target_images.append(all_images[random_idx][0])
                target_labels.append(all_images[random_idx][1])
                found_mismatch = True
                break
            attempts += 1

        if not found_mismatch:
            raise ValueError(f"Failed to find a mismatch for label {label} after {max_attempts} attempts.")

    # Convert lists to tensors
    target_images = torch.stack(target_images)
    target_labels = torch.tensor(target_labels)

    return target_images, target_labels

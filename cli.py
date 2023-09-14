# ignore warnings
import warnings
warnings.simplefilter(action='ignore', category=UserWarning)

from logger_config import setup_logging
setup_logging()

import click
from advsecurenet.attacks import lots
from advsecurenet.models.vgg16 import VGG16Features
import torch
from PIL import Image
import os
import datetime
import torchvision.transforms as transforms


@click.group()
def main():
    click.echo("Welcome to the Adversarial Secure Network CLI!")

def determine_output_directory(command_name, output):
    # Determine the output directory
    if output is None:
        output = os.path.join("outputs", command_name, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    else:
        output = os.path.join(output, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    if not os.path.exists(output):
        os.makedirs(output)

    return output

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@main.command()
@click.option('--targeted', is_flag=True, help='Run a targeted attack. Defaults to False (untargeted).', default=False) 
@click.option('--input-image-path', default=None, help='Path to the input image.')
@click.option('--input-descriptor-path', default=None, help='Path to the serialized input descriptor.')
@click.option('--target-image-path', default=None, help='Path to the target image.')
@click.option('--target-descriptor-path', default=None, help='Path to the serialized target descriptor.')
@click.option('--output', default=None, help='Custom output directory. Defaults to a timestamped folder in outputs/<command_name>.')
def run_lots(targeted, input_image_path,input_descriptor_path, target_image_path, target_descriptor_path, output):
    """Run LOTS to create an adversarial image."""
    # Initialize the model
    model = VGG16Features().eval()
    device = get_device()
    model.to(device)

    # Determine the output directory
    output = determine_output_directory("run_lots", output)

    if input_image_path:
        _, input_descriptor = model(model.preprocess(Image.open(input_image_path)).unsqueeze(0).to(device))
    elif input_descriptor_path:
        input_descriptor = torch.load(input_descriptor_path).to(device)
    else:
        print("You must provide either an input image path or an input descriptor path!")
        return

    input_tensor = model.preprocess(Image.open(input_image_path)).unsqueeze(0).to(device)

    if targeted and target_image_path:
        target_tensor = model.preprocess(Image.open(target_image_path)).unsqueeze(0).to(device)
    elif targeted and target_descriptor_path:
        target_tensor = torch.load(target_descriptor_path).to(device)
    elif targeted and (not target_image_path or not target_descriptor_path):
        print("You must provide either a target image path or a target descriptor path!")
        return
    elif not targeted and target_image_path:
        print("You cannot provide a target image path for an untargeted attack!")
        return
    elif not targeted and target_descriptor_path:
        print("You cannot provide a target descriptor path for an untargeted attack!")
        return
    else:
        target_descriptor = torch.zeros(4096).to(device)  # Dummy target descriptor

    adversarial, _ = lots.lots_iterative(model, input_tensor.squeeze(0), target_descriptor)
    
    print(f"Adversarial class for {input_image_path}: {model.predict_class(adversarial.unsqueeze(0))}")
    # save the original class and the adversarial class to a file in format "original_class,adversarial_class"
    with open(os.path.join(output, "original_class,adversarial_class"), "w") as f:
        f.write(f"{model.predict_class(input_tensor)},{model.predict_class(adversarial.unsqueeze(0))}")
    
    # save the adversarial image
    adversarial_image = transforms.ToPILImage()(adversarial.cpu().detach().squeeze(0))
    adversarial_image.save(os.path.join(output, "adversarial.png"))
    
    
if __name__ == "__main__":
    main()

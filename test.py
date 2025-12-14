import torch
import torchvision
import torch.optim
import os
import time
from model.DIM import DIM
from PIL import Image
import argparse
import glob
from torchvision.transforms import Compose, ToTensor
from torch.cuda.amp import autocast


def lowlight(image_path, net, result_path, device):
    """
    Process a single low-light image using the DIM model.

    Args:
        image_path (str): Path to input image.
        net (nn.Module): Pre-loaded DIM model.
        result_path (str): Path to output image.
        device (torch.device): Device to run the model on.
    """
    # Define image preprocessing pipeline
    transform = Compose([
        ToTensor()  # Convert PIL image to C×H×W Tensor and normalize to [0, 1]
    ])

    # Load and preprocess image
    data_lowlight = Image.open(image_path)
    data_lowlight = transform(data_lowlight)  # Apply transform
    data_lowlight = data_lowlight.unsqueeze(0).to(device)  # Add batch dimension

    # Perform inference
    with torch.no_grad():
        enhanced_images = net(data_lowlight)

    # If model returns a list of outputs, take the last (most enhanced) one
    if isinstance(enhanced_images, (list, tuple)):
        enhanced_img = enhanced_images[-1].squeeze(0)
    else:
        enhanced_img = enhanced_images.squeeze(0)

    # Save enhanced image
    # Make sure output directory exists
    if not os.path.exists(os.path.dirname(result_path)):
        os.makedirs(os.path.dirname(result_path))
    torchvision.utils.save_image(enhanced_img, result_path)

    # Clean up
    torch.cuda.empty_cache()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test ReF-DIM model on a folder of images')
    parser.add_argument('--input_folder', '-i', type=str, required=True, help='Path to input images folder')
    parser.add_argument('--model_path', '-m', type=str, required=True, help='Path to trained model')
    parser.add_argument('--output_folder', '-o', type=str, required=True, help='Path to output folder')

    args = parser.parse_args()

    # Process images
    # Ensure output folder exists
    os.makedirs(args.output_folder, exist_ok=True)

    # Initialize device and model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = DIM().to(device)
    if torch.cuda.is_available():
        net.load_state_dict(torch.load(args.model_path))
    else:
        net.load_state_dict(torch.load(args.model_path, map_location='cpu'))

    # Get all image files
    image_files = [
        os.path.join(args.input_folder, f) for f in os.listdir(args.input_folder)
    ]

    # process images one by one
    for image_path in image_files:
        result_path = os.path.join(args.output_folder, f"{os.path.splitext(os.path.basename(image_path))[0]}.png")
        start_time = time.time()
        lowlight([image_path], net, [result_path], device)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Saved enhanced image to {result_path}. Processing time: {elapsed_time:.4f} seconds")
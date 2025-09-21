import torch
import torchvision
import torch.optim
import os
from model.DIM import DIM
from PIL import Image
import argparse
import glob
from torchvision.transforms import Compose, ToTensor
from torch.cuda.amp import autocast


def lowlight(image_paths, net, result_paths, device):
    """
    Process a batch of low-light images using the DIM model.

    Args:
        image_paths (list): List of input image paths.
        net (nn.Module): Pre-loaded DIM model.
        result_paths (list): List of output image paths.
        device (torch.device): Device to run the model on.
    """
    # Define image preprocessing pipeline
    transform = Compose([
        ToTensor()  # Convert PIL image to C×H×W Tensor and normalize to [0, 1]
    ])

    # Load and preprocess images
    batch_images = []
    for image_path in image_paths:
        data_lowlight = Image.open(image_path)
        data_lowlight = transform(data_lowlight)  # Apply transform
        batch_images.append(data_lowlight)

    # Stack images into a batch
    batch_images = torch.stack(batch_images).to(device)

    # Print memory usage before inference
    print(f"Before inference: {torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB")

    # Perform inference with mixed precision
    with torch.no_grad():
        with autocast():
            enhanced_images = net(batch_images)

    # Print memory usage after inference
    print(f"After inference: {torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB")

    # Save enhanced images
    for img, result_path in zip(enhanced_images, result_paths):
        if not os.path.exists(os.path.dirname(result_path)):
            os.makedirs(os.path.dirname(result_path))
        torchvision.utils.save_image(img, result_path)

    # Clean up
    del batch_images, enhanced_images
    torch.cuda.empty_cache()

    # Print memory usage after cleanup
    print(f"After cleanup: {torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB")


def process_images(input_folder, model_path, output_folder, batch_size=1):
    """
    Process all images in the input folder using the DIM model.

    Args:
        input_folder (str): Path to input images folder.
        model_path (str): Path to trained model weights.
        output_folder (str): Path to output folder.
        batch_size (int): Number of images to process in a batch (default=1).
    """
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Initialize device and model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = DIM().to(device)
    if torch.cuda.is_available():
        net.load_state_dict(torch.load(model_path))
    else:
        net.load_state_dict(torch.load(model_path, map_location='cpu'))

    # Get all image files
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_files = []
    for extension in image_extensions:
        image_files.extend(glob.glob(os.path.join(input_folder, extension)))
        image_files.extend(glob.glob(os.path.join(input_folder, extension.upper())))

    # Process images in batches
    for i in range(0, len(image_files), batch_size):
        batch_files = image_files[i:i + batch_size]
        batch_result_paths = [
            os.path.join(output_folder, f"{os.path.splitext(os.path.basename(path))[0]}_enhanced.png")
            for path in batch_files
        ]

        print(f"Processing batch {i // batch_size + 1}/{len(image_files) // batch_size + 1}...")
        lowlight(batch_files, net, batch_result_paths, device)

        for path in batch_result_paths:
            print(f"Saved enhanced image to {path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test ReF-DIM model on a folder of images')
    parser.add_argument('--input_folder', '-i', type=str, required=True, help='Path to input images folder')
    parser.add_argument('--model_path', '-m', type=str, required=True, help='Path to trained model')
    parser.add_argument('--output_folder', '-o', type=str, required=True, help='Path to output folder')
    parser.add_argument('--batch_size', '-b', type=int, default=1, help='Batch size for processing images')

    args = parser.parse_args()

    # Process images
    process_images(args.input_folder, args.model_path, args.output_folder, args.batch_size)
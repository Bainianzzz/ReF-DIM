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
        enhanced_images, _ = net(batch_images)

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test ReF-DIM model on a folder of images')
    parser.add_argument('--input_folder', '-i', type=str, default=r'D:\Dataset\LOD\images\val', help='Path to input images folder')
    parser.add_argument('--model_path', '-m', type=str, default='snapshot/best.pth', help='Path to trained model')
    parser.add_argument('--output_folder', '-o', type=str, default=r'D:\Dataset\LOD\images\elephant-37', help='Path to output folder')

    args = parser.parse_args()

    # Process images
    # Ensure output folder exists
    os.makedirs(args.output_folder, exist_ok=True)

    # Initialize device and model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = DIM().to(device)

    if torch.cuda.is_available():
        state_dict = torch.load(args.model_path)

        filtered_state_dict = {}
        for k, v in state_dict.items():
            if 'total_ops' not in k and 'total_params' not in k:
                filtered_state_dict[k] = v
        net.load_state_dict(filtered_state_dict, strict=False)
    else:
        net.load_state_dict(torch.load(args.model_path, map_location='cpu'))

    # Get all image files
    image_files = [
        os.path.join(args.input_folder, f) for f in os.listdir(args.input_folder)
    ]

    # Process images in batches
    for i in range(0, len(image_files)):
        batch_files = image_files[i:i + 1]
        batch_result_paths = [
            os.path.join(args.output_folder, f"{os.path.splitext(os.path.basename(path))[0]}.png")
            for path in batch_files
        ]

        lowlight(batch_files, net, batch_result_paths, device)

        for path in batch_result_paths:
            print(f"Saved enhanced image to {path}")
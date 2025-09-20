import torch
import torchvision
import torch.optim
import os
from model.DIM import DIM
import numpy as np
from PIL import Image
import argparse
import glob
import torchvision.transforms as transforms


def lowlight(image_path, model, result_path):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # 使用torchvision.transforms简化预处理
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    data_lowlight = Image.open(image_path).convert("RGB")
    data_lowlight = transform(data_lowlight).unsqueeze(0).to(device)

    net = DIM().to(device)
    if torch.cuda.is_available():
        net.load_state_dict(torch.load(model))
    else:
        net.load_state_dict(torch.load(model, map_location='cpu'))

    enhanced_image = net(data_lowlight)

    if not os.path.exists(os.path.dirname(result_path)):
        os.makedirs(os.path.dirname(result_path))

    torchvision.utils.save_image(enhanced_image, result_path)


def process_images(input_folder, model_path, output_folder):
    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)
    
    # 支持的图片格式
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']

    # 获取所有图片文件
    image_files = []
    for extension in image_extensions:
        image_files.extend(glob.glob(os.path.join(input_folder, extension)))
        image_files.extend(glob.glob(os.path.join(input_folder, extension.upper())))
    
    # 处理每张图片
    with torch.no_grad():
        for image_path in image_files:
            # 获取文件名（不含扩展名）
            filename = os.path.splitext(os.path.basename(image_path))[0]
            
            # 构建输出路径
            result_path = os.path.join(output_folder, f"{filename}_enhanced.png")
            
            # 处理图片
            print(f"Processing {image_path}...")
            lowlight(image_path, model_path, result_path)
            print(f"Saved enhanced image to {result_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test ReF-DIM model on a folder of images')
    parser.add_argument('--input_folder', '-i', type=str, required=True, help='Path to input images folder')
    parser.add_argument('--model_path', '-m', type=str, required=True, help='Path to trained model')
    parser.add_argument('--output_folder', '-o', type=str, required=True, help='Path to output folder')
    
    args = parser.parse_args()
    
    # test_images
    with torch.no_grad():
        # 遍历并处理文件夹中所有图片
        process_images(args.input_folder, args.model_path, args.output_folder)
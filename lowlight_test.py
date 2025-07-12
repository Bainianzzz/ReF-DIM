import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
import torch.optim
import os
import sys
import argparse
import time

from tqdm import tqdm

# import dataloader
from model.ReF_DIM_C2f import ReF_DIM
import numpy as np
from torchvision import transforms
from PIL import Image
import glob
import time


def lowlight(image_path):
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    data_lowlight = Image.open(image_path).convert("RGB")

    data_lowlight = (np.asarray(data_lowlight) / 255.0)

    data_lowlight = torch.from_numpy(data_lowlight).float()
    data_lowlight = data_lowlight.permute(2, 0, 1)
    data_lowlight = data_lowlight.cuda().unsqueeze(0)

    net = ReF_DIM().cuda()
    net.load_state_dict(torch.load('snapshot_final/model_gradnorm_100.pth'))
    # start = time.time()
    enhanced_image = net(data_lowlight)

    # end_time = (time.time() - start)
    # print(end_time)
    image_path = image_path.replace('val', 'DIM_final_100')
    # image_path = image_path.replace('input', 'output')
    if not os.path.exists(os.path.dirname(image_path)):
        os.makedirs(os.path.dirname(image_path))
    result_path = image_path
    # if not os.path.exists(image_path.replace('/' + image_path.split("/")[-1], '')):
    #     os.makedirs(image_path.replace('/' + image_path.split("/")[-1], ''))

    torchvision.utils.save_image(enhanced_image, result_path)


if __name__ == '__main__':
    # test_images
    with torch.no_grad():
        filePath = r'D:\datasets\LOD\images\val'
        # filePath = r'test_data/input'

        file_list = os.listdir(filePath)

        for file_name in tqdm(file_list):
            # test_list = glob.glob(filePath+file_name+"/*")
            # for image in test_list:
            image = os.path.join(filePath, file_name)
            # print(file_name)
            lowlight(image)

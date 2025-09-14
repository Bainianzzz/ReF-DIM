import torch
import torchvision
import torch.optim
import os
from model.ReF_DIM_C2f import ReF_DIM
import numpy as np
from PIL import Image


def lowlight(image_path, model, result_path):
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    data_lowlight = Image.open(image_path).convert("RGB")
    data_lowlight = (np.asarray(data_lowlight) / 255.0)
    data_lowlight = torch.from_numpy(data_lowlight).float()
    data_lowlight = data_lowlight.permute(2, 0, 1)
    data_lowlight = data_lowlight.unsqueeze(0)

    net = ReF_DIM()
    if torch.cuda.is_available():
        net.load_state_dict(torch.load(model))
    else:
        net.load_state_dict(torch.load(model, map_location='cpu'))

    enhanced_image = net(data_lowlight)

    if not os.path.exists(os.path.dirname(result_path)):
        os.makedirs(os.path.dirname(result_path))

    torchvision.utils.save_image(enhanced_image, result_path)


if __name__ == '__main__':
    # test_images
    with torch.no_grad():
        image = 'input.jpg'
        model = 'path/to/weight.pth'
        result_path = 'result/output.jpg'
        lowlight(image, model, result_path)

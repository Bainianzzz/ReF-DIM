import torch
from thop import profile


def count_flops_and_params(model, input_shape=(1, 3, 224, 224), device=None):
    """
    计算模型的 FLOPs 和参数数量
    
    Args:
        model: PyTorch 模型
        input_shape: 输入张量的形状，默认为 (1, 3, 224, 224)
        device: 设备，如果为 None 则根据 CUDA 是否可用自动选择
    
    Returns:
        tuple: (flops, params) FLOPs 和参数数量
    """
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # 创建虚拟输入
    dummy_input = torch.randn(*input_shape).to(device)
    
    # 计算 FLOPs 和参数
    flops, params = profile(model, inputs=(dummy_input,), verbose=False)
    
    return flops, params


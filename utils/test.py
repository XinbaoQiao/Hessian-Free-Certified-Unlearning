

import torch
print(torch.cuda.is_available())  # 检查是否可以使用 GPU
print(torch.cuda.current_device())  # 显示当前使用的 GPU 设备索引
print(torch.cuda.get_device_name(torch.cuda.current_device()))  # 显示当前 GPU 的名称

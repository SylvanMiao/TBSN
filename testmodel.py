import torch
import torch.nn as nn
from model.apbsn import APBSNModel
from torchsummary import summary

# 加载checkpoint到GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
checkpoint = torch.load('./experiments/0521-001300_APBSNModel_TBSN/log/bsn_iter_00100000.pth', map_location=device)

# 需要先定义你的模型类（与训练时相同）
model = APBSNModel(opt=).to(device)
model.load_state_dict(checkpoint['model_state_dict'])

# 打印每一层的输入输出形状
summary(model, input_size=(1, 512, 512))  # (channels, height, width)
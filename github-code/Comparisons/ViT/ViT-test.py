import torch
import numpy as np
import torch.nn as nn
import torchvision
from PIL import Image
from torch.nn import Linear
from torchvision import transforms as T
from transformers import AutoFeatureExtractor, ViTForImageClassification
from torch.utils.data import DataLoader, Dataset
# from main import VGGNet_Transfer
from label import Waterlevel
torch.set_printoptions(threshold=np.inf)   # 将所有数据显示完整！！！！

# 测试所保存的模型
path = '/home/project/waterdepth/ViT/ViT-best.pt'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = ViTForImageClassification.from_pretrained('/home/project/waterdepth/ViT_weights')
model.classifier = nn.Linear(model.config.hidden_size, 5)
model.load_state_dict(torch.load(path, map_location = device))   # 加载预训练权重
print(model)

model = model.to(device)
if torch.cuda.device_count() > 1:
    print("使用多个GPU进行训练...")
    model = nn.DataParallel(model)

# 读取测试集中的数据
root_test = '/home/project/waterdepth/single'
test_dataset = Waterlevel(root_test)
test_dataloader = DataLoader(test_dataset, batch_size=1200, shuffle=True)
with torch.no_grad():
    for batch_idx, (img, target, path) in enumerate(test_dataloader):
        img, target = img.to(device), target.to(device)
        test_output = model(img)
        logits = test_output.logits
        test_output = logits.to(torch.float32)  # 将对入参和出参做一个类型转换便于计算MSE的值
        test_target = target.to(torch.float32)
        _, predicted = torch.max(test_output.data, dim=1)
        # print(paths)
        # print(test_output)
        print(predicted)
        print(test_target)

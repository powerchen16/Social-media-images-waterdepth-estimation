import torch
import numpy as np
import torch.nn as nn
import torchvision
from PIL import Image
from torch.nn import Linear
from torchvision import transforms as T
from torchvision.models import resnet50
from transformers import AutoFeatureExtractor, SwinForImageClassification
from torch.utils.data import DataLoader, Dataset
from label import Waterlevel
torch.set_printoptions(threshold=np.inf)   # 将所有数据显示完整！！！！

# 测试所保存的模型
path = '/home/cuiaoxue/project/qianmengchen/waterdepth/Resnet-SwinT/Res-SwinT-rank-best-noweight.pt'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
state_dict = torch.load(path, map_location=device)  # 加载.pt文件，得到state_dict字典
new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}  # 删除每个参数名中的'module.'


# 定义分类模型
class ResNetSwinParallel(nn.Module):
    def __init__(self, num_classes=5):
        super(ResNetSwinParallel, self).__init__()

        # 加载预训练的ResNet模型，去掉最后全连接层，用于特征提取
        self.resnet = resnet50(pretrained=False)
        # checkpoint = '/home/cuiaoxue/project/qianmengchen/waterdepth/Resnet_weight/resnet50-19c8e357.pth'
        # state_dict = torch.load(checkpoint)
        # self.resnet.load_state_dict(state_dict)
        self.resnet.fc = nn.Identity()

        # 加载预训练的Swin Transformer模型，替换默认分类头
        self.swin_transformer = SwinForImageClassification.from_pretrained('/home/cuiaoxue/project/qianmengchen/waterdepth/SwinT_weights')
        # self.swin_transformer.load_state_dict(torch.load('/home/cuiaoxue/project/qianmengchen/waterdepth/SwinT_weights/pytorch_model.bin'))
        self.swin_transformer.classifier = nn.Identity()

        # 定义分类头，将提取的特征连接起来用于分类
        self.classifier = nn.Linear(2048 + self.swin_transformer.config.hidden_size, num_classes)

    def forward(self, x):
        # 使用ResNet 和 Swin Transformer提取特征
        resnet_features = self.resnet(x)
        swin_output = self.swin_transformer(x)
        swin_features = swin_output.logits  # 获取logits或者主要输出

        # 结合两个模型的特征，使用的是拼接的方法
        combined_features = torch.cat((resnet_features, swin_features), dim=1)

        # 使用分类头获取分类结果
        output = self.classifier(combined_features)
        return output
# 定义分类模型
num_classes = 5  # 假设我们有5个类别
model = ResNetSwinParallel(num_classes=num_classes)
model.load_state_dict(new_state_dict)   # 加载预训练权重
model.eval()  # 切换到评估模式
print(model)

model = model.to(device)
if torch.cuda.device_count() > 1:
    print("使用多个GPU进行训练...")
    model = nn.DataParallel(model)

# 读取测试集中的数据
root_test = '/home/cuiaoxue/project/qianmengchen/waterdepth/single'
test_dataset = Waterlevel(root_test)
test_dataloader = DataLoader(test_dataset, batch_size=1200, shuffle=True)
with torch.no_grad():
    for batch_idx, (img, target, path) in enumerate(test_dataloader):
        img, target = img.to(device), target.to(device)
        test_output = model(img)
        # logits = test_output.logits
        test_output = test_output.to(torch.float32)  # 将对入参和出参做一个类型转换便于计算MSE的值
        test_target = target.to(torch.float32)
        _, predicted = torch.max(test_output.data, dim=1)
        # print(paths)
        # print(test_output)
        print(predicted)
        print(test_target)

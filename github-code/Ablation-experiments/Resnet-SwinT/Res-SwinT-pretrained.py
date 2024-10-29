import torch
import numpy as np
import torchvision.models
from matplotlib import pyplot as plt
from PIL import Image
from transformers import AutoFeatureExtractor, SwinForImageClassification
from tqdm import tqdm
from torch import nn
from torchvision.models import resnet50
from label import Waterlevel
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from functools import partial
from collections import OrderedDict
from typing import Optional, Callable
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torchvision.models import swin_transformer

# 1. prepare data
root = '../train-2'
train_dataset = Waterlevel(root)
train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True)

root_val = '../test-2'
val_dataset = Waterlevel(root_val)
val_dataloader = DataLoader(val_dataset, batch_size=128, shuffle=True)

class ResNetSwinParallel(nn.Module):
    def __init__(self, num_classes=5):
        super(ResNetSwinParallel, self).__init__()

        # 加载预训练的ResNet模型，去掉最后全连接层，用于特征提取
        self.resnet = resnet50(pretrained=True)
        self.resnet.fc = nn.Identity()

        # 加载预训练的Swin Transformer模型，替换默认分类头
        self.swin_transformer = SwinForImageClassification.from_pretrained('../SwinT_weights')
        self.swin_transformer.load_state_dict(torch.load('../SwinT_weights/pytorch_model.bin'))
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

# 实例化模型
num_classes = 5  # 假设我们有5个类别
model = ResNetSwinParallel(num_classes=num_classes)
print(model)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
if torch.cuda.device_count() > 1:
    print("使用多个GPU进行训练...")
    model = nn.DataParallel(model)

criterion = nn.CrossEntropyLoss()   # 做分类使用交叉熵损失函数
learning_rate = 1e-4
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

train_acc_list = []
train_loss_list = []
val_acc_list = []
val_loss_list = []
epoch_list = []
precision_list = []
recall_list = []
f1_list = []
for epoch in range(100):
    model.train()
    train_correct = 0
    train_total = 0
    train_loss = 0.0
    for batch_idx, (data, target, path) in enumerate(train_dataloader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        # data = (data - torch.min(data)) / (torch.max(data) - torch.min(data))
        # data = feature_extractor(images=data, return_tensor='pt')
        output = model(data)

        # output = output.to(torch.float32)       # 将对入参和出参做一个类型转换
        # target = target.to(torch.float32)
        loss = criterion(output, target.long())
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

        _, predicted = torch.max(output, dim=1)
        train_total += target.size(0)
        train_correct += (predicted == target).sum().item()

    acc_train = train_correct / train_total
    train_acc_list.append(acc_train)
    train_loss_list.append(train_loss)
    # print('epoch %d, loss: %f' % (epoch, loss.item()))

    # val
    model.eval()
    correct = 0
    total = 0
    val_loss_all = 0.0
    y_true = []   # 真实值以及预测值
    y_pred = []
    with torch.no_grad():
        for batch_idx, (data, target, path) in enumerate(val_dataloader):
            data, target = data.to(device), target.to(device)

            # data = (data - torch.min(data)) / (torch.max(data) - torch.min(data))
            # data = feature_extractor(images=data, return_tensor='pt')
            output = model(data)

            val_loss = criterion(output, target.long())             # 记录一下验证集的损失
            val_loss_all += val_loss.item()

            _, predicted = torch.max(output, dim=1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

            y_true.extend(target.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
    acc_val = correct / total

    val_acc_list.append(acc_val)
    val_loss_list.append(val_loss_all)
    epoch_list.append(epoch)

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro')   # macro表示平均
    precision_list.append(precision)
    recall = recall_score(y_true, y_pred, average='macro')
    recall_list.append(recall)
    f1 = f1_score(y_true, y_pred, average='macro')
    f1_list.append(f1)


    # save model
    torch.save(model.state_dict(), "ReSwin-last.pt")
    if acc_val == max(val_acc_list):
        torch.save(model.state_dict(), "ReSwin-best.pt")
        print("save epoch {} model".format(epoch))
    print("epoch = {},loss = {},acc = {},val_loss = {},acc_val = {}".format(epoch, train_loss, acc_train, val_loss_all, acc_val))
    print("Accuracy = {}, Precision = {}, Recall = {}, F1 score = {}".format(accuracy, precision, recall, f1))

print(epoch_list)
print(train_acc_list)
print(train_loss_list)
print(val_loss_list)
print(val_acc_list)
print("accuracy的平均值为：", np.mean(val_acc_list))
print("precision的平均值为：", np.mean(precision_list))
print("recall的平均值为：", np.mean(recall_list))
print("f1的平均值为：", np.mean(f1_list))

# 绘制每个epoch的变化图
fig, ax = plt.subplots()
line1 = ax.plot(epoch_list, train_loss_list, color='green', label="train_loss")
line3 = ax.plot(epoch_list, val_loss_list, color='blue', label='val_loss')
ax.set_ylabel('loss')
ax.set_xlabel('epoch')
plt.legend()
ax.spines['right'].set_visible(False)       # ax右轴隐藏

z_ax = ax.twinx() # 创建与轴群ax共享x轴的轴群z_ax
line2 = z_ax.plot(epoch_list, val_acc_list, color='red', label="val_acc")
line4 = z_ax.plot(epoch_list, train_acc_list, color='black', label="train_acc")
z_ax.set_ylabel('acc')

lns = line1+line2+line3+line4
# lns = line1
labs = [l.get_label() for l in lns]
ax.legend(lns, labs, loc=0)
plt.savefig('ReSwin-128-1e-4-数增-1e-4.png')
plt.show()

import torch
import torch.nn as nn
import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, Dataset
# from main import VGGNet_Transfer
from label import Waterlevel
import torchvision
from torch.nn import Linear

# 1. prepare data
root = r'D:\pycharm\project\water depth prediction\dataset\train-2'
train_dataset = Waterlevel(root)
train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True)

root_val = r'D:\pycharm\project\water depth prediction\dataset\test-2'
val_dataset = Waterlevel(root_val)
val_dataloader = DataLoader(val_dataset, batch_size=128, shuffle=True)

# 2. load model
densenet_true = torchvision.models.densenet121(pretrained=True)
# densenet_true.classifier = Linear(2048, 10)
num_ftrs = densenet_true.classifier.in_features
densenet_true.classifier = nn.Linear(num_ftrs, 5)
model = densenet_true
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
print(model)
# 3. prepare super parameters
criterion = nn.CrossEntropyLoss()  # 做分类使用交叉熵损失函数
# criterion = nn.MSELoss()    #做回归使用MSE(均方误差)、以及均方根误差（RMSE）
learning_rate = 1e-4
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# learning_rate = 1e-2
# optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
# print(optimizer)

# # 调整学习率
# for param_group in optimizer.param_groups:
#         param_group['lr'] *= 0.1  # 学习率为之前的0.1倍

# 4. train
train_loss_list = []
val_acc_list = []
val_loss_list = []
epoch_list = []
for epoch in range(100):
    model.train()
    train_loss = 0.0
    for batch_idx, (data, target) in enumerate(train_dataloader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        # output = output.to(torch.float32)       # 将对入参和出参做一个类型转换
        # target = target.to(torch.float32)
        loss = criterion(output, target.long())
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss_list.append(train_loss)
    print('epoch %d, loss: %f' % (epoch, loss.item()))

    # val
    model.eval()
    correct = 0
    total = 0
    val_loss_all = 0.0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(val_dataloader):
            data, target = data.to(device), target.to(device)
            output = model(data)

            val_loss = criterion(output, target.long())             # 记录一下验证集的损失
            val_loss_all += val_loss.item()

            _, predicted = torch.max(output.data, dim=1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    acc_val = correct / total

    val_acc_list.append(acc_val)
    val_loss_list.append(val_loss_all)
    epoch_list.append(epoch)
    # save model
    torch.save(model.state_dict(), "last.pt")
    if acc_val == max(val_acc_list):
        torch.save(model.state_dict(), "best.pt")
        print("save epoch {} model".format(epoch))
    print("epoch = {},  loss = {},val_loss = {} acc_val = {}".format(epoch, train_loss, val_loss_all,  acc_val))
print(epoch_list)
print(train_loss_list)
print(val_loss_list)
print(val_acc_list)


import torch
import numpy as np
import torchvision
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from matplotlib import pyplot as plt
from rank_label import WaterlevelPairDataset
from label import Waterlevel
from torchvision.models import resnet50
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from transformers import AutoFeatureExtractor, SwinForImageClassification
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 定义分类模型
class ResNetSwinParallel(nn.Module):
    def __init__(self, num_classes=5):
        super(ResNetSwinParallel, self).__init__()

        # 加载预训练的ResNet模型，去掉最后全连接层，用于特征提取
        self.resnet = resnet50(pretrained=False)
        checkpoint = '/home/cuiaoxue/project/qianmengchen/waterdepth/Resnet_weight/resnet50-19c8e357.pth'
        state_dict = torch.load(checkpoint)
        self.resnet.load_state_dict(state_dict)
        self.resnet.fc = nn.Identity()

        # 加载预训练的Swin Transformer模型，替换默认分类头
        self.swin_transformer = SwinForImageClassification.from_pretrained('/home/cuiaoxue/project/qianmengchen/waterdepth/SwinT_weights')
        self.swin_transformer.load_state_dict(torch.load('/home/cuiaoxue/project/qianmengchen/waterdepth/SwinT_weights/pytorch_model.bin'))
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


# 损失函数包含回归部分和排名部分
# 分类损失函数（比如对于二进制标签的多标签分类问题）
if __name__ == "__main__":
    classification_loss_function = nn.CrossEntropyLoss()
    # 排序损失函数
    margin_ranking_loss = nn.MarginRankingLoss(margin=0.0)
    # 组合这两种损失的函数
    def combined_loss(predictions_classification, targets_classification, predictions_img1, predictions_img2, targets_rank,
                      classification_loss_weight=1.0, ranking_loss_weight=1.0):
        # 计算分类损失
        loss_classification = classification_loss_function(predictions_classification, targets_classification)

        # 计算排序损失
        loss_ranking = margin_ranking_loss(predictions_img1, predictions_img2, targets_rank)

        # 组合这两种损失
        combined_loss = classification_loss_weight * loss_classification + ranking_loss_weight * loss_ranking
        return combined_loss

    # 加载数据集
    root = '/home/cuiaoxue/project/qianmengchen/waterdepth/train-2'
    object_dataset = Waterlevel(root)
    object_loader = DataLoader(object_dataset, batch_size=128, shuffle=True)
    image_dataset = WaterlevelPairDataset(root)
    image_loader = DataLoader(image_dataset, batch_size=128, shuffle=True)

    root = '/home/cuiaoxue/project/qianmengchen/waterdepth/test-2'
    val_dataset = Waterlevel(root)
    val_dataloader = DataLoader(val_dataset, batch_size=128, shuffle=True)

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
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay = 1e-4)

    # 训练过程
    num_epochs = 100
    train_acc_list = []
    train_loss_list = []
    val_acc_list = []
    val_loss_list = []
    epoch_list = []
    precision_list = []
    recall_list = []
    f1_list = []
    for epoch in range(num_epochs):
        # Object Regression
        model.train()
        train_correct = 0
        train_total = 0
        train_loss = 0.0
        for (obj_inputs, obj_targets), (img1_inputs, img2_inputs, rank_targets) in zip(object_loader, image_loader):
            obj_inputs, obj_targets = obj_inputs.float().cuda(), obj_targets.float().cuda()
            img1_inputs, img2_inputs, rank_targets = img1_inputs.float().cuda(), img2_inputs.float().cuda(), rank_targets.float().cuda()
            optimizer.zero_grad()
            # 获取回归和排名的预测值
            obj_predictions = model(obj_inputs)
            img1_predictions = model(img1_inputs)
            img2_predictions = model(img2_inputs)
            # 计算两个部分的损失
            total_loss = combined_loss(obj_predictions, obj_targets.long(), img1_predictions, img2_predictions, rank_targets, classification_loss_weight=1.0, ranking_loss_weight=1.0)
            # 反向传播
            total_loss.backward()
            optimizer.step()

            train_loss += total_loss.item()

            _, predicted = torch.max(obj_predictions.data, dim=1)
            train_total += obj_targets.size(0)
            train_correct += (predicted == obj_targets).sum().item()

        acc_train = train_correct / train_total
        train_acc_list.append(acc_train)
        train_loss_list.append(train_loss)

        model.eval()
        correct = 0
        total = 0
        val_loss_all = 0.0
        y_true = []   # 真实值以及预测值
        y_pred = []
        with torch.no_grad():
            for val_inputs, val_targets in val_dataloader:
                val_inputs, val_targets = val_inputs.cuda(), val_targets.cuda()
                val_outputs = model(val_inputs)

                val_loss = classification_loss_function(val_outputs, val_targets.long())
                val_loss_all += val_loss.item()

                _, predicted = torch.max(val_outputs.data, dim=1)
                total += val_targets.size(0)
                correct += (predicted == val_targets).sum().item()

                y_true.extend(val_targets.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy())
        acc_val = correct / total

        val_acc_list.append(acc_val)
        val_loss_list.append(val_loss_all)
        epoch_list.append(epoch)

        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='macro')  # macro表示平均
        precision_list.append(precision)
        recall = recall_score(y_true, y_pred, average='macro')
        recall_list.append(recall)
        f1 = f1_score(y_true, y_pred, average='macro')
        f1_list.append(f1)

        # save model
        torch.save(model.state_dict(), "/home/cuiaoxue/project/qianmengchen/waterdepth/Resnet-SwinT/Res-SwinT-rank-last-2.pt")
        if acc_val == max(val_acc_list):
            torch.save(model.state_dict(), "/home/cuiaoxue/project/qianmengchen/waterdepth/Resnet-SwinT/Res-SwinT-rank-best-2.pt")
            print("save epoch {} model".format(epoch))
        print("epoch = {},loss = {},acc = {},val_loss = {},acc_val = {}".format(epoch, train_loss, acc_train,
                                                                                val_loss_all, acc_val))
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
    ax.spines['right'].set_visible(False)  # ax右轴隐藏

    z_ax = ax.twinx()  # 创建与轴群ax共享x轴的轴群z_ax
    line2 = z_ax.plot(epoch_list, val_acc_list, color='red', label="val_acc")
    line4 = z_ax.plot(epoch_list, train_acc_list, color='black', label="train_acc")
    z_ax.set_ylabel('acc')

    lns = line1 + line2 + line3 + line4
    # lns = line1
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs, loc=0)
    plt.savefig('/home/cuiaoxue/project/qianmengchen/waterdepth/Resnet-SwinT/Res-SwinT-rank-128-1e-4-数增-1e-4-2.png')
    plt.show()





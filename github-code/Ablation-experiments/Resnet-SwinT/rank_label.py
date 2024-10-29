import os
import torch
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as T
import random
from torch.utils.data import DataLoader, Dataset


class WaterlevelPairDataset(data.Dataset):
    def __init__(self, root, transforms=None):
        self.level_to_images = {}
        # 遍历所有文件夹，把同一水平的图像分到一组
        for level in range(5):  # 有五个水位等级，从0到4
            level_path = os.path.join(root, f"level {level}")
            self.level_to_images[level] = [os.path.join(level_path, img) for img in os.listdir(level_path)]

        if transforms is None:
            normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            self.transforms = T.Compose([
                T.Resize((224, 224)),
                T.CenterCrop((224, 224)),
                T.ToTensor(),
                normalize
            ])
        else:
            self.transforms = transforms

    def __getitem__(self, index):
        # 随机选择两个不同水位等级
        level1, level2 = random.sample(range(5), 2)
        # 随机选择每个水位等级中的一张图像
        img_path1 = random.choice(self.level_to_images[level1])
        img_path2 = random.choice(self.level_to_images[level2])

        img1 = Image.open(img_path1)
        img2 = Image.open(img_path2)
        if img1.mode != "RGB":
            img1 = img1.convert("RGB")
        if img2.mode != "RGB":
            img2 = img2.convert("RGB")

        img1 = self.transforms(img1)
        img2 = self.transforms(img2)

        # 分配排名标签
        rank_label = torch.FloatTensor([1 if level1 > level2 else -1])

        return img1, img2, rank_label

    def __len__(self):
        # 由于是随机生成对，理论上是无限的，但实际上可以返回一个足够大的数字
        # 或者是所有可能成对组合的数量
        return sum(len(images) for images in self.level_to_images.values())

if __name__ == "__main__":
    # root = "/home/elvis/workfile/dataset/car/train"
    root = r'D:\pycharm\project\Spatiotemporal Analysis\water depth prediction2.0\dataset\test-2'
    train_dataset = WaterlevelPairDataset(root)
    train_dataloader = DataLoader(train_dataset, batch_size=5, shuffle=True)
    for img1, img2, rank_label  in train_dataloader:
        # print(data.shape)
        print(img1, img2, rank_label)
        break
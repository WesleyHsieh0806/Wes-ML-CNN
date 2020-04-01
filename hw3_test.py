import matplotlib.pyplot as plt
import sys
import time
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import torchvision.transforms as transforms
import torch.nn as nn
import torch
import cv2
import numpy as np
import os
import random
import wget

torch.manual_seed(0)
np.random.seed(0)
torch.cuda.manual_seed(0)
"""# Testing
利用剛剛 train 好的 model 進行 prediction
"""


def readfile(path, label):
    # label 是一個 boolean variable，代表需不需要回傳 y 值
    image_dir = sorted(os.listdir(path))
    x = np.zeros((len(image_dir), 128, 128, 3), dtype=np.uint8)
    y = np.zeros((len(image_dir)), dtype=np.uint8)
    for i, file in enumerate(image_dir):
        img = cv2.imread(os.path.join(path, file))
        x[i, :, :] = cv2.resize(img, (128, 128))
        if label:
            y[i] = int(file.split("_")[0])
    if label:
        return x, y
    else:
        return x


workspace_dir = sys.argv[1]
print("Reading data")
# train_x, train_y = readfile(os.path.join(workspace_dir, "training"), True)
# print("Size of training data = {}".format(len(train_x)))
# val_x, val_y = readfile(os.path.join(workspace_dir, "validation"), True)
# print("Size of validation data = {}".format(len(val_x)))
test_x = readfile(os.path.join(workspace_dir, "testing"), False)
print("Size of Testing data = {}".format(len(test_x)))


# training 時做 data augmentation
train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(),  # 隨機將圖片水平翻轉
    transforms.RandomRotation(15),  # 隨機旋轉圖片
    # 將圖片轉成 Tensor，並把數值normalize到[0,1](data normalization)
    transforms.ToTensor(),
])
# testing 時不需做 data augmentation
test_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
])


class ImgDataset(Dataset):
    def __init__(self, x, y=None, transform=None):
        self.x = x
        # label is required to be a LongTensor
        self.y = y
        if y is not None:
            self.y = torch.LongTensor(y)
        self.transform = transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        X = self.x[index]
        # make a seed with numpy generator
        seed = np.random.randint(2147483647)
        random.seed(seed)

        if self.transform is not None:
            X = self.transform(X)
        if self.y is not None:
            Y = self.y[index]
            return X, Y
        else:
            return X


batch_size = 64

"""# Model"""


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        #torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        #torch.nn.MaxPool2d(kernel_size, stride, padding)
        # input 維度 [3, 128, 128]
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),  # [64, 128, 128]
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),      # [64, 64, 64]

            nn.Conv2d(64, 128, 3, 1, 1),  # [128, 64, 64]
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),      # [128, 32, 32]

            nn.Conv2d(128, 256, 3, 1, 1),  # [256, 32, 32]
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),      # [256, 16, 16]

            nn.Conv2d(256, 512, 3, 1, 1),  # [512, 16, 16]
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),       # [512, 8, 8]

            nn.Conv2d(512, 512, 3, 1, 1),  # [512, 8, 8]
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),       # [512, 4, 4]
        )
        self.fc = nn.Sequential(
            nn.Linear(512*4*4, 1024),
            nn.Dropout(p=0.5),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.Dropout(p=0.5),
            nn.ReLU(),
            nn.Linear(512, 11),


        )

    def forward(self, x):
        out = self.cnn(x)
        out = out.view(out.size()[0], -1)
        return self.fc(out)


accuracy = []
val_accuracy = []

wget.download(
    'https://github.com/WesleyHsieh0806/hello/releases/download/0.0.0/CNN_model.h5', out='CNN_model.h5')
# Load model
model_best = Classifier().cuda()
model_best.load_state_dict(torch.load("CNN_model.h5"))


test_set = ImgDataset(test_x, transform=test_transform)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)


''' predict'''
model_best.eval()
prediction = []
with torch.no_grad():
    for i, data in enumerate(test_loader):
        test_pred = model_best(data.cuda())
        test_label = np.argmax(test_pred.cpu().data.numpy(), axis=1)
        for y in test_label:
            prediction.append(y)

# 將結果寫入 csv 檔
pred_file = sys.argv[2]
with open(pred_file, 'w') as f:
    f.write('Id,Category\n')
    for i, y in enumerate(prediction):
        f.write('{},{}\n'.format(i, y))

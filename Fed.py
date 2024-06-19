import timm
import torch
from torch import nn, optim, load, save, manual_seed, cuda, tensor
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision import transforms
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import os
import multiprocessing
from datetime import datetime
from typing import Optional
import random
import numpy as np
from PIL import Image
from scipy.special import softmax
import copy
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, confusion_matrix
from imblearn.metrics import sensitivity_score, specificity_score

class FdDataset(Dataset):
    def __init__(self, root_dir, csv_file, transform=None):
        """
        :param root_dir: 所有数据集根路径
        :param csv_file: path to the file containing images with corresponding labels.
        :param transform: optional transform to be applied on a sample.
        """
        super(FdDataset, self).__init__()
        import pandas as pd
        file = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.images = file['ImageID'].values
        self.labels = file.iloc[:, 1:].values.astype(int)
        self.transform = transform
        print(csv_file + '数量 images:{}, labels:{}'.format(len(self.images), len(self.labels)))

    def __getitem__(self, index):
        """
        :param index: index: the index of item
        :return: image and its labels
        """
        # items = self.images[index]
        image_name = os.path.join(self.root_dir, self.images[index])
        image = Image.open(image_name).convert('RGB')
        label = self.labels[index]
        if self.transform is not None:
            image = self.transform(image)
        # return items, index, image, torch.FloatTensor(label)
        return image, np.where(label == 1)[0][0]

    def __len__(self):
        return len(self.images)


def fed_weighted_average(w: list, dict_users_list: list):
    weight = [len(i) for i in dict_users_list]
    weight = [wei / sum(weight) for wei in weight]
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        w_avg[k] = torch.zeros_like(w_avg[k])
    for k in w_avg.keys():
        for i in range(len(w)):
            if w[i][k].dtype == torch.int64:
                w_avg[k] += int(w[i][k] * weight[i])
            else:
                w_avg[k] += torch.mul(w[i][k], weight[i])
    return w_avg


def test_acc(model, test_loader, epoch: int = -1, dataset: str = ""):
    labels_total = []
    predicted_total = []
    model.cpu()
    for images, labels in tqdm(test_loader, f"Epoch {epoch}, {dataset} test"):
        output = model(images)
        _, predicted = torch.max(output.data, 1)
        labels, predicted = labels.detach().numpy(), predicted.detach().numpy()
        labels_total.extend(labels)
        predicted_total.extend(predicted)
    Accuracy = accuracy_score(labels_total, predicted_total)
    return Accuracy, labels_total, predicted_total


def test(model, test_loader, epoch: int = -1, dataset: str = "", criterion=None):
    """
    :param model:
    :param test_loader:
    :return: 这些评价指标在所有测试集上的平均值
    """
    # model.cpu()
    # outputs = []
    labels_total = []
    predicted_total = []
    mean_loss = 0
    model.cpu()
    for images, labels in tqdm(test_loader, f"Epoch {epoch}, {dataset} test"):
        output = model(images)
        if criterion:
            loss = criterion(output, labels)
            mean_loss += loss
        _, predicted = torch.max(output.data, 1)  # dim=1 预测结果类别，计算每一行的最大值
        labels, predicted = labels.detach().numpy(), predicted.detach().numpy()
        labels_total.extend(labels)
        predicted_total.extend(predicted)
    if criterion:
        mean_loss /= len(test_loader)
    AUC, Sensitivity, Specificity, Accuracy, F1, ConfusionMatrix = (
        # roc_auc_score(labels_total, predicted_total, average="macro"),
        None,
        sensitivity_score(labels_total, predicted_total, average="macro"),
        specificity_score(labels_total, predicted_total, average="macro"),
        accuracy_score(labels_total, predicted_total),
        f1_score(labels_total, predicted_total, average="macro"),
        confusion_matrix(labels_total, predicted_total)
    )
    return AUC, Sensitivity, Specificity, Accuracy, F1, ConfusionMatrix, mean_loss

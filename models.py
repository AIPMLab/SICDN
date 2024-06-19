import numpy as np
import pandas as pd
import shap
import timm
import torch
import torch.nn as nn


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=False)
        self.downsample = downsample
        self.stride = stride

    # expansion 因子通常用于控制网络的宽度
    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        outt = out + identity
        outt = self.relu(outt)

        return outt

class shapDenseNet(nn.Module):
    def __init__(self, original_densenet,w=0.6):
        super(shapDenseNet, self).__init__()
        self.features = original_densenet.features
        self.global_pool = original_densenet.global_pool
        self.head_drop = original_densenet.head_drop
        self.classifier = original_densenet.classifier
        self.w = w

    def forward(self, x):
        x = self.features(x)
        x = self.global_pool(x)
        x = self.head_drop(x)
        if self.training:
            e = shap.GradientExplainer(self.classifier, x)
            shap_values = e.shap_values(x)
            shap_values_abs = np.abs(shap_values)
            shap_values_abs = np.mean(shap_values_abs, 0)
            if np.max(shap_values_abs) == np.min(shap_values_abs):
                shap_values_normalized = np.ones(shap_values.shape)
            else:
                shap_values_normalized = ((shap_values_abs - np.min(shap_values_abs)) /
                                          (np.max(shap_values_abs) - np.min(shap_values_abs)))
            shap_values_normalized = shap_values_normalized.T
            shap_values_normalized = shap_values_normalized.squeeze()
            weight = self.classifier.weight.data.detach().cpu().numpy()
            shap_value_product = weight * shap_values_normalized
            self.classifier.weight.data = torch.tensor(shap_value_product, dtype=torch.float32, device="cuda")
        x = self.classifier(x)
        return x

class shapDenseNet1(nn.Module):
    def __init__(self, original_densenet,w=0.5):
        super(shapDenseNet1, self).__init__()
        self.features = original_densenet.features
        self.global_pool = original_densenet.global_pool
        self.head_drop = original_densenet.head_drop
        self.classifier = original_densenet.classifier
        self.w = w
    def forward(self, x):
        x = self.features(x)
        x = self.global_pool(x)
        x = self.head_drop(x)
        if self.training:
            e = shap.GradientExplainer(self.classifier, x)
            shap_values = e.shap_values(x)
            shap_values_abs = np.abs(shap_values)
            shap_values_abs = np.mean(shap_values_abs, 0)
            if np.max(shap_values_abs) == np.min(shap_values_abs):
                shap_values_normalized = np.ones(shap_values.shape)
            else:
                shap_values_normalized = ((shap_values_abs - np.min(shap_values_abs)) /
                                          (np.max(shap_values_abs) - np.min(shap_values_abs)))
            weight = self.classifier.weight.data.detach().cpu().numpy()
            weight_abs= np.abs(weight)
            if np.max(weight_abs) == np.min(weight_abs):
                weight_normalized = np.ones(weight.shape)
            else:
                weight_normalized = ((weight_abs - np.min(weight_abs)) /
                (np.max(weight_abs) - np.min(weight)))
            shap_values_normalized = shap_values_normalized.T
            shap_values_normalized = shap_values_normalized.squeeze()
            shap_value_product = weight * (self.w * shap_values_normalized + (1 - self.w) * weight_normalized)
            self.classifier.weight.data = torch.tensor(shap_value_product, dtype=torch.float32, device="cuda")
        x = self.classifier(x)
        return x

if __name__ == '__main__':
    device = "cuda"
    pmodel = timm.create_model("densenet121", num_classes=2)
    model = shapDenseNet(pmodel)
    a = torch.randn(8, 3, 512, 512)
    b = pmodel(a)
    c = model(a)
    print(b)
    print(c)


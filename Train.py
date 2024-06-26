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
import copy
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, confusion_matrix
from imblearn.metrics import sensitivity_score, specificity_score

class FdDataset(Dataset):
    def __init__(self, root_dir, csv_file, transform=None):
        """
        :param root_dir: 
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
        print(csv_file + 'number of images:{}, labels:{}'.format(len(self.images), len(self.labels)))

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


class TransformTwice:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, inp):
        out1 = self.transform(inp)
        out2 = self.transform(inp)
        return out1, out2


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


def test(model, test_loader, epoch: int = -1, dataset: str = "", criterion=None):
    """
    :param model:
    :param test_loader:
    :return:
    """
    # model.cpu()
    # outputs = []
    labels_total = []
    predicted_total = []
    mean_loss = 0
    for images, labels in tqdm(test_loader, f"Epoch {epoch + 1}, {dataset} test"):
        output = model(images)
        if criterion:
            loss = criterion(output, labels)
            mean_loss += loss
        _, predicted = torch.max(output.data, 1)  # dim=1 
        labels, predicted = labels.detach().numpy(), predicted.detach().numpy()
        labels_total.extend(labels)
        predicted_total.extend(predicted)
    if criterion:
        mean_loss /= len(test_loader)
    AUC, Sensitivity, Specificity, Accuracy, F1, PPV, ConfusionMatrix = (
        # roc_auc_score(labels_total, predicted_total, average="macro"),
        None,
        sensitivity_score(labels_total, predicted_total, average="macro"),
        specificity_score(labels_total, predicted_total, average="macro"),
        accuracy_score(labels_total, predicted_total),
        f1_score(labels_total, predicted_total, average="macro"),
        precision_score(labels_total, predicted_total, average="macro"),
        confusion_matrix(labels_total, predicted_total)
    )
    return AUC, Sensitivity, Specificity, Accuracy, F1, PPV, ConfusionMatrix, mean_loss


class FedTrainer:
    def __init__(self,
                 root_path: str,
                 # csv_file: str,
                 normalize_list: transforms,
                 net: nn.Module,
                 epoches: int = 10,
                 batch_size: int = 32,
                 # num_classes = 2,
                 seed: int = 42,
                 check_point: Optional[str] = None,
                 lr: float = 1E-3,
                 num_worker: int = min(24, multiprocessing.cpu_count()),
                 save_path: str = "./check_point",
                 tensorboard: bool = True,
                 comment: list = [],
                 databalance: int = 0):
    
        self.ROOT_PATH = root_path
        self.normalize = normalize_list
        self.NET = net
        self.seed = seed
        self.check_point = check_point
        self.EPOCHES = epoches
        self.BATCH_SIZE = batch_size
        self.LR = lr
        self.NUM_WORKER = num_worker
        self.SAVE_PATH = save_path
        self.TB = tensorboard
        self.Comment = comment
        self.Databalance = databalance

    def start(self):
        
        random.seed(self.seed)
        np.random.seed(self.seed)
        manual_seed(42)
        cuda.manual_seed(self.seed)
        
        dataset_list = os.listdir(self.ROOT_PATH)
       
        train_datasets = []
        test_datasets = []
        train_dataloaders = []
        test_dataloaders = []
        local_models = []
        local_weights = []
        device = "cuda" if cuda.is_available() else "cpu"
        for i, path in enumerate(dataset_list):  
            train_datasets.append(
                random_split(
                    FdDataset(
                        root_dir=os.path.join(self.ROOT_PATH, path, "mix"),
                        csv_file=os.path.join(self.ROOT_PATH, path, "train.csv"),
                        transform=transforms.Compose([
                            transforms.Resize((224, 224)),
                            transforms.RandomAffine(degrees=10, translate=(0.02, 0.02)),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            self.normalize[i],
                        ])
                    ), [1.0])[0]
            )
            test_datasets.append(
                random_split(
                    FdDataset(
                        root_dir=os.path.join(self.ROOT_PATH, path, "mix"),
                        csv_file=os.path.join(self.ROOT_PATH, path, "test.csv"),
                        transform=transforms.Compose([
                            transforms.Resize((224, 224)),
                            transforms.ToTensor(),
                            self.normalize[i],
                        ])
                    ), [1.0])[0]
            )
            train_dataloaders.append(DataLoader(train_datasets[i], self.BATCH_SIZE, True, num_workers=self.NUM_WORKER))
            test_dataloaders.append(DataLoader(test_datasets[i], self.BATCH_SIZE, False, num_workers=self.NUM_WORKER))
            local_models.append(copy.deepcopy(self.NET))
            local_weights.append(copy.deepcopy(self.NET.state_dict()))
        optimizers = [optim.Adam(local_models[i].parameters(), lr=self.LR) for i, _ in enumerate(dataset_list)]
        schedulers = [optim.lr_scheduler.ReduceLROnPlateau(optimizer) for optimizer in optimizers]
        writers_list = []
        fed_writers_list = []
        if self.TB:
            for dataset in dataset_list:
                writers_list.append(SummaryWriter(comment=f"_{self.NET.__class__.__name__}_{dataset}" +
                                                          (f"_{'_'.join(self.Comment)}" if self.Comment else "")))
                fed_writers_list.append(SummaryWriter(comment=f"_{self.NET.__class__.__name__}_Fed_{dataset}" +
                                                              (f"_{'_'.join(self.Comment)}" if self.Comment else "")))
            # SummaryWriter
            # writers_list.append(SummaryWriter(comment=f"_{self.NET.__class__.__name__}_eva" +
            #                                           (f"_{'_'.join(self.Comment)}" if self.Comment else "")))
        for epoch in range(self.EPOCHES):
            for i, dataset in enumerate(dataset_list):  
                print(f"dataset：{dataset}")
                if optimizers[i].param_groups[0].get("lr", 0) == 0:
                    print("lr:0")
                    break
                local_models[i].to(device)
                local_models[i].train()
                mean_loss = 0
                for images, labels in tqdm(train_dataloaders[i], f"[Epoch {epoch + 1}], train"):
                    images, labels = images.to(device), labels.to(device)
                    optimizers[i].zero_grad()
                    outputs = local_models[i](images)
                    loss = criterion(outputs, labels)
                    mean_loss += loss
                    loss.backward()
                    optimizers[i].step()
                del images, labels, outputs
                schedulers[i].step(loss.data.item())
                # del loss
                mean_loss /= len(train_dataloaders[i])
                print(f"{dataset} mean_loss:{mean_loss}, lr:{optimizers[i].param_groups[0].get('lr', 0)}\n")
                local_weights[i] = local_models[i].state_dict()
                if self.TB:
                    writers_list[i].add_scalar("info/train_mean_loss", mean_loss, epoch + 1)
            with torch.no_grad():
                global_model_weights = fed_weighted_average(local_weights, train_dataloaders)
            global_model.load_state_dict(global_model_weights)
      
            for j, dataset in enumerate(dataset_list):
                with torch.no_grad():
                    local_models[j].eval()
                    AUC, Sensitivity, Specificity, Accuracy, F1, PPV, _, mean_loss = test(
                        local_models[j].cpu(),
                        test_dataloaders[j], epoch, dataset, criterion)
                if self.TB:
                    # writers_list[j].add_scalar("info/eva_AUC", AUC, epoch + 1)
                    writers_list[j].add_scalar("info/eva_Sensitivity", Sensitivity, epoch + 1)
                    writers_list[j].add_scalar("info/eva_Specificity", Specificity, epoch + 1)
                    writers_list[j].add_scalar("info/eva_Accuracy", Accuracy, epoch + 1)
                    writers_list[j].add_scalar("info/eva_F1", F1, epoch + 1)
                    writers_list[j].add_scalar("info/eva_PPV", PPV, epoch + 1)
                    # writers_list[j].add_scalar("info/eva_mean_loss", mean_loss, epoch + 1)
                print(f"epoch{epoch + 1}_{dataset}_model "
                      f"{AUC=}, {Sensitivity=:.6f}, {Specificity=:.6f}, "
                      f"{Accuracy=:.6f}, {F1=:.6f}, {PPV=:.6f}, {mean_loss=:.6f}\n")
                # print(f"epoch{epoch + 1}_{dataset}_model "
                #       f"AUC={AUC:.6f}, Sensitivity={Sensitivity:.6f}, Specificity={Specificity:.6f}, "
                #       f"Accuracy={Accuracy:.6f}, F1={F1:.6f}, PPV={PPV:.6f}, NPV={NPV:.6f}\n")
          
                if not os.path.exists(self.SAVE_PATH):
                    os.mkdir(self.SAVE_PATH)
                save({
                    'epoch': epoch + 1,
                    'model_state_dict': local_models[j].state_dict()},
                    os.path.join(self.SAVE_PATH,
                                 f"{datetime.now().strftime('%Y%m%d-%H%M%S')}_epoch{epoch + 1}"
                                 f"_{self.NET.__class__.__name__}_{dataset}" +
                                 (f"_{'_'.join(self.Comment)}.pth" if self.Comment else ".pth")))
         
            # best_eva = [math.inf, math.inf, math.inf, math.inf, math.inf, math.inf, math.inf]
            for k, dataset in enumerate(dataset_list):
                with torch.no_grad():
                    global_model.eval()
                    # AUC,Sensitivity,Specificity,Accuracy,
                    # PPV(Positive Predictive Value),NPV(Negative Predictive Value)
                    AUC, Sensitivity, Specificity, Accuracy, F1, PPV, _, mean_loss = test(global_model.cpu(),
                                                                                          test_dataloaders[k], epoch,
                                                                                          "Fed " + dataset, criterion)
                    # min_eva = [min(x, y) for x, y in zip(best_eva, eva_result)]
                    # best_eva = best_eva if best_eva == min_eva else min_eva
                if self.TB:
                    # fed_writers_list[k].add_scalar("info/eva_AUC", AUC, epoch + 1)
                    fed_writers_list[k].add_scalar("info/eva_Sensitivity", Sensitivity, epoch + 1)
                    fed_writers_list[k].add_scalar("info/eva_Specificity", Specificity, epoch + 1)
                    fed_writers_list[k].add_scalar("info/eva_Accuracy", Accuracy, epoch + 1)
                    fed_writers_list[k].add_scalar("info/eva_F1", F1, epoch + 1)
                    fed_writers_list[k].add_scalar("info/eva_PPV", PPV, epoch + 1)
                    fed_writers_list[k].add_scalar("info/eva_mean_loss", mean_loss, epoch + 1)
                print(f"epoch{epoch + 1}_FedModel_{dataset} "
                      # f"{AUC=:.6f}, {Sensitivity=:.6f}, {Specificity=:.6f}, "
                      f"{Accuracy=:.6f}, {F1=:.6f}, {PPV=:.6f}, {mean_loss=:.6f}\n")
            checkpoint = {
                'epoch': epoch + 1,  # 
                'model_state_dict': global_model.state_dict(),  
                # 'optimizer_state_dict': optimizers[j].state_dict(), 
                # 'loss': loss, 
                # 'lr': optimizers[j].param_groups[0].get('lr', 0)
                # 
            }
            # 
            if not os.path.exists(self.SAVE_PATH):
                os.mkdir(self.SAVE_PATH)
            save(checkpoint, os.path.join(self.SAVE_PATH,
                                          f"{datetime.now().strftime('%Y%m%d-%H%M%S')}_epoch{epoch + 1}"
                                          f"_{self.NET.__class__.__name__}_Fed_{self.ROOT_PATH}" +
                                          (f"_{'_'.join(self.Comment)}.pth" if self.Comment else ".pth")))


class LocalTrainer:
    def __init__(self,
                 root_path: str,
                 # csv_file: str,
                 normalize_list: transforms,
                 net: nn.Module,
                 check_point: str,
                 epoches: int = 10,
                 batch_size: int = 32,
                 # num_classes = 2,
                 seed: int = 42,
                 lr: float = 1E-3,
                 num_worker: int = min(24, multiprocessing.cpu_count()),
                 save_path: str = "./check_point",
                 tensorboard: bool = True,
                 comment: list = [],
                 databalance: int = 0):
        self.ROOT_PATH = root_path
        # self.CSV_FILE = csv_file
        self.normalize = normalize_list
        self.NET = net
        # self.NUM_CLASSES: num_classes
        self.seed = seed
        self.check_point = check_point
        self.EPOCHES = epoches
        self.BATCH_SIZE = batch_size
        self.LR = lr
        self.NUM_WORKER = num_worker
        self.SAVE_PATH = save_path
        self.TB = tensorboard
        self.Comment = comment
        self.Databalance = databalance

    def start(self):
         
        random.seed(self.seed)
        np.random.seed(self.seed)
        manual_seed(42)
        cuda.manual_seed(self.seed)
        
        dataset_list = os.listdir(self.ROOT_PATH)
        
        train_datasets = []
        test_datasets = []
        train_dataloaders = []
        test_dataloaders = []
        local_models = []
        device = "cuda" if cuda.is_available() else "cpu"
        self.NET.load_state_dict(load(self.check_point)["model_state_dict"])
        for i, path in enumerate(dataset_list):  
            train_datasets.append(
                random_split(
                    FdDataset(
                        root_dir=os.path.join(self.ROOT_PATH, path, "mix"),
                        csv_file=os.path.join(self.ROOT_PATH, path, "train.csv"),
                        transform=transforms.Compose([
                            transforms.Resize((224, 224)),
                            transforms.RandomAffine(degrees=10, translate=(0.02, 0.02)),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            self.normalize[i],
                        ])
                    ), [1.0])[0]
            )
            test_datasets.append(
                random_split(
                    FdDataset(
                        root_dir=os.path.join(self.ROOT_PATH, path, "mix"),
                        csv_file=os.path.join(self.ROOT_PATH, path, "test.csv"),
                        transform=transforms.Compose([
                            transforms.Resize((224, 224)),
                            transforms.ToTensor(),
                            self.normalize[i],
                        ])
                    ), [1.0])[0]
            )
            train_dataloaders.append(DataLoader(train_datasets[i], self.BATCH_SIZE, True, num_workers=self.NUM_WORKER))
            test_dataloaders.append(DataLoader(test_datasets[i], self.BATCH_SIZE, False, num_workers=self.NUM_WORKER))
            local_models.append(copy.deepcopy(self.NET))
        criterion = nn.CrossEntropyLoss()
        optimizers = [optim.Adam(local_models[i].parameters(), lr=self.LR) for i, _ in enumerate(dataset_list)]
        schedulers = [optim.lr_scheduler.ReduceLROnPlateau(optimizer) for optimizer in optimizers]
        writers_list = []
        if self.TB:
            # 
            for dataset in dataset_list:
                writers_list.append(SummaryWriter(comment=f"_{self.NET.__class__.__name__}_{dataset}" +
                                                          (f"_{'_'.join(self.Comment)}" if self.Comment else "")))
            # 
            # writers_list.append(SummaryWriter(comment=f"_{self.NET.__class__.__name__}_eva" +
            #                                           (f"_{'_'.join(self.Comment)}" if self.Comment else "")))
        for epoch in range(self.EPOCHES):
            for i, dataset in enumerate(dataset_list): 
                print(f"dataset：{dataset}")
                if optimizers[i].param_groups[0].get("lr", 0) == 0:
                    print("lr:0")
                    break
                local_models[i].to(device)
                local_models[i].train()
                mean_loss = 0
                for images, labels in tqdm(train_dataloaders[i], f"[Epoch {epoch + 1}], train"):
                    images, labels = images.to(device), labels.to(device)
                    optimizers[i].zero_grad()
                    outputs = local_models[i](images)
                    loss = criterion(outputs, labels)
                    mean_loss += loss
                    loss.backward()
                    optimizers[i].step()
                del images, labels, outputs
                schedulers[i].step(loss.data.item())
                del loss
                mean_loss /= len(train_dataloaders[i])
                print(f"{dataset} mean_loss:{mean_loss}, lr:{optimizers[i].param_groups[0].get('lr', 0)}\n")
                if self.TB:
                    writers_list[i].add_scalar("info/train_mean_loss", mean_loss, epoch + 1)
        
            for j, dataset in enumerate(dataset_list):
                with torch.no_grad():
                    local_models[j].eval()
                    AUC, Sensitivity, Specificity, Accuracy, F1, PPV, ConfusionMatrix, mean_loss = test(
                        local_models[j].cpu(),
                        test_dataloaders[j], epoch)
                if self.TB:
                    # writers_list[j].add_scalar("info/eva_AUC", AUC, epoch + 1)
                    writers_list[j].add_scalar("info/eva_Sensitivity", Sensitivity, epoch + 1)
                    writers_list[j].add_scalar("info/eva_Specificity", Specificity, epoch + 1)
                    writers_list[j].add_scalar("info/eva_Accuracy", Accuracy, epoch + 1)
                    writers_list[j].add_scalar("info/eva_F1", F1, epoch + 1)
                    writers_list[j].add_scalar("info/eva_PPV", PPV, epoch + 1)
                    # writers_list[j].add_scalar("info/eva_NPV", NPV, epoch + 1)
                print(f"epoch{epoch + 1}_{dataset}_model "
                      f"{Sensitivity=:.6f}, {Specificity=:.6f}, "
                      f"{Accuracy=:.6f}, {F1=:.6f}, {PPV=:.6f}\n")
                # print(f"epoch{epoch + 1}_{dataset}_model "
                #       f"{AUC=:.6f}, {Sensitivity=:.6f}, {Specificity=:.6f}, "
                #       f"{Accuracy=:.6f}, {F1=:.6f}, {PPV=:.6f}\n")
                # print(f"epoch{epoch + 1}_{dataset}_model "
                #       f"AUC={AUC:.6f}, Sensitivity={Sensitivity:.6f}, Specificity={Specificity:.6f}, "
                #       f"Accuracy={Accuracy:.6f}, F1={F1:.6f}, PPV={PPV:.6f}, NPV={NPV:.6f}\n")
        
                if not os.path.exists(self.SAVE_PATH):
                    os.mkdir(self.SAVE_PATH)
                save({
                    'epoch': epoch + 1,
                    'model_state_dict': local_models[j].state_dict()},
                    os.path.join(self.SAVE_PATH,
                                 f"{datetime.now().strftime('%Y%m%d-%H%M%S')}_epoch{epoch + 1}"
                                 f"_{self.NET.__class__.__name__}_{dataset}" +
                                 (f"_{'_'.join(self.Comment)}.pth" if self.Comment else ".pth")))


if __name__ == '__main__':
    a = FedTrainer(os.path.join("data"),
                   [transforms.Normalize([0.74264, 0.55969, 0.74975], [0.0905, 0.16546, 0.12208]),
                    transforms.Normalize([0.73976, 0.59303, 0.75426], [0.08841, 0.15652, 0.11861])],
                   timm.create_model("densenet121", num_classes=2), 100, num_worker=0)
    a.start()

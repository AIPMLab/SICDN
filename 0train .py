import logging
import os
from typing import Optional
import torch
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
from Train import FdDataset
from fed1 import FdDataset, fed_weighted_average, test, test_acc
import timm
from torch.backends import mps
from torch import optim, nn, cuda
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, random_split, ConcatDataset
from torchvision import transforms
import models
import random
def get_device():
    if cuda.is_available():
        return torch.device("cuda")
    elif mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")
def train_model(
        root_path: str,
        model,
        normalize,
        epochs: int = 30,
        batch_size: int = 1,
        lr: float = 1e-3,
        seed: int = 156,
        load_path: str = "",
        save_checkpoint=False,
        save_interval: int = 10,
        save_path: str = "./check_point",
        tensorboard: bool = True,
        device=get_device(),
        amp: bool = False
):
    torch.manual_seed(seed)
    dataset_train = FdDataset(root_dir=os.path.join(root_path, "mix"),
                              csv_file=os.path.join(root_path, "train.csv"),
                              transform=transforms.Compose([
                                  transforms.Resize((224, 224)),
                                  transforms.RandomAffine(degrees=10, translate=(0.02, 0.02)),
                                  transforms.RandomHorizontalFlip(),
                                  transforms.ToTensor(),
                                  normalize,
                              ]))
    dataset_train = random_split(dataset_train, [1.0])[0]
    dataset_test = FdDataset(root_dir=os.path.join(root_path, "mix"),
                             csv_file=os.path.join(root_path, "test.csv"),
                             transform=transforms.Compose([
                                 transforms.Resize((224, 224)),
                                 transforms.ToTensor(),
                                 normalize,
                             ]))
    dataset_test = random_split(dataset_test, [1.0])[0]
    train_loader = DataLoader(dataset_train, batch_size, True)
    test_loader = DataLoader(dataset_test, batch_size, False)

    # 3. Create data loaders
    info = f'''
        Seed: {seed}, \tBatch size: {batch_size}, \tEpochs: {epochs}
        Learning rate: {lr}, \tImageDir: {root_path}
        Training size: {len(dataset_train)}, \tTest size: {len(dataset_test)}
        Device: {device.type}, \tSaveCheckpoints: {save_checkpoint}
    '''
    logging.info(info)
    if load_path:
        model.load_state_dict(torch.load(load_path))
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max')
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    criterion = nn.CrossEntropyLoss()
    if tensorboard:
        writer = SummaryWriter(comment=f"_{model.__class__.__name__}")
        writer.add_text("train_info", info)
    # 5. Begin training
    for epoch in range(1, epochs + 1):
        if optimizer.param_groups[0].get("lr", 0) == 0:
            print("lr=0")
            break
        model.train()
        mean_loss = 0
        for images, labels in tqdm(train_loader, f"[Epoch {epoch}], train"):
            images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            labels = labels.to(device=device, dtype=torch.long)
            with torch.autocast(device.type, enabled=amp):
                outputs = model(images)
                loss = criterion(outputs, labels)
                mean_loss += loss

            optimizer.zero_grad(set_to_none=True)
            grad_scaler.scale(loss).backward()
            grad_scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            grad_scaler.step(optimizer)
            grad_scaler.update()
            mean_loss += loss.item()
            # logging.info(f"loss:{loss.item()}")

        mean_loss /= len(train_loader)
        # scheduler.step(mean_loss)
        logging.info(f"mean_loss:{mean_loss.item()}, lr:{optimizer.param_groups[0].get('lr', 0)}")
        if tensorboard:
            writer.add_scalar("info/mean_loss", mean_loss, epoch)
            writer.add_scalar("info/lr", optimizer.param_groups[0].get("lr", 0), epoch)
        model.eval()
        with torch.no_grad():
            # AUC, Sensitivity, Specificity, Accuracy, F1, PPV, _, mean_loss = test(
            #     model, test_loader, epoch, "breast1", criterion)
            Accuracy, labels_total, predicted_total = test_acc(model, test_loader, epoch, "chest")
            model.to(device)
            # logging.info(f"epoch{epoch + 1}_breast1_model "
            #              f"{AUC=}, {Sensitivity=:.6f}, {Specificity=:.6f}, "
            #              f"{Accuracy=:.6f}, {F1=:.6f}, {PPV=:.6f}, {mean_loss=:.6f}\n")
            logging.info(f"epoch{epoch + 1}_che3_model {Accuracy=:.6f}, {labels_total=}, {predicted_total=}")
        scheduler.step(Accuracy)
        if tensorboard:
            #     writer.add_scalar("info/mean_loss", mean_loss, epoch)
            #     writer.add_scalar("info/eva_Sensitivity", Sensitivity, epoch)
            #     writer.add_scalar("info/eva_Specificity", Specificity, epoch)
            writer.add_scalar("info/eva_Accuracy", Accuracy, epoch)
        #     writer.add_scalar("info/eva_F1", F1, epoch)
        #     writer.add_scalar("info/eva_PPV", PPV, epoch)
        #     writer.add_scalar("info/lr", optimizer.param_groups[0].get("lr", 0), epoch)

        if save_checkpoint and epoch % save_interval == 0:
            Path(save_path).mkdir(parents=True, exist_ok=True)
            state_dict = model.state_dict()
            # state_dict['mask_values'] = dataset.mask_values
            torch.save(state_dict, os.path.join(save_path,
                                                f"{datetime.now().strftime('%Y%m%d-%H%M%S')}"
                                                f"_epoch{epoch}_{model.__class__.__name__}.pth"))
            logging.info(f'Checkpoint {epoch} saved!')

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    logging.info(f'Using device {get_device()}')

    model1 = timm.create_model("densenet121", num_classes=2)
    model2 = timm.create_model("resnet50", num_classes=2)
    model3 = timm.create_model("edgenext_base", num_classes=2)
    model4 = timm.create_model("convnext_base", num_classes=2)
    shap_model = models.shapDenseNet(timm.create_model("densenet121", num_classes=2))

    train_model("data/",model3,
                transforms.Normalize([0.48028, 0.48028, 0.48028],[0.22246, 0.22246, 0.22246]),
                epochs=100, batch_size=8,
                save_checkpoint=True, save_interval=1)

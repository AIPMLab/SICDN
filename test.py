                                                                                                                                                                                                                                                                                                                                                         import os
import torch
import logging
import timm
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from fed1 import FdDataset, test1
from torch.utils.data import DataLoader, random_split
writer = SummaryWriter()

def test_and_log(model, test_loader, global_step: int, writer: SummaryWriter, dataset_name: str):
    with torch.no_grad():
        Recall, Accuracy, F1, labels_total, predicted_total = test1(model, test_loader, dataset=dataset_name)
        AUC = test2(model, test_loader, dataset=dataset_name)
        logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
        logging.info(f"{dataset_name}_model_epoch{global_step} {Accuracy=:.6f}, {AUC=:.6f}, {Recall=:.6f},{F1=:.6f}")
        writer.add_scalar(f'{dataset_name}/Accuracy', Accuracy, global_step)
        writer.add_scalar(f'{dataset_name}/AUC', AUC, global_step)
        writer.add_scalar(f'{dataset_name}/Recall', Recall, global_step)
        writer.add_scalar(f'{dataset_name}/F1', F1, global_step)

if __name__ == "__main__":
    checkpoints_dir = "XXXX"

    model = timm.create_model("resnet50", num_classes=2)
    # model = timm.create_model("edgenext_base", num_classes=2)
    #model = timm.create_model("convnext_base", num_classes=2)
    # model = timm.create_model("convnextv2_base", num_classes=2)
    checkpoint_files = [os.path.join(checkpoints_dir, file) for file in os.listdir(checkpoints_dir) if file.endswith('.pth')]
    dataset_root = "data/"
    normalize = transforms.Normalize([0.47449, 0.47449, 0.47449],[0.22669, 0.22669, 0.22669])
    batch_size = 8
    dataset_test = FdDataset(root_dir=os.path.join(dataset_root, "mix"),
                             csv_file=os.path.join(dataset_root, "test.csv"),
                             transform=transforms.Compose([
                                 transforms.Resize((224, 224)),
                                 transforms.ToTensor(),
                                 normalize,
                             ]))
    dataset_test = random_split(dataset_test, [1.0])[0]
    test_loader = DataLoader(dataset_test, batch_size, False)

    global_step = 0

    for i, checkpoint_path in enumerate(checkpoint_files):
        model.load_state_dict(torch.load(checkpoint_path))
        model.eval()
        test_and_log(model, test_loader, global_step, writer, "XXX")
        global_step += 1
    writer.close()                                                                                                                                                                                            

import torch
import torch.nn as nn
from torchnet import meter
from torchvision import transforms
import numpy as np
import random
from torch.backends import cudnn
from torchinfo import summary
import time
from tqdm import tqdm
from utils.config import config
from data import get_datasets, get_dataloaders
from models.ResNet_GFN import ResNetGFN

import argparse

parser = argparse.ArgumentParser()

parser.add_argument("-tr", "--trainpath", help="Train data path")
parser.add_argument("-te", "--testpath", help="Test data path")
parser.add_argument("-m", "--mask", help="Mask (can be none, topdown, bottomup)")

args = parser.parse_args()
train_data_path, val_data_path = args.trainpath, args.testpath
config.mask = args.mask or config.mask

print(f"Train data path: {train_data_path}, test data path: {val_data_path}")


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Devide being used is: {device}. Mask being used is: {config.mask}")
cudnn.benchmark = True

current_time = time.strftime("%Y%m%d%H%M%S", time.localtime())
print(f"Current time is: {current_time}")

img_transforms = transforms.Compose(
    [transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)), transforms.ToTensor()]
)

train_dataset, test_dataset = get_datasets(
    train_data_path, val_data_path, img_transforms
)
train_loader, test_loader = get_dataloaders(
    train_dataset, test_dataset, batch_size=config.BATCH_SIZE, shuffle=True
)


def train(**kwargs):
    if config.seed is not None:
        setup_seed(config.seed)
    best_val_loss = 1e9
    model = ResNetGFN()
    model = model.to(device)
    input_, target = next(iter(train_loader))
    summary(model, input_data=[input_, target])
    model.train()
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {pytorch_total_params}")

    def criterion(output, target_var):
        loss = nn.CrossEntropyLoss().to(device)(output, target_var)
        if config.gfn_dropout:
            reg_loss = model.module.regularization()
            return (loss + reg_loss).to(device)
        else:
            return loss

    if config.optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(), config.lr)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=config.schedule_milestone, gamma=config.lr_decay
        )
    elif config.optimizer == "momentum":
        optimizer = torch.optim.SGD(
            model.parameters(), config.lr, momentum=config.MOMENTUM, nesterov=True
        )
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=config.schedule_milestone, gamma=config.lr_decay
        )
    else:
        print("No optimizer")
        return

    loss_meter = meter.AverageValueMeter()
    if config.gfn_dropout:
        accuracy_meter = meter.AverageValueMeter()
        GFNloss_unconditional_meter = meter.AverageValueMeter()
        LogZ_unconditional_meter = meter.AverageValueMeter()
        LogPF_qz_meter = meter.AverageValueMeter()
        GFNloss_conditional_meter = meter.AverageValueMeter()
        LogZ_conditional_meter = meter.AverageValueMeter()
        LogPF_qzxy_meter = meter.AverageValueMeter()
        LogPF_BNN_meter = meter.AverageValueMeter()
        actual_dropout_rate_meter = meter.AverageValueMeter()
        COR_qz_meter = meter.AverageValueMeter()
        COR_qzxy_meter = meter.AverageValueMeter()
        Log_pz_meter = meter.AverageValueMeter()
        Log_pzx_meter = meter.AverageValueMeter()
    else:
        accuracy_meter = meter.ClassErrorMeter(accuracy=True)

    for epoch in tqdm(range(config.N_EPOCHS)):
        model.train()
        loss_meter.reset()
        accuracy_meter.reset()

        for ii, (input_, target) in enumerate(train_loader):
            print(f"Batch: {ii}")

            input_, target = input_.to(device), target.to(device)
            optimizer.zero_grad()

            if config.gfn_dropout == True:
                metric = model._gfn_step(
                    input_, target, mask_train="", mask=config.mask
                )

                loss = metric["CELoss"]
                acc = metric["acc"]

                loss_meter.add(loss)
                accuracy_meter.add(acc)
                GFNloss_unconditional_meter.add(metric["GFN_loss_unconditional"])
                LogZ_unconditional_meter.add(metric["LogZ_unconditional"])
                LogPF_qz_meter.add(metric["LogPF_qz"])
                GFNloss_conditional_meter.add(metric["GFN_loss_conditional"])
                LogZ_conditional_meter.add(metric["LogZ_conditional"])
                LogPF_qzxy_meter.add(metric["LogPF_qzxy"])
                LogPF_BNN_meter.add(metric["LogPF_BNN"])
                actual_dropout_rate_meter.add(metric["actual_dropout_rate"])
                COR_qz_meter.add(metric["COR_qz"])
                COR_qzxy_meter.add(metric["COR_qzxy"])
                Log_pz_meter.add(metric["Log_pz"])
                Log_pzx_meter.add(metric["Log_pzx"])

            if config.gfn_dropout == False:
                score = model(input_, target, "none")
                loss = criterion(score, target)
                loss.backward()

                optimizer.step()
                loss_meter.add(loss.data)
                accuracy_meter.add(score.data, target.data)

            if config.gfn_dropout:
                print(
                    "epoch:{epoch},lr:{lr},loss:{loss:.2f},train_acc:{train_acc:.2f} GFN_loss_conditional:{GFN_loss_conditional} GFN_loss_unconditional:{GFN_loss_unconditional} actual_dropout_rate:{actual_dropout_rate}".format(
                        epoch=epoch,
                        loss=loss_meter.value()[0],
                        train_acc=accuracy_meter.value()[0],
                        lr=optimizer.param_groups[0]["lr"],
                        GFN_loss_conditional=metric["GFN_loss_conditional"],
                        GFN_loss_unconditional=metric["GFN_loss_unconditional"],
                        actual_dropout_rate=metric["actual_dropout_rate"],
                    )
                )
            else:
                print(
                    "epoch:{epoch},lr:{lr},loss:{loss:.2f},train_acc:{train_acc:.2f}".format(
                        epoch=epoch,
                        loss=loss_meter.value()[0],
                        train_acc=accuracy_meter.value()[0],
                        lr=optimizer.param_groups[0]["lr"],
                    )
                )


print(f"Starting training function")
print("-" * 30)
train()

from ResnetNetwork import *
import torch.optim as optim
import torch.utils.data as data
import torch.nn as nn
from torch.autograd import Variable
from CenterLoss import *
import numpy as np
import os
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import math
from model import *
import dataloader
from scipy.io import loadmat
import wfdb
from EarlyStopping import *
from SignalPreprocessing.data_preprocess_function import *

REAL_DATASET = os.path.join(os.path.dirname(os.path.realpath(__file__)), "NewReal")

LOSSES = os.path.join(os.path.dirname(os.path.realpath(__file__)), "Losses")
if not os.path.exists(LOSSES):
    os.mkdir(LOSSES)

BATCH_SIZE = 32
epochs = 20
learning_rate_real =  1e-3

def main():

    pl.seed_everything(1234)
    real_dataset = dataloader.RealDataset(REAL_DATASET)
    train_size_real = int(0.6 * len(real_dataset))
    val_size_real = int(0.2 * len(real_dataset))
    test_size_real = int(0.2 * len(real_dataset))

    train_dataset_real, val_dataset_real, test_dataset_real = torch.utils.data.random_split(real_dataset, [train_size_real, val_size_real,test_size_real])

    train_data_loader_real = data.DataLoader(train_dataset_real, batch_size=BATCH_SIZE, shuffle=True, num_workers=12)
    val_data_loader_real = data.DataLoader(val_dataset_real, batch_size=BATCH_SIZE, shuffle=False)
    test_data_loader_real = data.DataLoader(test_dataset_real, batch_size=BATCH_SIZE, shuffle=False)

    #  use gpu if available
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    print(device)

    resnet_model = ResNet(1).cuda()
    best_model_accuracy_real = - math.inf
    val_loss_real = 0
    early_stopping_real = EarlyStopping(delta_min=0.01, patience=6, verbose=True)
    criterion = nn.L1Loss().cuda()
    criterion_cent = CenterLoss(num_classes=2, feat_dim=512*64, use_gpu=device)
    params = list(resnet_model.parameters()) + list(criterion_cent.parameters())
    optimizer_model_real = optim.SGD(params, lr=learning_rate_real, momentum=0.9, weight_decay=1e-4)
    scheduler_real = torch.optim.lr_scheduler.StepLR(optimizer_model_real, 6, gamma=0.1)

    train_loss_ecg_list = []
    validation_loss_ecg_list = []
    validation_corr_ecg_list = []

    resnet_model.load_state_dict(torch.load(str(network_save_folder_orig + network_file_name_best_sim)))
    criterion_cent.load_state_dict(torch.load(str(network_save_folder_orig + network_file_name_best_cent)))

    for epoch in range(epochs):
        # Train Real
        resnet_model.train()
        criterion_cent.train()
        train_real(resnet_model,
           train_data_loader_real,
           optimizer_model_real,
           epoch,
           epochs,
           criterion,
           criterion_cent,
           train_loss_ecg_list)
        # Validation Real
        resnet_model.eval()
        criterion_cent.eval()
        best_model_accuracy_real, val_loss_real = val_real(
           val_data_loader_real,
           resnet_model,
           criterion,
           epoch,
           epochs,
           criterion_cent,
           validation_loss_ecg_list,
           validation_corr_ecg_list,
           best_model_accuracy_real)
        scheduler_real.step()
        early_stopping_real(val_loss_real.cpu().detach().numpy(), resnet_model)
        if early_stopping_real.early_stop:
            print('Early stopping')
            break

    #Saving graphs training
    path_losses = os.path.join(LOSSES, "TL1ECG")
    np.save(path_losses, np.array(train_loss_ecg_list))

    #Saving graphs validation
    path_losses = os.path.join(LOSSES, "VL1ECG")
    np.save(path_losses, np.array(validation_loss_ecg_list))

    path_losses = os.path.join(LOSSES, "CorrECG")
    np.save(path_losses, np.array(validation_corr_ecg_list))


    #Test
    test_loss_ecg,test_corr_ecg = test(str(network_save_folder_orig + network_file_name_best_real),test_data_loader_real)

    with open("test_loss.txt", 'w') as f:
        f.write(",test_loss_ecg = {:.4f},test_corr_ecg = {:.4f}".format(test_loss_ecg,test_corr_ecg))
    del resnet_model
    del train_data_loader_real
    torch.cuda.empty_cache()


if __name__ == "__main__":

    main()


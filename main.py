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
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
import math
from model import *
import dataloader
from scipy.io import loadmat
from EarlyStopping import *

SIMULATED_DATASET = os.path.join(os.path.dirname(os.path.realpath(__file__)), "SimulatedDatabase")
LOSSES = os.path.join(os.path.dirname(os.path.realpath(__file__)), "Losses")
if not os.path.exists(LOSSES):
    os.mkdir(LOSSES)

network_save_folder_orig = "./Models"
network_file_name_last = "/last_model"
network_file_name_best = "./best_model"

BATCH_SIZE = 32
epochs = 150
learning_rate = 1e-3


def main(filename):


    pl.seed_everything(1234)
    list_simulated = simulated_database_list(SIMULATED_DATASET)

    list_simulated_overfit = list_simulated[:2040]  # TODO: put in comment after validating

    #remove_nan_signals(list_simulated_overfit) # TODO: change to original list

    simulated_dataset = dataloader.SimulatedDataset(SIMULATED_DATASET,list_simulated_overfit) # TODO: change to original list size after validating

    train_size_sim = int(0.6 * len(simulated_dataset))
    val_size_sim = int(0.2 * len(simulated_dataset))
    test_size_sim = int(0.2 * len(simulated_dataset))

    train_dataset_sim, val_dataset_sim, test_dataset_sim = torch.utils.data.random_split(simulated_dataset, [train_size_sim, val_size_sim,test_size_sim])

    train_data_loader_sim = data.DataLoader(train_dataset_sim, batch_size=BATCH_SIZE, shuffle=True, num_workers=12)
    val_data_loader_sim = data.DataLoader(val_dataset_sim, batch_size=BATCH_SIZE, shuffle=False)
    test_data_loader_sim = data.DataLoader(test_dataset_sim, batch_size=BATCH_SIZE, shuffle=False)

    #  use gpu if available
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    print(device)

    resnet_model = ResNet(1).cuda()
    resnet_model.load_state_dict(torch.load(filename))
    best_model_accuracy = math.inf
    val_loss = 0
    #early_stopping = EarlyStopping(delta_min=0.1, patience= 10, verbose=True)
    criterion = nn.L1Loss().cuda()
    criterion_cent = CenterLoss(num_classes=2, feat_dim=512*64, use_gpu=device)
    params = list(resnet_model.parameters()) + list(criterion_cent.parameters())
    optimizer_model = optim.SGD(params, lr=learning_rate, momentum=0.9,weight_decay=1e-4)
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer_model, step_size=6, gamma=0.1)

    #optimizer_centloss = optim.Adam(criterion_cent.parameters(), lr=learning_rate,amsgrad= True)

    train_loss_f_list = []
    train_loss_m_list = []
    train_loss_average_list = []
    validation_loss_f_list = []
    validation_loss_m_list = []
    validation_loss_average_list = []
    validation_corr_m_list = []
    validation_corr_f_list = []

    for epoch in range(epochs):
        #Train
        resnet_model.train()
        train(resnet_model,
              train_data_loader_sim,
              optimizer_model,
              criterion,
              criterion_cent,
              epoch,
              epochs,
              train_loss_f_list,
              train_loss_m_list,
              train_loss_average_list)
        #Evaluation
        resnet_model.eval()
        best_model_accuracy, val_loss = val(val_data_loader_sim,
                                    resnet_model,
                                    criterion,
                                    epoch,
                                    epochs,
                                    validation_loss_m_list,
                                    validation_loss_f_list,
                                    validation_loss_average_list,
                                    validation_corr_m_list,
                                    validation_corr_f_list,
                                    best_model_accuracy)
        #scheduler.step()

        """early_stopping(val_loss, resnet_model)

        if early_stopping.early_stop:
            print('Early stopping')
            break"""

    #Saving graphs training
    path_losses = os.path.join(LOSSES, "TL1M")
    np.save(path_losses, np.array(train_loss_m_list))
    path_losses = os.path.join(LOSSES, "TL1F")
    np.save(path_losses, np.array(train_loss_f_list))
    path_losses = os.path.join(LOSSES, "TL1Avg")
    np.save(path_losses, np.array(train_loss_average_list))

    #Saving graphs validation
    path_losses = os.path.join(LOSSES, "VL1M")
    np.save(path_losses, np.array(validation_loss_m_list))
    path_losses = os.path.join(LOSSES, "VL1F")
    np.save(path_losses, np.array(validation_loss_f_list))
    path_losses = os.path.join(LOSSES, "VL1Avg")
    np.save(path_losses, np.array(validation_loss_average_list))

    path_losses = os.path.join(LOSSES, "CorrM")
    np.save(path_losses, np.array(validation_corr_m_list))
    path_losses = os.path.join(LOSSES, "CorrF")
    np.save(path_losses, np.array(validation_corr_f_list))

    #Test
    test_loss_m, test_loss_f, test_loss_avg, test_corr_m, test_corr_f, test_corr_average = test(str(network_save_folder_orig  + network_file_name_best),test_data_loader_sim)

    with open("test_loss.txt", 'w') as f:
        f.write("test_loss_m = {:.4f},test_loss_f = {:.4f},test_loss_avg = {:.4f},"
                "test_corr_m = {:.4f},test_corr_f = {:.4f},test_corr_avg = {:.4f}\n".format(test_loss_m, test_loss_f, test_loss_avg,
                                                                                            test_corr_m,test_corr_f,test_corr_average))
    del resnet_model
    del simulated_dataset
    del train_data_loader_sim
    torch.cuda.empty_cache()


if __name__=="__main__":
    main(str(network_file_name_best))


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
import scipy.fftpack as function
from SignalPreprocessing.data_preprocess_function import *

SIMULATED_DATASET = os.path.join(os.path.dirname(os.path.realpath(__file__)), "SimulatedDatabase")

LOSSES = os.path.join(os.path.dirname(os.path.realpath(__file__)), "Losses")
if not os.path.exists(LOSSES):
    os.mkdir(LOSSES)

BAR_LIST_TEST = os.path.join(os.path.dirname(os.path.realpath(__file__)), "BarListTest")
if not os.path.exists(BAR_LIST_TEST):
    os.mkdir(BAR_LIST_TEST)

BATCH_SIZE = 32
epochs = 20
learning_rate = 1e-3

def main():

    pl.seed_everything(1234)
    list_simulated = simulated_database_list(SIMULATED_DATASET)[:10]

    #list_simulated = remove_nan_signals(list_simulated)

    simulated_dataset = dataloader.SimulatedDataset(SIMULATED_DATASET,list_simulated)

    train_size_sim = int(0.6 * len(simulated_dataset))
    val_size_sim = int(0.2 * len(simulated_dataset))
    test_size_sim = int(0.2 * len(simulated_dataset))

    train_dataset_sim, val_dataset_sim, test_dataset_sim = torch.utils.data.random_split(simulated_dataset, [train_size_sim, val_size_sim,test_size_sim])

    train_data_loader_sim = data.DataLoader(train_dataset_sim, batch_size=BATCH_SIZE, shuffle=True, num_workers=12)

    #  use gpu if available
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    print(device)

    resnet_model = ResNet(1).cuda()
    best_model_accuracy = - math.inf
    val_loss = 0
    criterion = nn.L1Loss().cuda()
    criterion_cent = CenterLoss(num_classes=2, feat_dim=512*64, use_gpu=device)
    params = list(resnet_model.parameters()) + list(criterion_cent.parameters())
    optimizer_model = optim.SGD(params, lr=learning_rate, momentum=0.9,weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer_model, milestones=[6, 12, 18], gamma=0.1)

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
        best_model_accuracy, val_loss = val(train_data_loader_sim,
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
        scheduler.step()

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
    test_loss_m, test_loss_f, test_loss_avg, test_corr_m, test_corr_f, test_corr_average,\
        list_bar_good_example_noisetype, list_bar_bad_example_noisetype,\
        list_bar_good_example_snr,list_bar_bad_example_snr, \
        list_bar_good_example_snrcase, list_bar_bad_example_snrcase = test(str(network_save_folder_orig + network_file_name_best),test_data_loader_sim)

    path_bar = os.path.join(BAR_LIST_TEST, "list_bar_good_example_noisetype")
    np.save(path_bar, np.array(list_bar_good_example_noisetype))
    path_bar = os.path.join(BAR_LIST_TEST, "list_bar_bad_example_noisetype")
    np.save(path_bar, np.array(list_bar_bad_example_noisetype))
    path_bar = os.path.join(BAR_LIST_TEST, "list_bar_good_example_snr")
    np.save(path_bar, np.array(list_bar_good_example_snr))
    path_bar = os.path.join(BAR_LIST_TEST, "list_bar_bad_example_snr")
    np.save(path_bar, np.array(list_bar_bad_example_snr))
    path_bar = os.path.join(BAR_LIST_TEST, "list_bar_good_example_snrcase")
    np.save(path_bar, np.array(list_bar_good_example_snrcase))
    path_bar = os.path.join(BAR_LIST_TEST, "list_bar_bad_example_snrcase")
    np.save(path_bar, np.array(list_bar_bad_example_snrcase))

    with open("test_loss.txt", 'w') as f:
        f.write("test_loss_m = {:.4f},test_loss_f = {:.4f},test_loss_avg = {:.4f},"
                "test_corr_m = {:.4f},test_corr_f = {:.4f},test_corr_avg = {:.4f}\n".format(test_loss_m, test_loss_f, test_loss_avg,
                                                                                            test_corr_m,test_corr_f,test_corr_average))
    del resnet_model
    del simulated_dataset
    del train_data_loader_sim
    torch.cuda.empty_cache()


if __name__=="__main__":

    main()
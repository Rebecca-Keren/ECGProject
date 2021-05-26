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


SIMULATED_DATASET = os.path.join(os.path.dirname(os.path.realpath(__file__)), "SimulatedDatabase")
REAL_DATASET = os.path.join(os.path.dirname(os.path.realpath(__file__)), "real_windows")
#SIMULATED_DATASET = os.path.join(os.path.dirname(os.path.realpath(__file__)), "SimulatedDatabase")

LOSSES = os.path.join(os.path.dirname(os.path.realpath(__file__)), "Losses")
if not os.path.exists(LOSSES):
    os.mkdir(LOSSES)

BAR_LIST_TEST = os.path.join(os.path.dirname(os.path.realpath(__file__)), "BarListTest")
if not os.path.exists(BAR_LIST_TEST):
    os.mkdir(BAR_LIST_TEST)

BATCH_SIZE = 32
epochs = 25
learning_rate_sim = 1e-3
learning_rate_real = 1e-5

def main():

    pl.seed_everything(1234)
    list_simulated = simulated_database_list(SIMULATED_DATASET)[:10] #[:122740]
    list_simulated = remove_nan_signals(list_simulated,SIMULATED_DATASET)

    simulated_dataset = dataloader.SimulatedDataset(SIMULATED_DATASET,list_simulated)
    real_dataset = dataloader.RealDataset(REAL_DATASET)

    train_size_sim = int(0.6 * len(simulated_dataset))
    val_size_sim = int(0.2 * len(simulated_dataset))
    test_size_sim = int(0.2 * len(simulated_dataset))

    train_size_real = int(0.6 * len(real_dataset))
    val_size_real = int(0.2 * len(real_dataset))
    test_size_real = int(0.2 * len(real_dataset))

    train_dataset_sim, val_dataset_sim, test_dataset_sim = torch.utils.data.random_split(simulated_dataset, [train_size_sim, val_size_sim,test_size_sim])
    train_dataset_real, val_dataset_real, test_dataset_real = torch.utils.data.random_split(real_dataset, [train_size_real, val_size_real,test_size_real])

    train_data_loader_sim = data.DataLoader(train_dataset_sim, batch_size=BATCH_SIZE, shuffle=True, num_workers=12)
    val_data_loader_sim = data.DataLoader(val_dataset_sim, batch_size=BATCH_SIZE, shuffle=False)
    test_data_loader_sim = data.DataLoader(test_dataset_sim, batch_size=BATCH_SIZE, shuffle=False)

    train_data_loader_real = data.DataLoader(train_dataset_real, batch_size=BATCH_SIZE, shuffle=True, num_workers=12)
    val_data_loader_real = data.DataLoader(val_dataset_real, batch_size=BATCH_SIZE, shuffle=False)
    test_data_loader_real = data.DataLoader(test_dataset_real, batch_size=BATCH_SIZE, shuffle=False)

    #  use gpu if available
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    print(device)

    resnet_model = ResNet(1).cuda()
    best_model_accuracy_real = - math.inf
    best_model_accuracy_sim = - math.inf
    val_loss_sim = 0
    val_loss_real = 0
    early_stopping_sim = EarlyStopping(delta_min=0.01, patience=6, verbose=True)
    early_stopping_real = EarlyStopping(delta_min=0.01, patience=6, verbose=True)
    criterion = nn.L1Loss().cuda()
    criterion_cent = CenterLoss(num_classes=2, feat_dim=512*64, use_gpu=device)
    params = list(resnet_model.parameters()) + list(criterion_cent.parameters())
    optimizer_model_sim = optim.SGD(params, lr=learning_rate_sim, momentum=0.9,weight_decay=1e-4)
    scheduler_sim = torch.optim.lr_scheduler.MultiStepLR(optimizer_model_sim, milestones=[4, 6, 8, 12], gamma=0.1)
    optimizer_model_real = optim.SGD(params, lr=learning_rate_real, momentum=0.9, weight_decay=1e-4)
    scheduler_real = torch.optim.lr_scheduler.StepLR(optimizer_model_real, 4, gamma=0.1)

    train_loss_f_list = []
    train_loss_m_list = []
    train_loss_ecg_list = []
    train_loss_average_list = []
    validation_loss_f_list = []
    validation_loss_m_list = []
    validation_loss_average_list = []
    validation_corr_m_list = []
    validation_corr_f_list = []
    validation_loss_ecg_list = []

    for epoch in range(epochs):
        #Train Sim
        resnet_model.train()
        train_sim(resnet_model,
                  train_data_loader_sim,
                  optimizer_model_sim,
                  criterion,
                  criterion_cent,
                  epoch,
                  epochs,
                  train_loss_f_list,
                  train_loss_m_list,
                  train_loss_average_list)
        #Validation Sim
        resnet_model.eval()
        best_model_accuracy_sim,val_loss_sim =val_sim(val_data_loader_sim,
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
        scheduler_sim.step()
        early_stopping_sim(val_loss_sim, resnet_model)
        if early_stopping_sim.early_stop:
            print('Early stopping')
            break

        # Train Real
        resnet_model.train()
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
        best_model_accuracy_real, val_loss_real = val_train(
            val_data_loader_real,
            resnet_model,
            criterion,
            epoch,
            epochs,
            validation_loss_ecg_list,
            best_model_accuracy)

        early_stopping_real(val_loss_real, resnet_model)
        if early_stopping_real.early_stop:
            print('Early stopping')
            break

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
    test_loss_m, test_loss_f, test_loss_ecg,test_loss_avg, test_corr_m, test_corr_f, test_corr_average,\
        list_bar_good_example_noisetype, list_bar_bad_example_noisetype,\
        list_bar_good_example_snr,list_bar_bad_example_snr, \
        list_bar_good_example_snrcase, list_bar_bad_example_snrcase = test(str(network_save_folder_orig + network_file_name_best),test_data_loader_sim,test_data_loader_real)

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
        f.write("test_loss_m = {:.4f},test_loss_f = {:.4f},test_loss_ecg = {:.4f},test_loss_avg = {:.4f},"
                "test_corr_m = {:.4f},test_corr_f = {:.4f},test_corr_avg = {:.4f}\n".format(test_loss_m, test_loss_f, test_loss_ecg,test_loss_avg,
                                                                                            test_corr_m,test_corr_f,test_corr_average))
    del resnet_model
    del simulated_dataset
    del train_data_loader_sim
    torch.cuda.empty_cache()


if __name__=="__main__":

    correlation_f = 0
    correlation_m = 0
    num_of_f = 0
    num_of_m = 0

    #main()

    BAR_LIST = os.path.join(os.path.dirname(os.path.realpath(__file__)), "BarListTest")


    #BAR REPRESENTATION
    ind = np.arange(4)
    x_labels = ['NONE', 'MA', 'MA+EM', 'MA+EM+BW']
    results = np.load(os.path.join(BAR_LIST,"list_bar_bad_example_noisetype.npy"))
    sum = np.sum(np.matrix(results))
    plt.bar(ind,results)
    plt.title('Failing signals according to noise type. Total: {}'.format(sum))
    plt.xticks(ind,('NONE', 'MA', 'MA+EM', 'MA+EM+BW'))
    plt.show()
    plt.close()


    ind = np.arange(4)
    x_labels = ['NONE', 'MA', 'MA+EM', 'MA+EM+BW']
    results = np.load(os.path.join(BAR_LIST,"list_bar_good_example_noisetype.npy"))
    sum = np.sum(np.matrix(results))
    plt.bar(ind,results)
    plt.title('Successful signals according to noise type. Total: {}'.format(sum))
    plt.xticks(ind,('NONE', 'MA', 'MA+EM', 'MA+EM+BW'))
    plt.show()
    plt.close()

    ind = np.arange(5)
    x_labels = ['00', '03', '06', '09', '12']
    results = np.load(os.path.join(BAR_LIST,"list_bar_bad_example_snr.npy"))
    sum = np.sum(np.matrix(results))
    plt.bar(ind,results)
    plt.title('Failing signals according to SNR [dB]. Total: {}'.format(sum))
    plt.xticks(ind,('00', '03', '06', '09', '12'))
    plt.show()
    plt.close()

    ind = np.arange(5)
    x_labels = ['00', '03', '06', '09', '12']
    results = np.load(os.path.join(BAR_LIST, "list_bar_good_example_snr.npy"))
    sum = np.sum(np.matrix(results))
    plt.bar(ind, results)
    plt.title('Successful signals according to SNR [dB]. Total: {}'.format(sum))
    plt.xticks(ind, ('00', '03', '06', '09', '12'))
    plt.show()
    plt.close()


    X = np.arange(7)
    data = np.load(os.path.join(BAR_LIST, "list_bar_bad_example_snrcase.npy"))
    print(np.sum(data, axis=0))
    sum = np.sum(np.matrix(data))
    a = plt.bar(X, data[0], color='b', width=0.1)
    b = plt.bar(X + 0.1 , data[1], color='g', width=0.1)
    c = plt.bar(X + 0.2, data[2], color='r', width=0.1)
    d = plt.bar(X + 0.3, data[3], color='c', width=0.1)
    e = plt.bar(X + 0.4, data[4], color='y', width=0.1)
    plt.legend((a, b, c, d, e), ('00', '03', '06', '09', '12'))
    plt.xticks(X, ('CO', 'C1', 'C2', 'C3','C4', 'C5', 'BASELINE'))
    plt.title('Failing signals according to physiological case and SNR [dB]. Total: {}'.format(sum))
    plt.show()
    plt.close()
   
    X = np.arange(7)
    data = np.load(os.path.join(BAR_LIST, "list_bar_good_example_snrcase.npy"))
    print(np.sum(data,axis=0))
    sum = np.sum(np.matrix(data))
    a = plt.bar(X, data[0], color='b', width=0.1)
    b = plt.bar(X + 0.1, data[1], color='g', width=0.1)
    c = plt.bar(X + 0.2, data[2], color='r', width=0.1)
    d = plt.bar(X + 0.3, data[3], color='c', width=0.1)
    e = plt.bar(X + 0.4, data[4], color='y', width=0.1)
    plt.legend((a, b, c, d, e), ('00', '03', '06', '09', '12'))
    plt.xticks(X, ('CO', 'C1', 'C2', 'C3', 'C4', 'C5', 'BASELINE'))
    plt.title('Successful signals according to physiological case and SNR [dB]. Total: {}'.format(sum))
    plt.show()
    plt.close()
    
    
    #DROPOUT1

    """LOSSESBASE = os.path.join(os.path.dirname(os.path.realpath(__file__)), "Losses" + str(127740))
    LOSSESLDROP = os.path.join(os.path.dirname(os.path.realpath(__file__)), "Losses")
    path_losses = os.path.join(LOSSESBASE, "VL1M.npy")
    m1 = np.load(path_losses)[:20]
    path_losses = os.path.join(LOSSESBASE, "VL1F.npy")
    f1 = np.load(path_losses)[:20]
    path_losses = os.path.join(LOSSESBASE, "VL1Avg.npy")
    a1 = np.load(path_losses)[:20]
    path_losses = os.path.join(LOSSESLDROP, "VL1M.npy")
    m2 = np.load(path_losses)[:20]
    path_losses = os.path.join(LOSSESLDROP, "VL1F.npy")
    f2 = np.load(path_losses)[:20]
    path_losses = os.path.join(LOSSESLDROP, "VL1Avg.npy")
    a2 = np.load(path_losses)[:20]
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
    ax1.plot(m1, label="noDROP")
    ax1.plot(m2, label="DROP")
    ax1.set_ylabel("L1 M")
    ax1.set_xlabel("Epoch")
    ax2.plot(f1, label="noDROP")
    ax2.plot(f2, label="DROP")
    ax2.set_ylabel("L1 F")
    ax2.set_xlabel("Epoch")
    ax3.plot(a1, label="noDROP")
    ax3.plot(a2, label="DROP")
    ax3.set_ylabel("L1 Avg")
    ax3.set_xlabel("Epoch")
    ax1.legend()
    ax2.legend()
    ax3.legend()
    plt.show()
    plt.close()
    path_losses = os.path.join(LOSSESBASE, "TL1M.npy")
    m1 = np.load(path_losses)[:20]
    path_losses = os.path.join(LOSSESBASE, "TL1F.npy")
    f1 = np.load(path_losses)[:20]
    path_losses = os.path.join(LOSSESBASE, "TL1Avg.npy")
    a1 = np.load(path_losses)[:20]
    path_losses = os.path.join(LOSSESLDROP, "TL1M.npy")
    m2 = np.load(path_losses)[:20]
    path_losses = os.path.join(LOSSESLDROP, "TL1F.npy")
    f2 = np.load(path_losses)[:20]
    path_losses = os.path.join(LOSSESLDROP, "TL1Avg.npy")
    a2 = np.load(path_losses)[:20]
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
    ax1.plot(m1, label="noDROP")
    ax1.plot(m2, label="DROP")
    ax1.set_ylabel("L1 M")
    ax1.set_xlabel("Epoch")
    ax2.plot(f1, label="noDROP")
    ax2.plot(f2, label="DROP")
    ax2.set_ylabel("L1 F")
    ax2.set_xlabel("Epoch")
    ax3.plot(a1, label="noDROP")
    ax3.plot(a2, label="DROP")
    ax3.set_ylabel("L1 Avg")
    ax3.set_xlabel("Epoch")
    ax1.legend()
    ax2.legend()
    ax3.legend()
    plt.show()
    plt.close()"""


    #BAR REPRESENTATION
    """ind = np.arange(4)
    x_labels = ['NONE', 'MA', 'MA+EM', 'MA+EM+BW']
    students = np.load(os.path.join(BAR_LIST,"list_bar_bad_example.npy"))
    plt.bar(ind,students)
    plt.xticks(ind,('NONE', 'MA', 'MA+EM', 'MA+EM+BW'))
    plt.show()"""

    """ECG_OUTPUTS_TEST = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                    "ECGOutputsTest")
    LOSSES = os.path.join(os.path.dirname(os.path.realpath(__file__)), "Losses")


    for filename in os.listdir(ECG_OUTPUTS_TEST): #present the fecg outputs
        if "fecg" in filename:
            num_of_f += 1
            path = os.path.join(ECG_OUTPUTS_TEST, filename)
            number_file = filename.index("g") + 1
            end_path = filename[number_file:]
            path_label = os.path.join(ECG_OUTPUTS_TEST,"label_f" + end_path)
            real = np.load(path)
            label = np.load(path_label)
            correlation = check_correlation(real, label)
            if(correlation < 0.70):
                correlation_f += 1
            fig, (ax1, ax2) = plt.subplots(2, 1)
            ax1.plot(real)
            ax1.set_ylabel("FECG")
            ax2.plot(label)
            ax2.set_ylabel("LABEL")
            plt.show()
            plt.close()


        if "mecg" in filename:
            num_of_m += 1
            path = os.path.join(ECG_OUTPUTS_TEST, filename)
            number_file = filename.index("g") + 1
            end_path = filename[number_file:]
            path_label = os.path.join(ECG_OUTPUTS_TEST, "label_m" + end_path)
            real = np.load(path)
            label = np.load(path_label)
            correlation = check_correlation(real, label)
            if (correlation < 0.70):
                correlation_m += 1
            fig, (ax1, ax2) = plt.subplots(2, 1)
            ax1.plot(np.load(path))
            ax1.set_ylabel("MECG")
            ax2.plot(np.load(path_label))
            ax2.set_ylabel("LABEL")
            #plt.show()
            plt.close()

    print(correlation_f)
    print(num_of_f)
    print(correlation_m)
    print(num_of_m)

    path_losses = os.path.join(LOSSES, "TL1M.npy")
    train_loss_m_list = np.load(path_losses)
    path_losses = os.path.join(LOSSES, "TL1F.npy")
    train_loss_f_list = np.load(path_losses)
    path_losses = os.path.join(LOSSES, "TL1Avg.npy")
    train_loss_average_list = np.load(path_losses)
    path_losses = os.path.join(LOSSES, "VL1M.npy")
    validation_loss_m_list = np.load(path_losses)
    path_losses = os.path.join(LOSSES, "VL1F.npy")
    validation_loss_f_list = np.load(path_losses)
    path_losses = os.path.join(LOSSES, "VL1Avg.npy")
    validation_loss_average_list = np.load(path_losses)

    path_losses = os.path.join(LOSSES, "CorrF.npy")
    correlation_f_list = np.load(path_losses)
    path_losses = os.path.join(LOSSES, "CorrM.npy")
    correlation_m_list = np.load(path_losses)

    # plotting validation and training losses and saving them
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
    ax1.plot(train_loss_m_list, label="training")
    ax1.plot(validation_loss_m_list, label="validation")
    ax1.set_ylabel("L1 M")
    ax1.set_xlabel("Epoch")
    ax2.plot(train_loss_f_list, label="training")
    ax2.plot(validation_loss_f_list, label="validation")
    ax2.set_ylabel("L1 F")
    ax2.set_xlabel("Epoch")
    ax3.plot(train_loss_average_list, label="training")
    ax3.plot(validation_loss_average_list, label="validation")
    ax3.set_ylabel("L1 Avg")
    ax3.set_xlabel("Epoch")
    ax1.legend()
    ax2.legend()
    ax3.legend()
    plt.show()
    plt.close()

    fig, (ax1, ax2) = plt.subplots(2, 1)
    ax1.plot(correlation_f_list)
    ax1.set_ylabel("CorrF")
    ax1.set_xlabel("Epoch")
    ax2.plot(correlation_m_list)
    ax2.set_ylabel("CorrM")
    ax2.set_xlabel("Epoch")
    plt.show()
    plt.close()"""

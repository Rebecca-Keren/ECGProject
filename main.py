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

REAL_DATASET = os.path.join(os.path.dirname(os.path.realpath(__file__)), "NormalizedReal")

LOSSES = os.path.join(os.path.dirname(os.path.realpath(__file__)), "Losses")
if not os.path.exists(LOSSES):
    os.mkdir(LOSSES)

BATCH_SIZE = 32
epochs = 15
learning_rate_real = 1e-5

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
    torch.cuda.clear_memory_allocated()
    resnet_model = ResNet(1).cuda()
    best_model_accuracy_real = - math.inf
    val_loss_real = 0
    #early_stopping_real = EarlyStopping(delta_min=0.01, patience=6, verbose=True)
    criterion = nn.L1Loss().cuda()
    criterion_cent = CenterLoss(num_classes=2, feat_dim=512*64, use_gpu=device)
    params = list(resnet_model.parameters()) + list(criterion_cent.parameters())
    optimizer_model_real = optim.SGD(params, lr=learning_rate_real, momentum=0.9, weight_decay=1e-5)
    scheduler_real = torch.optim.lr_scheduler.OneCycleLR(optimizer_model_real, max_lr=1e-2, steps_per_epoch=int(np.ceil(len(train_data_loader_real.dataset)/BATCH_SIZE)),epochs=epochs+1)

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
           train_loss_ecg_list,
           scheduler_real)
        # Validation Real
        resnet_model.eval()
        criterion_cent.eval()
        best_model_accuracy_real, val_loss_real = val_real(
           val_data_loader_real,
           resnet_model,
           criterion,
           criterion_cent,
           epoch,
           epochs,
           validation_loss_ecg_list,
           validation_corr_ecg_list,
           best_model_accuracy_real)
        #early_stopping_real(val_loss_real.cpu().detach().numpy(), resnet_model)
        #if early_stopping_real.early_stop:
        #    print('Early stopping')
        #    break

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

    """path_losses = os.path.join(LOSSES, "TL1ECG.npy")
    train_loss_m_list = np.load(path_losses)
    path_losses = os.path.join(LOSSES, "VL1ECG.npy")
    validation_loss_m_list = np.load(path_losses)

    path_losses = os.path.join(LOSSES, "CorrECG.npy")
    correlation_f_list = np.load(path_losses)

    # plotting validation and training losses and saving them
    fig, (ax1,ax2) = plt.subplots(2, 1)
    ax1.plot(train_loss_m_list, label="training")
    ax1.plot(validation_loss_m_list, label="validation")
    ax1.set_ylabel("L1 ECG")
    ax1.set_xlabel("Epoch")
    ax2.plot(correlation_f_list)
    plt.show()
    plt.close()

    for filename in os.listdir(ECG_OUTPUTS_TEST_REAL):  # present the fecg outputs
        if "label_ecg" in filename:
            path_label = os.path.join(ECG_OUTPUTS_TEST_REAL, filename)
            number_file = filename.index("g") + 1
            end_path = filename[number_file:]
            path = os.path.join(ECG_OUTPUTS_TEST_REAL, "ecg" + end_path)
            mecg_label = os.path.join(ECG_OUTPUTS_TEST_REAL, "mecg" + end_path)
            fecg_label = os.path.join(ECG_OUTPUTS_TEST_REAL, "fecg" + end_path)

            fig, (ax1, ax2,ax3,ax4) = plt.subplots(4, 1)
            ax1.plot(np.load(path)[0])
            ax1.set_ylabel("ECG")
            ax2.plot(np.load(path_label)[0])
            ax2.set_ylabel("LABEL ECG")
            ax3.plot(np.load(mecg_label)[0])
            ax3.set_ylabel("MECG")
            ax4.plot(np.load(fecg_label)[0])
            ax4.set_ylabel("FECG")
            plt.show()
            plt.close()"""
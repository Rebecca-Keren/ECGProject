from ResnetNetwork import *
import torch.optim as optim
import torch.utils.data as data
from torch.autograd import Variable
from CenterLoss import *
import numpy as np
import os
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import math
from model import *
import dataloader


SIMULATED_DATASET = os.path.join(os.path.dirname(os.path.realpath(__file__)), "simulated_windows")


BATCH_SIZE = 32
epochs = 10
learning_rate = 1e-3


def main(dataset_size):
    LOSSES = os.path.join(os.path.dirname(os.path.realpath(__file__)), "Losses" + str(dataset_size))
    if not os.path.exists(LOSSES):
        os.mkdir(LOSSES)

    pl.seed_everything(1234)
    list_simulated = simulated_database_list(SIMULATED_DATASET)

    list_simulated_overfit = list_simulated[:dataset_size]  # TODO: put in comment after validating

    simulated_dataset = dataloader.SimulatedDataset(SIMULATED_DATASET,list_simulated_overfit) # TODO: change to original list size after validating

    train_size_sim = int(0.6 * len(simulated_dataset))
    val_size_sim = int(0.2 * len(simulated_dataset))
    test_size_sim = int(0.2 * len(simulated_dataset))

    train_dataset_sim, val_dataset_sim, test_dataset_sim = torch.utils.data.random_split(simulated_dataset, [train_size_sim, val_size_sim,test_size_sim])

    train_data_loader_sim = data.DataLoader(train_dataset_sim, batch_size=BATCH_SIZE, shuffle=True, num_workers=12)
    val_data_loader_sim = data.DataLoader(val_dataset_sim, batch_size=BATCH_SIZE, shuffle=False)
    test_data_loader_sim = data.DataLoader(test_dataset_sim, batch_size=BATCH_SIZE, shuffle=False)
    print("REBECCA")
    print("train")
    for i, batch_features in enumerate(train_data_loader_sim):
        for elem in batch_features:
            result = torch.any(torch.isnan(1000. * elem.transpose(1, 2).float()))
            if(result):
                print("True")
                print(i)
    print("val")
    for i, batch_features in enumerate(val_data_loader_sim):
        for elem in batch_features:
            result = torch.any(torch.isnan(1000. * elem.transpose(1, 2).float()))
            if(result):
                print("True")
                print(i)
    print("test")
    for i, batch_features in enumerate(test_data_loader_sim):
        for elem in batch_features:
            result = torch.any(torch.isnan(1000. * elem.transpose(1, 2).float()))
            if (result):
                print("True")
                print(i)

    #  use gpu if available
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    print(device)

    resnet_model = ResNet(1).cuda()
    best_model_accuracy = math.inf
    criterion = nn.L1Loss().cuda()
    criterion_cent = CenterLoss(num_classes=2, feat_dim=512*64, use_gpu=device)
    params = list(resnet_model.parameters()) + list(criterion_cent.parameters())
    optimizer_model = optim.SGD(params, lr=learning_rate, momentum=0.9,weight_decay=1e-4)
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
              train_loss_average_list,
              dataset_size)
        #Evaluation
        resnet_model.eval()
        val(val_data_loader_sim,
            resnet_model,
            criterion,
            epoch,
            epochs,
            validation_loss_m_list,
            validation_loss_f_list,
            validation_loss_average_list,
            validation_corr_m_list,
            validation_corr_f_list,
            best_model_accuracy,
            dataset_size)

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
    test_loss_m, test_loss_f, test_loss_avg, test_corr_m, test_corr_f, test_corr_average = test(str(network_save_folder_orig + str(dataset_size) + network_file_name_best),test_data_loader_sim,dataset_size)

    with open("test_loss" + str(dataset_size) + ".txt", 'w') as f:
        f.write("test_loss_m = {:.4f},test_loss_f = {:.4f},test_loss_avg = {:.4f},"
                "test_corr_m = {:.4f},test_corr_f = {:.4f},test_corr_avg = {:.4f}\n".format(test_loss_m, test_loss_f, test_loss_avg,
                                                                                            test_corr_m,test_corr_f,test_corr_average))
    del resnet_model
    del simulated_dataset
    del train_data_loader_sim
    torch.cuda.empty_cache()


if __name__=="__main__":

    #dataset_size = [50000,80000,100000]
    dataset_size = [127750]

    for size in dataset_size:
        main(size)
        """print(size)

        ECG_OUTPUTS_VAL = os.path.join(os.path.dirname(os.path.realpath(__file__)), "ECGOutputsVal" + str(size))
        ECG_OUTPUTS_TEST = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                        "ECGOutputsTest" + str(size))
        LOSSES = os.path.join(os.path.dirname(os.path.realpath(__file__)), "Losses" + str(size))
        for filename in os.listdir(ECG_OUTPUTS_TEST): #present the fecg outputs
            if "fecg" in filename:
                path = os.path.join(ECG_OUTPUTS_TEST, filename)
                number_file = filename.index("g") + 1
                end_path = filename[number_file:]
                path_label = os.path.join(ECG_OUTPUTS_TEST,"label_f" + end_path)
                fig, (ax1, ax2) = plt.subplots(2, 1)
                ax1.plot(np.load(path))
                ax1.set_ylabel("FECG")
                ax2.plot(np.load(path_label))
                ax2.set_ylabel("LABEL")
                plt.show()
                plt.close()
            if "mecg" in filename:
                path = os.path.join(ECG_OUTPUTS_TEST, filename)
                number_file = filename.index("g") + 1
                end_path = filename[number_file:]
                path_label = os.path.join(ECG_OUTPUTS_TEST, "label_m" + end_path)
                fig, (ax1, ax2) = plt.subplots(2, 1)
                ax1.plot(np.load(path))
                ax1.set_ylabel("MECG")
                ax2.plot(np.load(path_label))
                ax2.set_ylabel("LABEL")
                plt.show()
                plt.close()

        print("VAL")
        for filename in os.listdir(ECG_OUTPUTS_VAL): #present the fecg outputs
            if "fecg" in filename:
                path = os.path.join(ECG_OUTPUTS_VAL, filename)
                number_file = filename.index("g") + 1
                end_path = filename[number_file:]
                path_label = os.path.join(ECG_OUTPUTS_VAL,"label_f" + end_path)
                fig, (ax1, ax2) = plt.subplots(2, 1)
                ax1.plot(np.load(path))
                ax1.set_ylabel("FECG")
                ax2.plot(np.load(path_label))
                ax2.set_ylabel("LABEL")
                plt.show()
                plt.close()
            if "mecg" in filename:
                path = os.path.join(ECG_OUTPUTS_VAL, filename)
                number_file = filename.index("g") + 1
                end_path = filename[number_file:]
                path_label = os.path.join(ECG_OUTPUTS_VAL, "label_m" + end_path)
                fig, (ax1, ax2) = plt.subplots(2, 1)
                ax1.plot(np.load(path))
                ax1.set_ylabel("MECG")
                ax2.plot(np.load(path_label))
                ax2.set_ylabel("LABEL")
                plt.show()
                plt.close()

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


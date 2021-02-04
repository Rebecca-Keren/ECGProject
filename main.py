from ResnetNetwork import *
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision
from disentangledModel import *
from CenterLoss import *
from __future__ import print_function
import numpy as np
from torch.utils.data import Dataset
from scipy.io import loadmat
import os

REAL_DATASET = os.path.join(os.path.dirname(os.path.realpath(__file__)),"Real Database")
SIMULATED_DATASET = os.path.join(os.path.dirname(os.path.realpath(__file__)),"Merged Simulated Database")

BATCH_SIZE = 64
epochs = 20
learning_rate = 1e-3

class ResNet(nn.Module):
    def __init__(self, in_channels, n_classes, *args, **kwargs):
        super().__init__()
        self.latentSize = 256
        self.encoder = ResNetEncoder(in_channels, *args, **kwargs)
        self.disentangledRepresentation = DisentangledModel(self.latentSize)
        self.Mdecoder = ResnetDecoder(self.encoder.blocks[-1].blocks[-1].expanded_channels, n_classes)
        self.Fdecoder = ResnetDecoder(self.encoder.blocks[-1].blocks[-1].expanded_channels, n_classes)

        def forward(self, x):
            x = self.encoder(x)
            m,f = self.disentangledRepresentation(x)
            m_out = self.Mdecoder(m)
            f_out = self.Fdecoder(f)
            return m_out, f_out

class RealDataset(Dataset):
    def __init__(self, real_dir):
        self.real_dir = real_dir
        self.real_signals = os.listdir(real_dir)

    def __len__(self):
        return len(self.real_signals)

    def __getitem__(self, idx):
        path_signal = os.path.join(self.real_dir, self.real_signals[idx])
        signal = loadmat(path_signal)['data']
        return signal

class SimulatedDataset(Dataset):
    def __init__(self, simulated_dir,list):
        self.simulated_dir = simulated_dir
        self.simulated_signals = list

    def __len__(self):
        return len(self.simulated_signals)

    def __getitem__(self, idx):
        path_mix = os.path.join(self.simulated_dir, self.simulated_signals[idx][0])
        path_mecg = os.path.join(self.simulated_dir, self.simulated_signals[idx][1])
        path_fecg = os.path.join(self.simulated_dir, self.simulated_signals[idx][2])
        mix = loadmat(path_mix)['data']
        mecg = loadmat(path_mecg)['data']
        fecg = loadmat(path_fecg)['data']
        return [mix,mecg,fecg]

def main():

    list_simulated = simulated_database_list(SIMULATED_DATASET)
    real_dataset = RealDataset(REAL_DATASET)
    simulated_dataset = SimulatedDataset(SIMULATED_DATASET,list_simulated)

    train_size_real = int(0.8 * len(real_dataset))
    test_size_real = len(real_dataset) - train_size_real
    train_dataset_real, test_dataset_real = torch.utils.data.random_split(real_dataset, [train_size_real, test_size_real])

    train_size_sim = int(0.8 * len(simulated_dataset))
    test_size_sim = len(simulated_dataset) - train_size_sim
    train_dataset_sim, test_dataset_sim = torch.utils.data.random_split(simulated_dataset, [train_size_sim, test_size_sim])

    train_data_loader_real = data.DataLoader(train_dataset_real, batch_size=BATCH_SIZE, shuffle=True)
    test_data_loader_real = data.DataLoader(test_dataset_real, batch_size=BATCH_SIZE, shuffle=True)

    train_data_loader_sim = data.DataLoader(train_dataset_sim, batch_size=BATCH_SIZE, shuffle=True)
    test_data_loader_sim = data.DataLoader(test_dataset_sim, batch_size=BATCH_SIZE, shuffle=True)

    #  use gpu if available
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    resnet_model = ResNet().to(device)#todo - add paameters
    optimizer_model = optim.Adam(resnet_model.parameters(), lr=learning_rate)

    criterion = nn.MSELoss()
    criterion_clustering = nn.HingeEmbeddingLoss()

    criterion_cent = CenterLoss(num_classes=dataset.num_classes, feat_dim=2, use_gpu=device)  #todo - change params
    optimizer_centloss = optim.Adam(criterion_cent.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        loss_m = 0
        loss_f = 0

        for batch_features in train_data_loader:
            batch_features = batch_features.to(device)
            optimizer_model.zero_grad()
            optimizer_centloss.zero_grad()

            outputs_m,outputs_f = resnet_model(batch_features).to(device)

            #COST(M,M^)
            train_loss_mecg = criterion(outputs_m, batch_features)
            train_loss_mecg.backward()

            #COST(F,F^)
            train_loss_fecg = criterion(outputs_f, batch_features)
            train_loss_fecg.backward()

            #Center loss(one before last decoder M, one before last decoder F)
            one_before_last_m = nn.Sequential(*list(outputs_m.children())[:-2])
            one_before_last_f = nn.Sequential(*list(outputs_f.children())[:-2])
            loss_cent = criterion_cent(one_before_last_m, one_before_last_f)
            loss_cent.backword()

            #Clustering loss(one before last decoder M, one before last decoder F)
            hinge_loss = criterion_clustering(one_before_last_m, one_before_last_f)
            hinge_loss.backward()

            optimizer_model.step()
            optimizer_centloss.step()

            loss_m += train_loss_mecg.item()
            loss_f += train_loss_mecg.item()
            del batch_features
            torch.cuda.empty_cache()

        # compute the epoch training loss
        loss_m = loss_m / (len(train_data_loader)/2)
        loss_f = loss_f / (len(train_data_loader)/2)

        # display the epoch training loss
        print("epoch : {}/{}, loss mecg = {:.8f}, loss fecg = {:.8f}".format(epoch + 1, epochs, loss_m, loss_f))

    torch.cuda.empty_cache()
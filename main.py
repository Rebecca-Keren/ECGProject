from ResnetNetwork import *
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision
from CenterLoss import *
from __future__ import print_function
import numpy as np
from torch.utils.data import Dataset
from scipy.io import loadmat
import os

REAL_DATASET = os.path.join(os.path.dirname(os.path.realpath(__file__)),"Real Database")
SIMULATED_DATASET = os.path.join(os.path.dirname(os.path.realpath(__file__)),"Merged Simulated Database")

BATCH_SIZE = 16
epochs = 20
learning_rate = 1e-3
delta = 1e-2

class ResNet(nn.Module):
    def __init__(self, in_channels, *args, **kwargs):
        super().__init__()
        self.encoder = ResNetEncoder(in_channels, *args, **kwargs)
        self.Mdecoder = ResnetDecoder()
        self.Fdecoder = ResnetDecoder()

        def forward(self, x):
            x = self.encoder(x)
            latent_half = (x.size()[2] / 2)
            m = x[:,:,:latent_half]
            f = x[:,:,latent_half:]
            m_out,one_before_last_m = self.Mdecoder(m)
            f_out, one_before_last_f = self.Fdecoder(f)
            return m_out,one_before_last_m,f_out,one_before_last_f

# class RealDataset(Dataset):
#     def __init__(self, real_dir):
#         self.real_dir = real_dir
#         self.real_signals = os.listdir(real_dir)
#
#     def __len__(self):
#         return len(self.real_signals)
#
#     def __getitem__(self, idx):
#         path_signal = os.path.join(self.real_dir, self.real_signals[idx])
#         signal = loadmat(path_signal)['data']
#         return signal

class SimulatedDataset(Dataset):
    def __init__(self, simulated_dir,list):
        self.simulated_dir = simulated_dir
        self.simulated_signals = list

    def __len__(self):
        return len(self.simulated_signals)

    # def get_m(self,idx):
    #     path_mecg = os.path.join(self.simulated_dir, self.simulated_signals[idx][1])
    #     mecg = torch.from_numpy(loadmat(path_mecg)['data'])
    #     return mecg
    #
    # def get_f(self,idx):
    #     path_fecg = os.path.join(self.simulated_dir, self.simulated_signals[idx][2])
    #     fecg = torch.from_numpy(loadmat(path_fecg)['data'])
    #     return fecg

    def __getitem__(self, idx):
        path_mix = os.path.join(self.simulated_dir, self.simulated_signals[idx][0])
        path_mecg = os.path.join(self.simulated_dir, self.simulated_signals[idx][1])
        path_fecg = os.path.join(self.simulated_dir, self.simulated_signals[idx][2])
        mix =  torch.from_numpy(loadmat(path_mix)['data'])
        mecg =  torch.from_numpy(loadmat(path_mecg)['data'])
        fecg =  torch.from_numpy(loadmat(path_fecg)['data'])
        return mix,mecg,fecg

def main():

    list_simulated = simulated_database_list(SIMULATED_DATASET)
    #real_dataset = RealDataset(REAL_DATASET)
    simulated_dataset = SimulatedDataset(SIMULATED_DATASET,list_simulated)

    # train_size_real = int(0.8 * len(real_dataset))
    # test_size_real = len(real_dataset) - train_size_real
    # train_dataset_real, test_dataset_real = torch.utils.data.random_split(real_dataset, [train_size_real, test_size_real])

    train_size_sim = int(0.8 * len(simulated_dataset))
    test_size_sim = len(simulated_dataset) - train_size_sim
    train_dataset_sim, test_dataset_sim = torch.utils.data.random_split(simulated_dataset, [train_size_sim, test_size_sim])

    # train_data_loader_real = data.DataLoader(train_dataset_real, batch_size=BATCH_SIZE, shuffle=True)
    # test_data_loader_real = data.DataLoader(test_dataset_real, batch_size=BATCH_SIZE, shuffle=True)

    train_data_loader_sim = data.DataLoader(train_dataset_sim, batch_size=BATCH_SIZE, shuffle=True)
    test_data_loader_sim = data.DataLoader(test_dataset_sim, batch_size=BATCH_SIZE, shuffle=True)

    #  use gpu if available
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    resnet_model = ResNet(1).to(device)
    optimizer_model = optim.Adam(resnet_model.parameters(), lr=learning_rate)

    criterion = nn.MSELoss()

    criterion_cent = CenterLoss(num_classes=2, feat_dim=512*16, use_gpu=device)
    optimizer_centloss = optim.Adam(criterion_cent.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        total_loss_epoch = 0
        total_loss_m = 0
        total_loss_f = 0
        total_loss_cent = 0
        total_loss_hinge = 0

        for i, batch_features in enumerate(train_data_loader_sim):
            batch_for_model = batch_features[0].to(device)
            batch_for_m =  batch_features[1].to(device)
            batch_for_f = batch_features[2].to(device)
            # batch_features = batch_features.to(device)
            optimizer_model.zero_grad()
            optimizer_centloss.zero_grad()

            outputs_m,one_before_last_m,outputs_f,one_before_last_f = resnet_model(batch_for_model).to(device)

            #COST(M,M^)
            train_loss_mecg = criterion(outputs_m, batch_for_m)

            #COST(F,F^)
            train_loss_fecg = criterion(outputs_f, batch_for_f)

            #Center loss(one before last decoder M, one before last decoder F)
            flatten_m,flatten_f = torch.flatten(one_before_last_m,start_dim=1), torch.flatten(one_before_last_f,start_dim=1) #TODO check
            input = torch.cat((flatten_f,flatten_m), 0) #TODO check
            first_label,second_label = torch.zeros(BATCH_SIZE), torch.ones(BATCH_SIZE)
            labels = torch.cat((first_label,second_label))
            loss_cent = criterion_cent(input, labels)

            #Clustering loss(one before last decoder M, one before last decoder F)
            hinge_loss = hinge_loss(one_before_last_m, one_before_last_f,delta)

            total_loss = train_loss_mecg + train_loss_fecg + loss_cent + hinge_loss
            total_loss.backward()
            optimizer_model.step()
            optimizer_centloss.step()

            total_loss_m += train_loss_mecg.item()
            total_loss_f += train_loss_fecg.item()
            total_loss_cent += loss_cent.item()
            total_loss_hinge += hinge_loss.item()
            total_loss_epoch += total_loss
            del batch_features
            torch.cuda.empty_cache()

        # compute the epoch training loss
        total_loss_m = total_loss_m / (len(train_data_loader_sim))
        total_loss_f = total_loss_f / (len(train_data_loader_sim))
        total_loss_cent = total_loss_cent / (len(train_data_loader_sim))
        total_loss_hinge = total_loss_hinge / (len(train_data_loader_sim))

        # display the epoch training loss
        print("epoch : {}/{}, total_loss = {:.8f}, loss_mecg = {:.8f}, loss_fecg = {:.8f}, loss_cent = {:.8f}, loss_hinge = {:.8f}".format(epoch + 1, epochs, total_loss_epoch, total_loss_m, total_loss_f, total_loss_cent, total_loss_hinge))

    del resnet_model
    del simulated_dataset
    del train_data_loader_sim
    torch.cuda.empty_cache()
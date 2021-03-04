from __future__ import print_function
from ResnetNetwork import *
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision
from CenterLoss import *
import numpy as np
from torch.utils.data import Dataset
from scipy.io import loadmat
import os

#REAL_DATASET = os.path.join(os.path.dirname(os.path.realpath(__file__)),"Real Database") #TODO sistemare grandezza
SIMULATED_DATASET = os.path.join(os.path.dirname(os.path.realpath(__file__)), "SimulatedDatabase")

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
        x, indices = self.encoder(x)
        latent_half = int((x.size()[2] / 2))
        m = x[:,:,:latent_half]
        f = x[:,:,latent_half:]
        m_out,one_before_last_m = self.Mdecoder(m, indices)
        f_out, one_before_last_f = self.Fdecoder(f, indices)
        return m_out,one_before_last_m,f_out,one_before_last_f

class RealDataset(Dataset):
    def __init__(self, real_dir):
        self.real_dir = real_dir
        self.real_signals = os.listdir(real_dir)

    def __len__(self):
        return len(self.real_signals)

    def __getitem__(self, idx):
        path_signal = os.path.join(self.real_dir, self.real_signals[idx])
        signal = torch.from_numpy(loadmat(path_signal)['data'])
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
        mix =  torch.from_numpy(loadmat(path_mix)['data']) #TODO cambiare il cast
        mecg =  torch.from_numpy(loadmat(path_mecg)['data'])
        fecg =  torch.from_numpy(loadmat(path_fecg)['data'])
        return mix,mecg,fecg

def main():
    #torch.set_default_tensor_type('torch.FloatTensor')
    list_simulated = simulated_database_list(SIMULATED_DATASET)
    #print(list_simulated)
    #real_dataset = RealDataset(REAL_DATASET)
    simulated_dataset = SimulatedDataset(SIMULATED_DATASET,list_simulated)

    #train_size_real = int(0.8 * len(real_dataset))
    #test_size_real = len(real_dataset) - train_size_real
    #train_dataset_real, test_dataset_real = torch.utils.data.random_split(real_dataset, [train_size_real, test_size_real])

    train_size_sim = int(0.8 * len(simulated_dataset))
    test_size_sim = len(simulated_dataset) - train_size_sim
    train_dataset_sim, test_dataset_sim = torch.utils.data.random_split(simulated_dataset, [train_size_sim, test_size_sim])

    #train_data_loader_real = data.DataLoader(train_dataset_real, batch_size=BATCH_SIZE, shuffle=True)
    #test_data_loader_real = data.DataLoader(test_dataset_real, batch_size=BATCH_SIZE, shuffle=True)

    train_data_loader_sim = data.DataLoader(train_dataset_sim, batch_size=BATCH_SIZE, shuffle=True)
    test_data_loader_sim = data.DataLoader(test_dataset_sim, batch_size=BATCH_SIZE, shuffle=True)

    #  use gpu if available
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    #resnet_model = ResNet(1).to(device)
    resnet_model = ResNet(1)
    optimizer_model = optim.Adam(resnet_model.parameters(), lr=learning_rate)

    criterion = nn.MSELoss()

    criterion_cent = CenterLoss(num_classes=2, feat_dim=1024, use_gpu=device)
    optimizer_centloss = optim.Adam(criterion_cent.parameters(), lr=learning_rate)


    for epoch in range(epochs):
        total_loss_epoch = 0
        total_loss_m = 0
        total_loss_f = 0
        total_loss_ecg = 0
        total_loss_cent = 0
        total_loss_hinge = 0

        #real_epoch = ((epoch % 5) == 0)
        real_epoch = 0

        #if (real_epoch):
        #    data_loader = train_data_loader_real
        #else:
        data_loader = train_data_loader_sim

        for i, batch_features in enumerate(data_loader):
            print(i)
            if (real_epoch):
                batch_for_model = batch_features.to(device)
            else:
                batch_for_model = batch_features[0].to(device)
                batch_for_m =  batch_features[1].to(device)
                batch_for_f = batch_features[2].to(device)
            batch_size = batch_for_model.size()[0]
            print("batch: "+str(batch_for_model.size()[0]))
            optimizer_model.zero_grad()
            optimizer_centloss.zero_grad()

            #outputs_m,one_before_last_m,outputs_f,one_before_last_f = resnet_model(batch_for_model).to(device)
            #print(batch_for_model.size())
            batch_for_model = batch_for_model.transpose(1,2)
            batch_for_m = batch_for_m.transpose(1, 2)
            batch_for_f = batch_for_f.transpose(1, 2)

            #print(batch_for_model.size())
            outputs_m, one_before_last_m, outputs_f, one_before_last_f = resnet_model(batch_for_model.double())

            if(not real_epoch):
                #COST(M,M^)
                #print(batch_for_m.size())
                #print(outputs_m.size())
                train_loss_mecg = criterion(batch_for_m.float(),outputs_m)
                #COST(F,F^)
                train_loss_fecg = criterion(batch_for_f.float(),outputs_f)
            else:
                outputs_m = torch.add(outputs_m,outputs_f)
                train_loss_ecg = criterion(batch_for_model.float(),outputs_m)


            #Center loss(one before last decoder M, one before last decoder F)
            flatten_m,flatten_f = torch.flatten(one_before_last_m,start_dim=1), torch.flatten(one_before_last_f,start_dim=1)
            input = torch.cat((flatten_f,flatten_m), 0)
            first_label,second_label = torch.zeros(batch_size), torch.ones(batch_size)
            labels = torch.cat((first_label,second_label))
            print("input: " +str(input.size()))
            print("labels: "+ str(labels.size()))
            loss_cent = criterion_cent(input, labels)

            #Clustering loss(one before last decoder M, one before last decoder F)
            hinge_loss = criterion_hinge_loss(one_before_last_m, one_before_last_f,delta)
            hinge_loss = torch.tensor(hinge_loss,dtype=torch.float32)

            # print(train_loss_mecg.float().dtype)
            # print(train_loss_fecg.float().dtype)
            # print(loss_cent.dtype)
            # print(torch.tensor(hinge_loss,dtype=torch.float32).dtype)

            if(not real_epoch):
                total_loss = train_loss_mecg + train_loss_fecg + loss_cent + hinge_loss
            else:
                total_loss = train_loss_ecg + loss_cent + hinge_loss

            #print(total_loss.dtype)
            total_loss.backward()
            optimizer_model.step()
            optimizer_centloss.step()

            if(not real_epoch):
                total_loss_m += train_loss_mecg.item()
                total_loss_f += train_loss_fecg.item()
            else:
                total_loss_ecg += train_loss_ecg.item()
            total_loss_cent += loss_cent.item()
            total_loss_hinge += hinge_loss.item()
            total_loss_epoch += total_loss
            del batch_features
            torch.cuda.empty_cache()

        # compute the epoch training loss
        if(not real_epoch):
            total_loss_m = total_loss_m / (len(data_loader))
            total_loss_f = total_loss_f / (len(data_loader))
        else:
            total_loss_ecg = total_loss_ecg / (len(data_loader))
        total_loss_cent = total_loss_cent / (len(data_loader))
        total_loss_hinge = total_loss_hinge / (len(data_loader))
        total_loss = total_loss / (len(data_loader))
        # display the epoch training loss
        if(not real_epoch):
            print("epoch S : {}/{}, total_loss = {:.8f}, loss_mecg = {:.8f}, loss_fecg = {:.8f}, loss_cent = {:.8f}, loss_hinge = {:.8f}".format(epoch + 1, epochs, total_loss_epoch, total_loss_m, total_loss_f, total_loss_cent, total_loss_hinge))
        else:
            print(
                "epoch R : {}/{}, total_loss = {:.8f}, loss_ecg = {:.8f}, loss_cent = {:.8f}, loss_hinge = {:.8f}".format(
                    epoch + 1, epochs, total_loss_epoch, total_loss_ecg, total_loss_cent, total_loss_hinge))

    del resnet_model
    del simulated_dataset
    del train_data_loader_sim
    torch.cuda.empty_cache()

if __name__=="__main__":
    device = torch.device("cuda:1" if (torch.cuda.is_available()) else "cpu")
    #print(device)
    main()
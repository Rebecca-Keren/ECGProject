from ResnetNetwork import *
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision
from torch.utils.data import Dataset
from disentangledModel import *
import os
from CenterLoss import *

BATCH_SIZE = 64
epochs = 20
learning_rate = 1e-3
TRAIN_DATA_PATH = 0

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

class ECGSignalsDataSet(Dataset):#hi
    def __init__(self, main_dir):
        self.main_dir = main_dir
        self.signals = os.listdir(main_dir)
        self.mecg = 0 #todo - get the mecg
        self.fecg = 0 #todo - get the fecg

    def __getitem__(self, idx):
        signal_loc = os.path.join(self.main_dir, self.signals[idx])
        image = Image.open(img_loc).convert("RGB")#todo - change f
        tensor_image = self.transform(image)#todo - change
        return tensor_image#todo - change

def main():
    ecgSignals = ECGSignalsDataSet(TRAIN_DATA_PATH)

    originalMECG = ecgSignals.mecg
    originalFECG = ecgSignals.fecg

    train_size = int(0.8 * len(ecgSignals))
    test_size = len(ecgSignals) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(ecgSignals, [train_size, test_size])

    train_data_loader = data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_data_loader = data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

    #  use gpu if available
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    m_out,f_out = ResNet().to(device)#todo - add paameters
    optimizer_m = optim.Adam(m_out.parameters(), lr=learning_rate)
    optimizer_f = optim.Adam(f_out.parameters(), lr=learning_rate)

    criterion = nn.MSELoss()
    criterion_clustering = nn.HingeEmbeddingLoss()

    criterion_cent = CenterLoss(num_classes=dataset.num_classes, feat_dim=2, use_gpu=device) #todo - change params
    optimizer_centloss = optim.Adam(criterion_cent.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        loss_m = 0
        loss_f = 0
        for batch_features in train_data_loader:
            batch_features = batch_features.to(device)
            optimizer_m.zero_grad()
            optimizer_f.zero_grad()
            optimizer_centloss.zero_grad()

            outputs_m = m_out(batch_features).to(device)
            outputs_f = f_out(batch_features).to(device)

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

            optimizer_m.step()
            optimizer_f.step()
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
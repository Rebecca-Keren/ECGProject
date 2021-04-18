from ResnetNetwork import *
import torch.optim as optim
import torch.utils.data as data
from torch.autograd import Variable
from CenterLoss import *
import numpy as np
from torch.utils.data import Dataset
from scipy.io import loadmat
import os
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import math


SIMULATED_DATASET = os.path.join(os.path.dirname(os.path.realpath(__file__)), "simulated_windows")
ECG_OUTPUTS = os.path.join(os.path.dirname(os.path.realpath(__file__)), "ECGOutputs")
ECG_OUTPUTS_TEST = os.path.join(os.path.dirname(os.path.realpath(__file__)), "ECGOutputsTest")
ECG_OUTPUTS_VAL = os.path.join(os.path.dirname(os.path.realpath(__file__)), "ECGOutputsVal")
MODELS = os.path.join(os.path.dirname(os.path.realpath(__file__)), "Models")
LOSSES = os.path.join(os.path.dirname(os.path.realpath(__file__)), "Losses")

network_save_folder = "./Models/"
network_file_name_last = "last_model"
network_file_name_best = "best_model"


BATCH_SIZE = 32
epochs = 700
learning_rate = 1e-4
delta = 3

fecg_lamda = 10.
cent_lamda = 0.01
hinge_lamda = 0.5

mecg_weight = fecg_weight = 1.
cent_weight = 1.
hinge_weight = 1.

include_mecg_loss = True
include_fecg_loss = True
include_center_loss = True
include_hinge_loss = True


class ResNet(nn.Module):
    def __init__(self, in_channels,*args, **kwargs):
        super().__init__()
        self.encoder = ResNetEncoder(in_channels, *args, **kwargs)
        self.Mdecoder = ResnetDecoder()
        self.Fdecoder = ResnetDecoder()

    def forward(self, x):
        x = self.encoder(x)
        latent_half = x.size()[1] // 2
        m = x[:, :latent_half, :]
        f = x[:, latent_half:, :]
        m_out, one_before_last_m = self.Mdecoder(m)
        f_out, one_before_last_f = self.Fdecoder(f)
        return m_out, one_before_last_m, f_out, one_before_last_f


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
    def __init__(self, simulated_dir, list):
        self.simulated_dir = simulated_dir
        self.simulated_signals = list

    def __len__(self):
        return len(self.simulated_signals)

    def __getitem__(self, idx):
        path_mix = os.path.join(self.simulated_dir, self.simulated_signals[idx][0])
        path_mecg = os.path.join(self.simulated_dir, self.simulated_signals[idx][1])
        path_fecg = os.path.join(self.simulated_dir, self.simulated_signals[idx][2])
        mix = torch.from_numpy(loadmat(path_mix)['data'])
        mecg = torch.from_numpy(loadmat(path_mecg)['data'])
        fecg = torch.from_numpy(loadmat(path_fecg)['data'])
        return mix, mecg, fecg

def test_ecg_model(filename,test_data_loader_sim):
    resnet_model = ResNet(1)
    resnet_model.load_state_dict(torch.load(filename))
    resnet_model.eval()

    resnet_model.cuda()

    criterion = nn.L1Loss().cuda()

    test_loss_m = 0
    test_loss_f = 0
    with torch.no_grad():
        for i, batch_features in enumerate(test_data_loader_sim):
            batch_for_model_test = Variable(1000. * batch_features[0].transpose(1, 2).float().cuda())
            batch_for_m_test = Variable(1000. * batch_features[1].transpose(1, 2).float().cuda())
            batch_for_f_test = Variable(1000. * batch_features[2].transpose(1, 2).float().cuda())
            outputs_m_test, _, outputs_f_test, _ = resnet_model(batch_for_model_test)
            test_loss_m += criterion(outputs_m_test, batch_for_m_test)
            test_loss_f += criterion(outputs_f_test, batch_for_f_test)
    test_loss_m /= len(test_data_loader_sim.dataset)
    test_loss_f /= len(test_data_loader_sim.dataset)
    test_loss_average = (test_loss_m + test_loss_f) / 2  # TODO check
    if not os.path.exists(ECG_OUTPUTS_TEST):
        os.mkdir(ECG_OUTPUTS_TEST)
    path = os.path.join(ECG_OUTPUTS_TEST, "ecg_all" + str(i))
    np.save(path, batch_features[0][0].cpu().detach().numpy()[:, 0])
    path = os.path.join(ECG_OUTPUTS_TEST, "label_m" + str(i))
    np.save(path, batch_features[1][0].cpu().detach().numpy()[:, 0])
    path = os.path.join(ECG_OUTPUTS_TEST, "label_f" + str(i))
    np.save(path, batch_features[2][0].cpu().detach().numpy()[:, 0])
    path = os.path.join(ECG_OUTPUTS_TEST, "fecg" + str(i))
    np.save(path, outputs_f_test[0][0].cpu().detach().numpy() / 1000.)
    path = os.path.join(ECG_OUTPUTS_TEST, "mecg" + str(i))
    np.save(path, outputs_m_test[0][0].cpu().detach().numpy() / 1000.)
    return test_loss_m,test_loss_f,test_loss_average

def main():
    pl.seed_everything(1234)
    list_simulated = simulated_database_list(SIMULATED_DATASET)

    list_simulated_overfit = list_simulated[:50000]  # TODO: put in comment after validating

    simulated_dataset = SimulatedDataset(SIMULATED_DATASET,list_simulated_overfit) # TODO: change to original list size after validating

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

    optimizer_model = optim.SGD(resnet_model.parameters(), lr=learning_rate, momentum=0.9)
    best_model_accuracy = math.inf
    criterion = nn.L1Loss().cuda()
    criterion_cent = CenterLoss(num_classes=2, feat_dim=512*64, use_gpu=device)
    optimizer_centloss = optim.Adam(criterion_cent.parameters(), lr=learning_rate)
    validation_loss_f_list = []
    validation_loss_m_list = []
    validation_loss_average_list = []
    validation_corr_m_list = []
    validation_corr_f_list = []
    for epoch in range(epochs):

        total_loss_epoch = 0.
        total_loss_m = 0.
        total_loss_f = 0.
        total_loss_ecg = 0.
        total_loss_cent = 0.
        total_loss_hinge = 0.

        real_epoch = False
        resnet_model.train()
        for i, batch_features in enumerate(train_data_loader_sim):
            optimizer_model.zero_grad()
            optimizer_centloss.zero_grad()

            batch_for_model = Variable(1000.*batch_features[0].transpose(1,2).float().cuda())
            batch_for_m = Variable(1000.*batch_features[1].transpose(1, 2).float().cuda())
            batch_for_f = Variable(1000.*batch_features[2].transpose(1, 2).float().cuda())

            outputs_m, one_before_last_m, outputs_f, one_before_last_f = resnet_model(batch_for_model)

            if epoch+1 == epochs:
                if not os.path.exists(ECG_OUTPUTS):
                    os.mkdir(ECG_OUTPUTS)
                path = os.path.join(ECG_OUTPUTS, "ecg_all" + str(i))
                np.save(path, batch_features[0][0].cpu().detach().numpy()[:, 0])
                path = os.path.join(ECG_OUTPUTS, "label_m" + str(i))
                np.save(path, batch_features[1][0].cpu().detach().numpy()[:, 0])
                path = os.path.join(ECG_OUTPUTS, "label_f" + str(i))
                np.save(path, batch_features[2][0].cpu().detach().numpy()[:, 0])
                path = os.path.join(ECG_OUTPUTS, "fecg" + str(i))
                np.save(path, outputs_f[0][0].cpu().detach().numpy()/1000.)
                path = os.path.join(ECG_OUTPUTS, "mecg" + str(i))
                np.save(path, outputs_m[0][0].cpu().detach().numpy()/1000.)

            if not real_epoch:
                #COST(M,M^)
                train_loss_mecg = criterion(outputs_m, batch_for_m)

                #COST(F,F^)
                train_loss_fecg = criterion(outputs_f, batch_for_f)
            else:
                outputs_m += outputs_f
                train_loss_ecg = criterion(outputs_m, batch_for_model)

            flatten_m, flatten_f = torch.flatten(one_before_last_m, start_dim=1), torch.flatten(one_before_last_f,
                                                                                                 start_dim=1)
            hinge_loss = criterion_hinge_loss(one_before_last_m, one_before_last_f, delta)
            batch_size = one_before_last_m.size()[0]
            labels_center_loss = Variable(torch.cat((torch.zeros(batch_size), torch.ones(batch_size))).cuda())
            loss_cent = criterion_cent(torch.cat((flatten_f, flatten_m), 0), labels_center_loss)

            # if not real_epoch:
            total_loss = mecg_weight * train_loss_mecg + fecg_weight*fecg_lamda*train_loss_fecg
            if include_center_loss:
                 total_loss += cent_weight*cent_lamda*loss_cent
            if include_hinge_loss:
             total_loss += hinge_weight*hinge_lamda*hinge_loss
            # else:
            #     total_loss = train_loss_ecg + cent_weight*cent_lamda*loss_cent + hinge_weight*hinge_lamda*hinge_loss #TODO: check lamda for ecg and change loss ecg


            total_loss.backward()
            optimizer_model.step()
            optimizer_centloss.step()

            if not real_epoch:
                total_loss_m += mecg_weight*train_loss_mecg.item()
                total_loss_f += fecg_weight*fecg_lamda*train_loss_fecg.item()
            else:
                total_loss_ecg += train_loss_ecg.item()#TODO: check adding lamdas and weights
            total_loss_cent += cent_weight*cent_lamda*loss_cent.item()
            total_loss_hinge += hinge_weight*hinge_lamda*hinge_loss.item()
            total_loss_epoch += total_loss.item()
            batch_features, batch_for_model, batch_for_m, batch_for_f, total_loss, outputs_m, one_before_last_m, \
            outputs_f, one_before_last_f, train_loss_mecg, train_loss_fecg = None, None, None, None, None, None, None, \
                                                                             None, None, None, None

        # compute the epoch training loss
        if not real_epoch:
            total_loss_m = total_loss_m / (len(train_data_loader_sim))
            total_loss_f = total_loss_f / (len(train_data_loader_sim))
        else:
            total_loss_ecg = total_loss_ecg / (len(train_data_loader_sim))
        total_loss_cent = total_loss_cent / (len(train_data_loader_sim))
        total_loss_hinge = total_loss_hinge / (len(train_data_loader_sim))
        total_loss_epoch = total_loss_epoch / (len(train_data_loader_sim))
        # display the epoch training loss
        if not real_epoch:
            if not include_center_loss and not include_hinge_loss:
                print("epoch S : {}/{}, total_loss = {:.8f}, loss_mecg = {:.8f}, loss_fecg = {:.8f}".format(epoch + 1, epochs, total_loss_epoch, total_loss_m, total_loss_f))
            else:
                print("epoch S : {}/{}, total_loss = {:.8f}, loss_mecg = {:.8f}, loss_fecg = {:.8f}, loss_cent = {:.8f}, loss_hinge = {:.8f}".format(
                        epoch + 1, epochs, total_loss_epoch, total_loss_m, total_loss_f, total_loss_cent,total_loss_hinge))
        else:
            print("epoch R : {}/{}, total_loss = {:.8f}, loss_ecg = {:.8f}, loss_cent = {:.8f}, loss_hinge = {:.8f}".format(
                    epoch + 1, epochs, total_loss_epoch, total_loss_ecg, total_loss_cent, total_loss_hinge))

        if epoch + 1 == epochs:
            with open("losses.txt", 'a') as f:
                f.write("epoch S : {}/{}, total_loss = {:.8f}, loss_mecg = {:.8f}, loss_fecg = {:.8f}, loss_cent = {:.8f}, loss_hinge = {:.8f}\n".format(
                        epoch + 1, epochs, total_loss_epoch, total_loss_m, total_loss_f, total_loss_cent,total_loss_hinge))


        resnet_model.eval()
        val_loss_m = 0
        val_loss_f = 0
        val_corr_m = 0
        val_corr_f = 0
        with torch.no_grad():
            for i, batch_features in enumerate(val_data_loader_sim):
                batch_for_model_val = Variable(1000. * batch_features[0].transpose(1, 2).float().cuda())
                batch_for_m_val = Variable(1000. * batch_features[1].transpose(1, 2).float().cuda())
                batch_for_f_val = Variable(1000. * batch_features[2].transpose(1, 2).float().cuda())
                outputs_m_test, _, outputs_f_test, _ = resnet_model(batch_for_model_val)
                val_loss_m += criterion(outputs_m_test, batch_for_m_val)
                val_loss_f += criterion(outputs_f_test, batch_for_f_val)
                for i,elem in enumerate(outputs_m_test):
                    val_corr_m += np.corrcoef(outputs_m_test.cpu().detach().numpy()[i],batch_for_m_val.cpu().detach().numpy()[i])[0][1]
                    val_corr_f += np.corrcoef(outputs_f_test.cpu().detach().numpy()[i],batch_for_f_val.cpu().detach().numpy()[i])[0][1]
        val_loss_m /= len(val_data_loader_sim.dataset)
        val_loss_f /= len(val_data_loader_sim.dataset)
        val_corr_m /= len(val_data_loader_sim.dataset)
        val_corr_f /= len(val_data_loader_sim.dataset)
        val_corr_average = (val_corr_m + val_corr_f) / 2
        val_loss_average = (val_loss_m + val_loss_f) / 2 #TODO check

        #saving validation losses
        validation_loss_m_list.append(val_loss_m.cpu().detach())
        validation_loss_f_list.append(val_loss_f.cpu().detach())
        validation_loss_average_list.append(val_loss_average.cpu().detach())
        validation_corr_m_list.append(val_corr_m)
        validation_corr_f_list.append(val_corr_f)

        #saving last model
        torch.save(resnet_model.state_dict(), str(network_save_folder + network_file_name_last))
        #saving best model
        if (val_loss_average < best_model_accuracy):
            best_model_accuracy = val_loss_average
            torch.save(resnet_model.state_dict(), str(network_save_folder + network_file_name_best))
            print("saving best model")

        print('Validation: Average loss M: {:.4f}, Average Loss F: {:.4f}, Average Loss M+F: {:.4f}, Correlation M: {:.4f},Correlation F: {:.4f},Correlation Average: {:.4f})\n'.format(val_loss_m, val_loss_f,val_loss_average,val_corr_m,val_corr_f,val_corr_average))

        if epoch + 1 == epochs:
            if not os.path.exists(ECG_OUTPUTS_VAL):
                os.mkdir(ECG_OUTPUTS_VAL)
            path = os.path.join(ECG_OUTPUTS_VAL, "ecg_all" + str(i))
            np.save(path, batch_features[0][0].cpu().detach().numpy()[:, 0])
            path = os.path.join(ECG_OUTPUTS_VAL, "label_m" + str(i))
            np.save(path, batch_features[1][0].cpu().detach().numpy()[:, 0])
            path = os.path.join(ECG_OUTPUTS_VAL, "label_f" + str(i))
            np.save(path, batch_features[2][0].cpu().detach().numpy()[:, 0])
            path = os.path.join(ECG_OUTPUTS_VAL, "fecg" + str(i))
            np.save(path, outputs_f_test[0][0].cpu().detach().numpy() / 1000.)
            path = os.path.join(ECG_OUTPUTS_VAL, "mecg" + str(i))
            np.save(path, outputs_m_test[0][0].cpu().detach().numpy() / 1000.)

    """#plotting validation losses and saving them
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
    ax1.plot(validation_loss_m_list)
    ax1.set_ylabel("L1 M")
    ax1.set_xlabel("Epoch")
    ax2.plot(validation_loss_f_list)
    ax2.set_ylabel("L1 F")
    ax2.set_xlabel("Epoch")
    ax3.plot(validation_loss_average_list)
    ax3.set_ylabel("L1 Avg")
    ax3.set_xlabel("Epoch")
    plt.show()
    plt.close()

    fig, (ax1, ax2) = plt.subplots(2, 1)
    ax1.plot(validation_corr_m_list)
    ax1.set_ylabel("Corr M")
    ax1.set_xlabel("Epoch")
    ax2.plot(validation_corr_f_list)
    ax2.set_ylabel("Corr F")
    ax2.set_xlabel("Epoch")"""


    path_losses = os.path.join(LOSSES, "L1M")
    np.save(path_losses, np.array(validation_loss_m_list))
    path_losses = os.path.join(LOSSES, "L1F")
    np.save(path_losses, np.array(validation_loss_f_list))
    path_losses = os.path.join(LOSSES, "L1Avg")
    np.save(path_losses, np.array(validation_loss_average_list))
    path_losses = os.path.join(LOSSES, "CorrM")
    np.save(path_losses, np.array(validation_corr_m_list))
    path_losses = os.path.join(LOSSES, "CorrF")
    np.save(path_losses, np.array(validation_corr_f_list))

    #TODO UNCOMMENT
    """test_loss_m, test_loss_f, test_loss_avg = test_ecg_model(str(network_save_folder + network_file_name_best),test_data_loader_sim)

    with open("test_loss.txt", 'w') as f:
        f.write("test_loss_m = {:.4f},test_loss_f = {:.4f},test_loss_avg = {:.4f}\n".format(test_loss_m,test_loss_f,test_loss_avg))"""
    del resnet_model
    del simulated_dataset
    del train_data_loader_sim
    torch.cuda.empty_cache()


if __name__=="__main__":
    # device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    main()

    """for filename in os.listdir(ECG_OUTPUTS_TEST): #present the fecg outputs
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

    path_losses = os.path.join(LOSSES, "L1M.npy")
    validation_loss_m_list = np.load(path_losses)
    path_losses = os.path.join(LOSSES, "L1F.npy")
    validation_loss_f_list = np.load(path_losses)
    path_losses = os.path.join(LOSSES, "L1Avg.npy")
    validation_loss_average_list = np.load(path_losses)

    # plotting validation losses and saving them
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
    ax1.plot(validation_loss_m_list)
    ax1.set_ylabel("L1 M")
    ax1.set_xlabel("Epoch")
    ax2.plot(validation_loss_f_list)
    ax2.set_ylabel("L1 F")
    ax2.set_xlabel("Epoch")
    ax3.plot(validation_loss_average_list)
    ax3.set_ylabel("L1 Avg")
    ax3.set_xlabel("Epoch")
    plt.show()
    plt.close()"""

    """for filename in os.listdir(ECG_OUTPUTS_TEST): #present the fecg outputs
        if "ecg_all" in filename:
            path = os.path.join(ECG_OUTPUTS_TEST, filename)
            plt.plot(np.load(path))
            plt.title("all")
            plt.show()
        if "fecg" in filename:
            path = os.path.join(ECG_OUTPUTS_TEST, filename)
            plt.plot(np.load(path))
            plt.title("fecg")
            plt.show()
        if "mecg" in filename:
            print("mecg")
            path = os.path.join(ECG_OUTPUTS_TEST, filename)
            plt.plot(np.load(path))
            plt.title("mecg")
            plt.show()
        if "label_m" in filename:
            print("label_m")
            path = os.path.join(ECG_OUTPUTS_TEST, filename)
            plt.plot(np.load(path))
            plt.title("label_m")
            plt.show()
        if "label_f" in filename:
            print("label_f")
            path = os.path.join(ECG_OUTPUTS_TEST, filename)
            plt.plot(np.load(path))
            plt.title("label_f")
            plt.show()"""

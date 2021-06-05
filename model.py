import torch
import os
from ResnetNetwork import *
from torch.autograd import Variable
import math
from HelpFunctions import *

network_save_folder_orig = "./Models"
network_file_name_best_sim = "/best_model_sim"
network_file_name_best_cent = "/criterion_cent"
network_file_name_best_real = "/best_model_real"

delta = 3

ecg_lamda = 1.
cent_lamda = 0.01
hinge_lamda = 0.5

ecg_weight = 1.
cent_weight = 1.
hinge_weight = 1.

include_ecg_loss = True
include_center_loss = True
include_hinge_loss = True

ECG_OUTPUTS_REAL = os.path.join(os.path.dirname(os.path.realpath(__file__)), "ECGOutputsTrainReal")
if not os.path.exists(ECG_OUTPUTS_REAL):
    os.mkdir(ECG_OUTPUTS_REAL)

ECG_OUTPUTS_VAL_REAL = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                               "ECGOutputsValReal")
if not os.path.exists(ECG_OUTPUTS_VAL_REAL):
    os.mkdir(ECG_OUTPUTS_VAL_REAL)

ECG_OUTPUTS_TEST_REAL = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                            "ECGOutputsTestReal")
if not os.path.exists(ECG_OUTPUTS_TEST_REAL):
    os.mkdir(ECG_OUTPUTS_TEST_REAL)

def train_real(resnet_model,
              train_data_loader_real,
              optimizer_model,
              epoch,
              epochs,
              criterion,
              criterion_cent,
              train_loss_ecg_list):

    total_loss_epoch = 0.
    total_loss_ecg = 0.
    total_loss_cent = 0.
    total_loss_hinge = 0.

    for i, batch_features in enumerate(train_data_loader_real):
        optimizer_model.zero_grad()
        batch_for_model = Variable(1000. * batch_features.float().cuda())
        outputs_m, one_before_last_m, outputs_f, one_before_last_f = resnet_model(batch_for_model)
        for j, elem in enumerate(outputs_f):
            path = os.path.join(ECG_OUTPUTS_REAL, "label_ecg" + str(j) + str(i) + str(epoch))
            np.save(path, batch_features[j].cpu().detach().numpy())
            path = os.path.join(ECG_OUTPUTS_REAL, "ecg" + str(j) + str(i) + str(epoch))
            np.save(path, (outputs_m[j] + outputs_f[j]).cpu().detach().numpy() / 1000.)
            path = os.path.join(ECG_OUTPUTS_REAL, "mecg" + str(j) + str(i) + str(epoch))
            np.save(path, (outputs_m[j]).cpu().detach().numpy() / 1000.)
            path = os.path.join(ECG_OUTPUTS_REAL, "fecg" + str(j) + str(i) + str(epoch))
            np.save(path, (outputs_f[j]).cpu().detach().numpy() / 1000.)

        outputs_ecg = outputs_m + outputs_f
        train_loss_ecg = criterion(outputs_ecg, batch_for_model)
        flatten_m, flatten_f = torch.flatten(one_before_last_m, start_dim=1), torch.flatten(one_before_last_f,
                                                                                            start_dim=1)
        hinge_loss = criterion_hinge_loss(one_before_last_m, one_before_last_f, delta)
        batch_size = one_before_last_m.size()[0]
        labels_center_loss = Variable(torch.cat((torch.zeros(batch_size), torch.ones(batch_size))).cuda())
        loss_cent = criterion_cent(torch.cat((flatten_f, flatten_m), 0), labels_center_loss)

        total_loss = ecg_weight * ecg_lamda * train_loss_ecg
        if include_center_loss:
            total_loss += cent_weight * cent_lamda * loss_cent
        if include_hinge_loss:
            total_loss += hinge_weight * hinge_lamda * hinge_loss
        total_loss.backward()
        optimizer_model.step()

        total_loss_ecg += train_loss_ecg.item()

        total_loss_cent += cent_weight * cent_lamda * loss_cent.item()
        total_loss_hinge += hinge_weight * hinge_lamda * hinge_loss.item()
        total_loss_epoch += total_loss.item()
        batch_features, batch_for_model, total_loss, outputs_m, one_before_last_m, \
        outputs_f, one_before_last_f, train_loss_ecg = None, None, None, None, None,\
                                                                         None, None, None


    total_loss_ecg = total_loss_ecg / (len(train_data_loader_real.dataset))
    train_loss_ecg_list.append(total_loss_ecg)

    total_loss_cent = total_loss_cent / (len(train_data_loader_real.dataset))
    total_loss_hinge = total_loss_hinge / (len(train_data_loader_real.dataset))
    total_loss_epoch = total_loss_epoch / (len(train_data_loader_real.dataset))

    print("epoch S : {}/{}  total_loss = {:.8f}".format(epoch + 1, epochs, total_loss_epoch))
    if include_ecg_loss:
        print("loss_ecg = {:.8f} ".format(total_loss_ecg))
    if include_center_loss:
        print("loss_cent = {:.8f} ".format(total_loss_cent))
    if include_hinge_loss:
        print("loss_hinge = {:.8f} ".format(total_loss_hinge))
    print("\n")

    if epoch + 1 == epochs:
        with open("train_loss_last_epoch.txt", 'w') as f:
            f.write("L1ECG = {:.4f},LCent = {:.4f},"
                    "LHinge = {:.4f},LTot = {:.4f}\n".format(total_loss_ecg,
                                                             total_loss_cent,
                                                             total_loss_hinge,
                                                             total_loss_epoch))

def val_real(
        val_data_loader_real,
        resnet_model,
        criterion,
        criterion_cent,
        epoch,
        epochs,
        validation_loss_ecg_list,
        validation_corr_ecg_list,
        best_model_accuracy):

    val_loss_ecg = 0
    val_corr_average = 0
    total_loss_cent = 0

    for i, batch_features in enumerate(val_data_loader_real):
        batch_for_model_val = Variable(1000. * batch_features.float().cuda())
        outputs_m_val, one_before_last_m, outputs_f_val, one_before_last_f = resnet_model(batch_for_model_val)
        val_loss_ecg += criterion(outputs_m_val + outputs_f_val, batch_for_model_val)
        flatten_m, flatten_f = torch.flatten(one_before_last_m, start_dim=1), torch.flatten(one_before_last_f,
                                                                                            start_dim=1)
        batch_size = one_before_last_m.size()[0]
        labels_center_loss = Variable(torch.cat((torch.zeros(batch_size), torch.ones(batch_size))).cuda())
        total_loss_cent += cent_weight * cent_lamda * criterion_cent(torch.cat((flatten_f, flatten_m), 0), labels_center_loss)
        for j, elem in enumerate(outputs_f_val):
            val_corr_average += np.corrcoef((outputs_m_val[j] + outputs_f_val[j]).cpu().detach().numpy(), batch_for_model_val.cpu().detach().numpy()[j])[0][1]
            path = os.path.join(ECG_OUTPUTS_VAL_REAL, "label_ecg" + str(j) + str(i) + str(epoch))
            np.save(path, batch_features[j].cpu().detach().numpy())
            path = os.path.join(ECG_OUTPUTS_VAL_REAL, "ecg" + str(j) + str(i) + str(epoch))
            np.save(path, (outputs_m_val[j] + outputs_f_val[j]).cpu().detach().numpy() / 1000.)
            path = os.path.join(ECG_OUTPUTS_VAL_REAL, "mecg" + str(j) + str(i) + str(epoch))
            np.save(path, (outputs_m_val[j]).cpu().detach().numpy() / 1000.)
            path = os.path.join(ECG_OUTPUTS_VAL_REAL, "fecg" + str(j) + str(i) + str(epoch))
            np.save(path, (outputs_f_val[j]).cpu().detach().numpy() / 1000.)

    val_loss_ecg /= len(val_data_loader_real.dataset)
    val_corr_average /= len(val_data_loader_real.dataset)
    total_loss_cent = total_loss_cent / (len(val_data_loader_real.dataset))

    validation_loss_ecg_list.append(val_loss_ecg.cpu().detach())
    validation_corr_ecg_list.append(val_corr_average)


    if epoch + 1 == epochs:
        with open("val_loss_last_epoch.txt", 'w') as f:
            f.write("LECG = {:.4f},CorrECG = {:.4f}, LossCent\n".format(val_loss_ecg,val_corr_average,total_loss_cent))
            torch.save(resnet_model.state_dict(), str(network_save_folder_orig + 'last_model'))
    if (val_corr_average > best_model_accuracy):
        best_model_accuracy = val_corr_average
        torch.save(resnet_model.state_dict(), str(network_save_folder_orig + network_file_name_best_real))
        print("saving best model")
        with open("best_model_epoch_real.txt", 'w') as f:
            f.write(str(epoch))
    print(
    'Validation: Average loss ECG: {:.4f},Correlation Average ECG: {:.4f})\n'.format(
        val_loss_ecg,val_corr_average))
    return best_model_accuracy,val_loss_ecg

def test(filename_real, test_data_loader_real):

    resnet_model_real = ResNet(1)
    resnet_model_real.load_state_dict(torch.load(filename_real))
    resnet_model_real.eval()
    resnet_model_real.cuda()

    criterion = nn.L1Loss().cuda()

    test_loss_ecg = 0

    with torch.no_grad():
        for i, batch_features in enumerate(test_data_loader_real):
            batch_for_model_test = Variable(1000. * batch_features.float().cuda())
            outputs_m_test_real, _, outputs_f_test_real, _ = resnet_model_real(batch_for_model_test)
            test_loss_ecg += criterion(outputs_m_test_real + outputs_f_test_real, batch_for_model_test)
            for j, elem in enumerate(outputs_f_test_real):
                test_corr_ecg += np.corrcoef((outputs_m_test_real[j] + outputs_f_test_real[j]).cpu().detach().numpy(),
                                                batch_for_model_val.cpu().detach().numpy()[j])[0][1]
                path = os.path.join(ECG_OUTPUTS_TEST_REAL, "label_ecg" + str(j) + str(i))
                np.save(path, batch_features[j].cpu().detach().numpy())
                path = os.path.join(ECG_OUTPUTS_TEST_REAL, "ecg" + str(j) + str(i))
                np.save(path, (outputs_m_test_real[j] + outputs_f_test_real[j]).cpu().detach().numpy() / 1000.)
                path = os.path.join(ECG_OUTPUTS_TEST_REAL, "mecg" + str(j) + str(i))
                np.save(path, (outputs_m_test_real[j]).cpu().detach().numpy() / 1000.)
                path = os.path.join(ECG_OUTPUTS_TEST_REAL, "fecg" + str(j) + str(i))
                np.save(path, (outputs_f_test_real[j]).cpu().detach().numpy() / 1000.)


    test_loss_ecg /= len(test_data_loader_real.dataset)
    test_corr_ecg /= len(test_data_loader_real.dataset)

    return test_loss_ecg,test_corr_ecg
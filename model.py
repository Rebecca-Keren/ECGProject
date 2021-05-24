import torch
import os
from ResnetNetwork import *
from torch.autograd import Variable
import math
from HelpFunctions import *

network_save_folder_orig = "./Models"
network_file_name_last = "/last_model"
network_file_name_best = "/best_model"

BAR_LIST_TRAIN = os.path.join(os.path.dirname(os.path.realpath(__file__)), "BarListTrain")
if not os.path.exists(BAR_LIST_TRAIN):
    os.mkdir(BAR_LIST_TRAIN)

if not os.path.exists(network_save_folder_orig):
    os.mkdir(network_save_folder_orig)

delta = 3

fecg_lamda = 1.
cent_lamda = 0.01
hinge_lamda = 0.5

mecg_weight = 1.
fecg_weight = 1.
cent_weight = 1.
hinge_weight = 1.

include_mecg_loss = True
include_fecg_loss = True
include_center_loss = True
include_hinge_loss = True

ECG_OUTPUTS = os.path.join(os.path.dirname(os.path.realpath(__file__)), "ECGOutputsTrain")
if not os.path.exists(ECG_OUTPUTS):
    os.mkdir(ECG_OUTPUTS)

ECG_OUTPUTS_VAL = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                               "ECGOutputsVal")
if not os.path.exists(ECG_OUTPUTS_VAL):
    os.mkdir(ECG_OUTPUTS_VAL)

ECG_OUTPUTS_TEST = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                            "ECGOutputsTest")
if not os.path.exists(ECG_OUTPUTS_TEST):
    os.mkdir(ECG_OUTPUTS_TEST)

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

def train(resnet_model,
              train_data_loader_sim,
              train_data_loader_real,
              optimizer_model,
              criterion,
              criterion_cent,
              epoch,
              epochs,
              train_loss_f_list,
              train_loss_m_list,
              train_loss_ecg_list,
              train_loss_average_list):

    total_loss_epoch = 0.
    total_loss_m = 0.
    total_loss_f = 0.
    total_loss_ecg = 0.
    total_loss_cent = 0.
    total_loss_hinge = 0.

    real_epoch = False
    train_data_loader = train_data_loader_sim
    list_bar_bad_example_noisetype = [0, 0, 0, 0]
    list_bar_good_example_noisetype = [0, 0, 0, 0]
    list_bar_bad_example_snr = [0, 0, 0, 0, 0]
    list_bar_good_example_snr = [0, 0, 0, 0, 0]
    list_bar_bad_example_snrcase = [[0, 0, 0, 0, 0, 0, 0],[0, 0, 0, 0, 0, 0, 0],
                                     [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0],
                                     [0, 0, 0, 0, 0, 0, 0]]
    list_bar_good_example_snrcase = [[0, 0, 0, 0, 0, 0, 0],[0, 0, 0, 0, 0, 0, 0],
                                     [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0],
                                     [0, 0, 0, 0, 0, 0, 0]]

    if (epoch % 4 == 0):
        real_epoch = True
        train_data_loader = train_data_loader_real
    for i, batch_features in enumerate(train_data_loader):
        optimizer_model.zero_grad()

        if real_epoch:
            batch_for_model = Variable(1000. * batch_features.transpose(1, 2).float().cuda())
        else:
            batch_for_model = Variable(1000. * batch_features[0].transpose(1, 2).float().cuda())
            batch_for_m = Variable(1000. * batch_features[1].transpose(1, 2).float().cuda())
            batch_for_f = Variable(1000. * batch_features[2].transpose(1, 2).float().cuda())
            batch_for_noise_test = batch_features[6].cpu().detach().numpy()
            batch_for_snr_test = batch_features[7].cpu().detach().numpy()
            batch_for_case_test = batch_features[8].cpu().detach().numpy()

        outputs_m, one_before_last_m, outputs_f, one_before_last_f = resnet_model(batch_for_model)

        if epoch + 1 == epochs:
            if not real_epoch:
                for j, elem in enumerate(outputs_f):
                    corr_f = \
                    np.corrcoef(outputs_f.cpu().detach().numpy()[j], batch_for_f.cpu().detach().numpy()[j])[0][1]
                    if (corr_f < 0.4):
                        list_bar_bad_example_noisetype[batch_for_noise_test[j]] += 1
                        list_bar_bad_example_snr[batch_for_snr_test[j]] += 1
                        list_bar_bad_example_snrcase[batch_for_snr_test[j]][batch_for_case_test[j]] += 1
                    else:
                        list_bar_good_example_noisetype[batch_for_noise_test[j]] += 1
                        list_bar_good_example_snr[batch_for_snr_test[j]] += 1
                        list_bar_good_example_snrcase[batch_for_snr_test[j]][batch_for_case_test[j]] += 1

                path_bar = os.path.join(BAR_LIST_TRAIN, "list_bar_good_example_noisetype")
                np.save(path_bar, np.array(list_bar_good_example_noisetype))
                path_bar = os.path.join(BAR_LIST_TRAIN, "list_bar_bad_example_noisetype")
                np.save(path_bar, np.array(list_bar_bad_example_noisetype))
                path_bar = os.path.join(BAR_LIST_TRAIN, "list_bar_good_example_snr")
                np.save(path_bar, np.array(list_bar_good_example_snr))
                path_bar = os.path.join(BAR_LIST_TRAIN, "list_bar_bad_example_snr")
                np.save(path_bar, np.array(list_bar_bad_example_snr))
                path_bar = os.path.join(BAR_LIST_TRAIN, "list_bar_good_example_snrcase")
                np.save(path_bar, np.array(list_bar_good_example_snrcase))
                path_bar = os.path.join(BAR_LIST_TRAIN, "list_bar_bad_example_snrcase")
                np.save(path_bar, np.array(list_bar_bad_example_snrcase))

                path = os.path.join(ECG_OUTPUTS, "ecg_all" + str(i))
                np.save(path, batch_features[0][0].cpu().detach().numpy()[:, 0])
                path = os.path.join(ECG_OUTPUTS, "label_m" + str(i))
                np.save(path, batch_features[1][0].cpu().detach().numpy()[:, 0])
                path = os.path.join(ECG_OUTPUTS, "label_f" + str(i))
                np.save(path, batch_features[2][0].cpu().detach().numpy()[:, 0])
                path = os.path.join(ECG_OUTPUTS, "fecg" + str(i))
                np.save(path, outputs_f[0][0].cpu().detach().numpy() / 1000.)
                path = os.path.join(ECG_OUTPUTS, "mecg" + str(i))
                np.save(path, outputs_m[0][0].cpu().detach().numpy() / 1000.)

        if real_epoch:
            path = os.path.join(ECG_OUTPUTS_REAL, "label_ecg" + str(i))
            np.save(path, batch_features[0].cpu().detach().numpy()[:, 0])
            path = os.path.join(ECG_OUTPUTS_REAL, "ecg" + str(i))
            np.save(path, (outputs_m[0]+outputs_f[0]).cpu().detach().numpy() / 1000.)

        if not real_epoch:
            # COST(M,M^)
            train_loss_mecg = criterion(outputs_m, batch_for_m)

            # COST(F,F^)
            train_loss_fecg = criterion(outputs_f, batch_for_f)

        else:
            outputs_ecg = outputs_m + outputs_f
            train_loss_ecg = criterion(outputs_ecg, batch_for_model)

        flatten_m, flatten_f = torch.flatten(one_before_last_m, start_dim=1), torch.flatten(one_before_last_f,
                                                                                            start_dim=1)
        hinge_loss = criterion_hinge_loss(one_before_last_m, one_before_last_f, delta)
        batch_size = one_before_last_m.size()[0]
        labels_center_loss = Variable(torch.cat((torch.zeros(batch_size), torch.ones(batch_size))).cuda())
        loss_cent = criterion_cent(torch.cat((flatten_f, flatten_m), 0), labels_center_loss)

        if not real_epoch: #TODO add when real data
            total_loss = mecg_weight * train_loss_mecg + fecg_weight * fecg_lamda * train_loss_fecg
            if include_center_loss:
                total_loss += cent_weight * cent_lamda * loss_cent
            if include_hinge_loss:
                total_loss += hinge_weight * hinge_lamda * hinge_loss
        else:
            total_loss = train_loss_ecg #TODO: check lamda for ecg and change loss ecg
            if include_center_loss:
                total_loss += cent_weight * cent_lamda * loss_cent
            if include_hinge_loss:
                total_loss += hinge_weight * hinge_lamda * hinge_loss
        total_loss.backward()
        optimizer_model.step()

        if not real_epoch:
            total_loss_m += mecg_weight * train_loss_mecg.item()
            total_loss_f += fecg_weight * fecg_lamda * train_loss_fecg.item()


        else:
           total_loss_ecg += train_loss_ecg.item()

        total_loss_cent += cent_weight * cent_lamda * loss_cent.item()
        total_loss_hinge += hinge_weight * hinge_lamda * hinge_loss.item()
        total_loss_epoch += total_loss.item()
        batch_features, batch_for_model, batch_for_m, batch_for_f, total_loss, outputs_m, one_before_last_m, \
        outputs_f, one_before_last_f, train_loss_mecg, train_loss_fecg = None, None, None, None, None, None, None, \
                                                                         None, None, None, None

    # compute the epoch training loss
    if not real_epoch:
        total_loss_m = total_loss_m / (len(train_data_loader.dataset))
        total_loss_f = total_loss_f / (len(train_data_loader.dataset))
        train_loss_f_list.append(total_loss_f)
        train_loss_m_list.append(total_loss_m)
        train_loss_average_list.append((total_loss_m+total_loss_f)/2)

    else:
        total_loss_ecg = total_loss_ecg / (len(train_data_loader.dataset))
        train_loss_ecg_list.append(total_loss_ecg)

    total_loss_cent = total_loss_cent / (len(train_data_loader.dataset))
    total_loss_hinge = total_loss_hinge / (len(train_data_loader.dataset))
    total_loss_epoch = total_loss_epoch / (len(train_data_loader.dataset))

    # display the epoch training loss
    if not real_epoch:
        print("epoch S : {}/{}  total_loss = {:.8f}".format(epoch + 1, epochs, total_loss_epoch))
        if include_mecg_loss:
            print("loss_mecg = {:.8f} ".format(total_loss_m))
        if include_fecg_loss:
            print("loss_fecg = {:.8f} ".format(total_loss_f))
        if include_center_loss:
            print("loss_cent = {:.8f} ".format(total_loss_cent))
        if include_hinge_loss:
            print("loss_hinge = {:.8f} ".format(total_loss_hinge))
        print("\n")

    else:
        print("epoch S : {}/{}  total_loss = {:.8f}".format(epoch + 1, epochs, total_loss_epoch))
        if include_mecg_loss:
            print("loss_ecg = {:.8f} ".format(total_loss_ecg))
        if include_center_loss:
            print("loss_cent = {:.8f} ".format(total_loss_cent))
        if include_hinge_loss:
            print("loss_hinge = {:.8f} ".format(total_loss_hinge))
        print("\n")

    if epoch + 1 == epochs:
        with open("train_loss_last_epoch.txt", 'w') as f:
            f.write("L1 M = {:.4f},L1 F= {:.4f},L1ECG = {:.4f},LCent = {:.4f},"
                    "LHinge = {:.4f},LTot = {:.4f}\n".format(total_loss_m,
                                                            total_loss_f,
                                                            total_loss_ecg,
                                                            total_loss_cent,
                                                            total_loss_hinge,
                                                            total_loss_epoch))

def val(val_data_loader_sim,
        val_data_loader_real,
        resnet_model,
        criterion,
        epoch,
        epochs,
        validation_loss_m_list,
        validation_loss_f_list,
        validation_loss_ecg_list,
        validation_loss_average_list,
        validation_corr_m_list,
        validation_corr_f_list,
        best_model_accuracy):

    val_loss_m = 0
    val_loss_f = 0
    val_loss_ecg = 0
    val_corr_m = 0
    val_corr_f = 0
    real_epoch = False
    val_data_loader = val_data_loader_sim

    if(epoch % 4 == 0):
        real_epoch = True
        val_data_loader = val_data_loader_real
    with torch.no_grad():
        for i, batch_features in enumerate(val_data_loader):
            if real_epoch:
                batch_for_model_val = Variable(1000. * batch_features.transpose(1, 2).float().cuda())
            else:
                batch_for_model_val = Variable(1000. * batch_features[0].transpose(1, 2).float().cuda())
                batch_for_m_val = Variable(1000. * batch_features[1].transpose(1, 2).float().cuda())
                batch_for_f_val = Variable(1000. * batch_features[2].transpose(1, 2).float().cuda())
            outputs_m_val, _, outputs_f_val, _ = resnet_model(batch_for_model_val)

            if not real_epoch:
                val_loss_m += criterion(outputs_m_val, batch_for_m_val)
                val_loss_f += (criterion(outputs_f_val, batch_for_f_val)* fecg_weight)
                for j, elem in enumerate(outputs_m_val):
                    val_corr_m += \
                    np.corrcoef(outputs_m_val.cpu().detach().numpy()[j], batch_for_m_val.cpu().detach().numpy()[j])[0][1]
                    val_corr_f += \
                    np.corrcoef(outputs_f_val.cpu().detach().numpy()[j], batch_for_f_val.cpu().detach().numpy()[j])[0][1]
            else:
                val_loss_ecg += criterion(outputs_m_val+outputs_f_val, batch_for_model_val)


            if epoch + 1 == epochs:
                if not real_epoch:
                    path = os.path.join(ECG_OUTPUTS_VAL, "ecg_all" + str(i))
                    np.save(path, batch_features[0][0].cpu().detach().numpy()[:, 0])
                    path = os.path.join(ECG_OUTPUTS_VAL, "label_m" + str(i))
                    np.save(path, batch_features[1][0].cpu().detach().numpy()[:, 0])
                    path = os.path.join(ECG_OUTPUTS_VAL, "label_f" + str(i))
                    np.save(path, batch_features[2][0].cpu().detach().numpy()[:, 0])
                    path = os.path.join(ECG_OUTPUTS_VAL, "fecg" + str(i))
                    np.save(path, outputs_f_val[0][0].cpu().detach().numpy() / 1000.)
                    path = os.path.join(ECG_OUTPUTS_VAL, "mecg" + str(i))
                    np.save(path, outputs_m_val[0][0].cpu().detach().numpy() / 1000.)

            if real_epoch:
                path = os.path.join(ECG_OUTPUTS_VAL_REAL, "label_ecg" + str(i))
                np.save(path, batch_features[0].cpu().detach().numpy()[:, 0])
                path = os.path.join(ECG_OUTPUTS_VAL_REAL, "ecg" + str(i))
                np.save(path, (outputs_m_val[0] + outputs_f_val[0]).cpu().detach().numpy() / 1000.)

    if not real_epoch:
        val_loss_m /= len(val_data_loader.dataset)
        val_loss_f /= len(val_data_loader.dataset)
        val_corr_m /= len(val_data_loader.dataset)
        val_corr_f /= len(val_data_loader.dataset)
        val_corr_average = (val_corr_m + val_corr_f) / 2
        val_loss_average = (val_loss_m + val_loss_f) / 2
        # saving validation losses
        validation_loss_m_list.append(val_loss_m.cpu().detach())
        validation_loss_f_list.append(val_loss_f.cpu().detach())
        validation_loss_average_list.append(val_loss_average.cpu().detach())
        validation_corr_m_list.append(val_corr_m)
        validation_corr_f_list.append(val_corr_f)

    else:
        val_loss_ecg /= len(val_data_loader.dataset)
        validation_loss_ecg_list.append(val_loss_ecg.cpu().detach())

    if epoch + 1 == epochs:
        with open("val_loss_last_epoch.txt", 'w') as f:
            f.write("L1 M = {:.4f},L1 F= {:.4f},LAvg = {:.4f},LECG = {:.4f},"
                    "CorrM = {:.4f},CorrF = {:.4f}, CorrAvg = {:.4f}\n".format(val_loss_m,
                                                            val_loss_f,
                                                            val_loss_average,
                                                            val_loss_ecg,
                                                            val_corr_m,
                                                            val_corr_f,
                                                            val_corr_average))
    # saving last model

    torch.save(resnet_model.state_dict(), str(network_save_folder_orig + network_file_name_last))
    # saving best model
    if not real_epoch: #TODO ASK
        if (val_corr_average < best_model_accuracy):
            best_model_accuracy = val_corr_average
            torch.save(resnet_model.state_dict(), str(network_save_folder_orig + network_file_name_best))
            print("saving best model")
            with open("best_model_epoch.txt", 'w') as f:
                f.write(str(epoch))
        print(
        'Validation: Average loss M: {:.4f}, Average Loss F: {:.4f}, Average Loss M+F: {:.4f}, Correlation M: {:.4f},Correlation F: {:.4f},Correlation Average: {:.4f})\n'.format(
            val_loss_m, val_loss_f, val_loss_average, val_corr_m, val_corr_f, val_corr_average))
    else:
        print('Validation: Average loss ECG: {:.4f}\n'.format(val_loss_ecg))
    return best_model_accuracy,val_loss_average


def test(filename,test_data_loader_sim, test_data_loader_real):
    resnet_model = ResNet(1)
    resnet_model.load_state_dict(torch.load(filename))
    resnet_model.eval()

    resnet_model.cuda()

    criterion = nn.L1Loss().cuda()

    test_loss_m = 0
    test_loss_f = 0
    test_loss_ecg = 0
    test_corr_m = 0
    test_corr_f = 0

    list_bar_bad_example_noisetype = [0, 0, 0, 0]
    list_bar_good_example_noisetype = [0, 0, 0, 0]
    list_bar_bad_example_snr = [0, 0, 0, 0, 0]
    list_bar_good_example_snr = [0, 0, 0, 0, 0]
    list_bar_bad_example_snrcase = [[0, 0, 0, 0, 0, 0, 0],[0, 0, 0, 0, 0, 0, 0],
                                     [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0],
                                     [0, 0, 0, 0, 0, 0, 0]]
    list_bar_good_example_snrcase = [[0, 0, 0, 0, 0, 0, 0],[0, 0, 0, 0, 0, 0, 0],
                                     [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0],
                                     [0, 0, 0, 0, 0, 0, 0]]

    with torch.no_grad():
        for i, batch_features in enumerate(test_data_loader_sim):
            batch_for_model_test = Variable(1000. * batch_features[0].transpose(1, 2).float().cuda())
            batch_for_m_test = Variable(1000. * batch_features[1].transpose(1, 2).float().cuda())
            batch_for_f_test = Variable(1000. * batch_features[2].transpose(1, 2).float().cuda())
            batch_for_noise_test = batch_features[6].cpu().detach().numpy()
            batch_for_snr_test = batch_features[7].cpu().detach().numpy()
            batch_for_case_test = batch_features[8].cpu().detach().numpy()
            outputs_m_test_sim, _, outputs_f_test_sim, _ = resnet_model(batch_for_model_test)
            test_loss_m += criterion(outputs_m_test_sim, batch_for_m_test)
            test_loss_f += criterion(outputs_f_test_sim, batch_for_f_test)
            for j, elem in enumerate(outputs_m_test_sim):
                corr_m = np.corrcoef(outputs_m_test_sim.cpu().detach().numpy()[j], batch_for_m_test.cpu().detach().numpy()[j])[0][1]
                test_corr_m += corr_m
                corr_f = np.corrcoef(outputs_f_test_sim.cpu().detach().numpy()[j], batch_for_f_test.cpu().detach().numpy()[j])[0][1]
                test_corr_f += corr_f
                if(corr_f < 0.4):
                    list_bar_bad_example_noisetype[batch_for_noise_test[j]] += 1
                    list_bar_bad_example_snr[batch_for_snr_test[j]] += 1
                    list_bar_bad_example_snrcase[batch_for_snr_test[j]][batch_for_case_test[j]] += 1
                else:
                    list_bar_good_example_noisetype[batch_for_noise_test[j]] += 1
                    list_bar_good_example_snr[batch_for_snr_test[j]] += 1
                    list_bar_good_example_snrcase[batch_for_snr_test[j]][batch_for_case_test[j]] += 1

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

        for i, batch_features in enumerate(test_data_loader_real):
            batch_for_model_test = Variable(1000. * batch_features[0].transpose(1, 2).float().cuda())
            outputs_m_test_real, _, outputs_f_test_real, _ = resnet_model(batch_for_model_test)
            test_loss_m += criterion(outputs_m_test_real, batch_for_m_test)
            test_loss_f += criterion(outputs_f_test_real, batch_for_f_test)

            path = os.path.join(ECG_OUTPUTS_TEST_REAL, "label_ecg" + str(i))
            np.save(path, batch_features[0].cpu().detach().numpy()[:, 0])
            path = os.path.join(ECG_OUTPUTS_TEST_REAL, "ecg" + str(i))
            np.save(path, (outputs_m_test_real[0] + outputs_f_val[0]).cpu().detach().numpy() / 1000.)

    test_loss_m /= len(test_data_loader_sim.dataset)
    test_loss_f /= len(test_data_loader_sim.dataset)
    test_loss_ecg /= len(test_data_loader_sim.dataset)
    test_loss_average = (test_loss_m + test_loss_f) / 2
    test_corr_m /= len(test_data_loader_sim.dataset)
    test_corr_f /= len(test_data_loader_sim.dataset)
    test_corr_average = (test_corr_m + test_corr_f) / 2

    return test_loss_m, test_loss_f,test_loss_ecg, test_loss_average, test_corr_m, test_corr_f, test_corr_average,\
           list_bar_good_example_noisetype,list_bar_bad_example_noisetype, \
            list_bar_good_example_snr,list_bar_bad_example_snr, \
            list_bar_good_example_snrcase,list_bar_bad_example_snrcase
import torch
import os
from ResnetNetwork import *
from torch.autograd import Variable

MODELS = os.path.join(os.path.dirname(os.path.realpath(__file__)), "Models")

network_save_folder_orig = "./Models"
network_file_name_last = "/last_model"
network_file_name_best = "/best_model"

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

def train(resnet_model,
              train_data_loader_sim,
              optimizer_model,
              optimizer_centloss,
              criterion,
              criterion_cent,
              epoch,
              epochs,
              train_loss_f_list,
              train_loss_m_list,
              train_loss_average_list,
              dataset_size):

    ECG_OUTPUTS = os.path.join(os.path.dirname(os.path.realpath(__file__)), "ECGOutputs" + str(dataset_size))
    if not os.path.exists(ECG_OUTPUTS):
        os.mkdir(ECG_OUTPUTS)

    total_loss_epoch = 0.
    total_loss_m = 0.
    total_loss_f = 0.
    #total_loss_ecg = 0. #TODO add when real data
    total_loss_cent = 0.
    total_loss_hinge = 0.

    # real_epoch = False #TODO add when real data
    for i, batch_features in enumerate(train_data_loader_sim):
        optimizer_model.zero_grad()
        optimizer_centloss.zero_grad()

        batch_for_model = Variable(1000. * batch_features[0].transpose(1, 2).float().cuda())
        batch_for_m = Variable(1000. * batch_features[1].transpose(1, 2).float().cuda())
        batch_for_f = Variable(1000. * batch_features[2].transpose(1, 2).float().cuda())

        outputs_m, one_before_last_m, outputs_f, one_before_last_f = resnet_model(batch_for_model)

        if epoch + 1 == epochs:
            if not os.path.exists(ECG_OUTPUTS):
                os.mkdir(ECG_OUTPUTS)
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

        # if not real_epoch: #TODO add when real data
        # COST(M,M^)
        train_loss_mecg = criterion(outputs_m, batch_for_m)

        # COST(F,F^)
        train_loss_fecg = criterion(outputs_f, batch_for_f)

        # else: #TODO add when real data
        #   outputs_m += outputs_f
        #   train_loss_ecg = criterion(outputs_m, batch_for_model)

        flatten_m, flatten_f = torch.flatten(one_before_last_m, start_dim=1), torch.flatten(one_before_last_f,
                                                                                            start_dim=1)
        hinge_loss = criterion_hinge_loss(one_before_last_m, one_before_last_f, delta)
        batch_size = one_before_last_m.size()[0]
        labels_center_loss = Variable(torch.cat((torch.zeros(batch_size), torch.ones(batch_size))).cuda())
        loss_cent = criterion_cent(torch.cat((flatten_f, flatten_m), 0), labels_center_loss)

        # if not real_epoch: #TODO add when real data
        total_loss = mecg_weight * train_loss_mecg + fecg_weight * fecg_lamda * train_loss_fecg
        if include_center_loss:
            total_loss += cent_weight * cent_lamda * loss_cent
        if include_hinge_loss:
            total_loss += hinge_weight * hinge_lamda * hinge_loss
        # else: #TODO add when real data
        #     total_loss = train_loss_ecg + cent_weight*cent_lamda*loss_cent + hinge_weight*hinge_lamda*hinge_loss #TODO: check lamda for ecg and change loss ecg

        total_loss.backward()
        optimizer_model.step()
        optimizer_centloss.step()

        # if not real_epoch: #TODO add when real data
        total_loss_m += mecg_weight * train_loss_mecg.item()
        total_loss_f += fecg_weight * fecg_lamda * train_loss_fecg.item()

        # else: #TODO add when real data
        #   total_loss_ecg += train_loss_ecg.item()

        total_loss_cent += cent_weight * cent_lamda * loss_cent.item()
        total_loss_hinge += hinge_weight * hinge_lamda * hinge_loss.item()
        total_loss_epoch += total_loss.item()
        batch_features, batch_for_model, batch_for_m, batch_for_f, total_loss, outputs_m, one_before_last_m, \
        outputs_f, one_before_last_f, train_loss_mecg, train_loss_fecg = None, None, None, None, None, None, None, \
                                                                         None, None, None, None

        # compute the epoch training loss
        # if not real_epoch: #TODO add when real data
        total_loss_m = total_loss_m / (len(train_data_loader_sim))
        total_loss_f = total_loss_f / (len(train_data_loader_sim))
        train_loss_f_list.append(total_loss_f)
        train_loss_m_list.append(total_loss_m)
        train_loss_average_list.append((total_loss_m+total_loss_f)/2)

    # else: #TODO add when real data
    #    total_loss_ecg = total_loss_ecg / (len(train_data_loader_sim))

    total_loss_cent = total_loss_cent / (len(train_data_loader_sim))
    total_loss_hinge = total_loss_hinge / (len(train_data_loader_sim))
    total_loss_epoch = total_loss_epoch / (len(train_data_loader_sim))

    # display the epoch training loss
    # if not real_epoch: #TODO add when real data
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

    # else: #TODO add when real data
    #    print("epoch R : {}/{}, total_loss = {:.8f}, loss_ecg = {:.8f}, loss_cent = {:.8f}, loss_hinge = {:.8f}".format(
    #            epoch + 1, epochs, total_loss_epoch, total_loss_ecg, total_loss_cent, total_loss_hinge))


def val(val_data_loader_sim,
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
        dataset_size):
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
            for i, elem in enumerate(outputs_m_test):
                val_corr_m += \
                np.corrcoef(outputs_m_test.cpu().detach().numpy()[i], batch_for_m_val.cpu().detach().numpy()[i])[0][1]
                val_corr_f += \
                np.corrcoef(outputs_f_test.cpu().detach().numpy()[i], batch_for_f_val.cpu().detach().numpy()[i])[0][1]
    val_loss_m /= len(val_data_loader_sim.dataset)
    val_loss_f /= len(val_data_loader_sim.dataset)
    val_corr_m /= len(val_data_loader_sim.dataset)
    val_corr_f /= len(val_data_loader_sim.dataset)
    val_corr_average = (val_corr_m + val_corr_f) / 2
    val_loss_average = (val_loss_m + val_loss_f) / 2

    # saving validation losses
    validation_loss_m_list.append(val_loss_m.cpu().detach())
    validation_loss_f_list.append(val_loss_f.cpu().detach())
    validation_loss_average_list.append(val_loss_average.cpu().detach())
    validation_corr_m_list.append(val_corr_m)
    validation_corr_f_list.append(val_corr_f)

    # saving last model
    network_save_folder = network_save_folder_orig + str(dataset_size)
    if not os.path.exists(network_save_folder):
        os.mkdir(network_save_folder)
    torch.save(resnet_model.state_dict(), str(network_save_folder + network_file_name_last))
    # saving best model
    if (val_loss_average < best_model_accuracy):
        best_model_accuracy = val_loss_average
        torch.save(resnet_model.state_dict(), str(network_save_folder + network_file_name_best))
        print("saving best model")

    print(
        'Validation: Average loss M: {:.4f}, Average Loss F: {:.4f}, Average Loss M+F: {:.4f}, Correlation M: {:.4f},Correlation F: {:.4f},Correlation Average: {:.4f})\n'.format(
            val_loss_m, val_loss_f, val_loss_average, val_corr_m, val_corr_f, val_corr_average))

    ECG_OUTPUTS_VAL = os.path.join(os.path.dirname(os.path.realpath(__file__)), "ECGOutputsVal" + str(dataset_size))
    if not os.path.exists(ECG_OUTPUTS_VAL):
        os.mkdir(ECG_OUTPUTS_VAL)

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

def test(filename,test_data_loader_sim,dataset_size):
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
    test_loss_average = (test_loss_m + test_loss_f) / 2

    ECG_OUTPUTS_TEST = os.path.join(os.path.dirname(os.path.realpath(__file__)), "ECGOutputsTest" + str(dataset_size))
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
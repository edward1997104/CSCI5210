import torch
import torch.nn as nn
import torch.utils.data as Data
import numpy as np
import math
import os
from util.debugger import MyDebugger
from torch.optim import Adam, Optimizer
from inputs import config
from util.points_util import write_point_cloud
from metrics.evaluation_metrics import CD_loss_avg, emd_approx
import h5py

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open('./data/shape_names.txt') as f:
    label_names = f.readlines()

def train_model(
          model : nn.Module,
          loss_fn,
          debugger: MyDebugger,
          data_train : torch.Tensor,
          data_test : torch.Tensor,
          label_test : np.array,
          data_query : torch.Tensor,
          label_query : np.array,
          lr : float = 1e-3 ,
          batch_size  : int = 32,
          training_epoch : int =200 ,
          model_saving_epoch : int = 20,
          data_loading_workers : int = 4,
          optimizer_type = Adam,
          tolerance_epoch = 5):
    print("Training Start!!!", flush=True)

    #### optimizer
    optimizer =  optimizer_type(model.parameters(), lr = lr)

    ###### REFERENCE CODE from https://morvanzhou.github.io/tutorials/machine-learning/torch/3-05-train-on-batch/

    torch_dataset = Data.TensorDataset(data_train, data_train)

    loader = Data.DataLoader(
        dataset=torch_dataset,  # torch TensorDataset format
        batch_size=batch_size,  # mini batch size
        shuffle=True,
        num_workers= data_loading_workers,
    )

    #### for early stopping
    min_testing_loss = float("inf")
    tolerance_cnt = 0

    ##### different folders
    save_dir_models = debugger.file_path('models')
    if not os.path.isdir(save_dir_models):
        os.mkdir(save_dir_models)

    save_dir_reconstruction_base = debugger.file_path('reconstruction')
    if not os.path.isdir(save_dir_reconstruction_base):
        os.mkdir(save_dir_reconstruction_base)

    save_dir_query_base = debugger.file_path('query')
    if not os.path.isdir(save_dir_query_base):
        os.mkdir(save_dir_query_base)

    for epoch in range(training_epoch):

        training_losses = []
        for step, (batch_x, batch_y) in enumerate(loader):

            #### training
            batch_x = batch_x.to(device)
            optimizer.zero_grad()
            batch_x_reconstructed = model(batch_x)
            loss = loss_fn(batch_x_reconstructed, batch_x)
            loss.backward()
            training_losses.append(loss.cpu().detach().numpy())
            optimizer.step()

            # print(f"loss for epoch {epoch} stpe {step} : {loss.item()}")

        print(f"training loss for epoch {epoch} : {np.mean(training_losses)}")

        ### testing losss
        testing_loss, CD_loss, EMD_loss = calculate_loss(data_test, model, loss_fn)
        print(f"testing loss for epoch {epoch} : {testing_loss} {CD_loss} {EMD_loss}")

        if testing_loss < min_testing_loss:

            ### reset tolerance
            min_testing_loss = testing_loss
            tolerance_cnt = 0

            ### save reconstruction on testing data
            save_dir_reconstruction = os.path.join(save_dir_reconstruction_base, f'epoch_{epoch}')
            os.mkdir(save_dir_reconstruction)
            save_reconstructed_point(data_test, model, save_dir_reconstruction, label_test)

            torch.save(model, os.path.join(save_dir_models, f'{epoch}_model.pth'))

            ### get query results
            save_dir_query = os.path.join(save_dir_query_base, f'epoch_{epoch}')
            os.mkdir(save_dir_query)
            query_acc = caculate_query_acc(input_shapes = data_query,
                                           data_matrix = data_test,
                                           top_k = config.top_k,
                                           model = model,
                                           input_shapes_labels = label_query,
                                           data_labels = label_test,
                                           save_base = save_dir_query)

            print(f"average query acc : {query_acc}")
        else:
            tolerance_cnt += 1

            #### end the training if no more improvement
            if tolerance_cnt == tolerance_epoch:
                break






    print("Training Done!!!")

def query_shape(input_shape: torch.Tensor, database_codes, top_k, model):

    input_shape = input_shape.unsqueeze(0)
    input_code = model[0](input_shape)[0]
    cos_distances = torch.mv(database_codes, input_code)

    ordered_indices = torch.argsort(cos_distances, descending = True)
    return ordered_indices[:top_k]

def caculate_query_acc(input_shapes : torch.Tensor, data_matrix : torch.Tensor,
                       top_k : int, model : torch.nn.Module, input_shapes_labels : np.array,
                       data_labels : np.array, save_base : str):
    ## clean up
    torch.cuda.empty_cache()
    accs = []

    ### code matrix
    batch_num = int(math.ceil(data_matrix.size(0) / config.batch_size))

    matrix_codes = []
    for i in range(batch_num):
        data = data_matrix[i*config.batch_size:(i+1)*config.batch_size]
        code = model[0](data).detach()
        matrix_codes.append(code)

    matrix_codes = torch.cat(matrix_codes, dim = 0)

    for i in range(input_shapes.size(0)):
        queried_indices = query_shape(input_shapes[i], matrix_codes, top_k, model)
        acc = np.mean([input_shapes_labels[i] == data_labels[idx] for idx in queried_indices])
        accs.append(acc)

        save_dir = os.path.join(save_base, f'{i}_{label_names[input_shapes_labels[i]]}')
        os.mkdir(save_dir)
        for idx in queried_indices:
            write_point_cloud(data_matrix[idx], os.path.join(save_dir, f'{idx}_{label_names[data_labels[idx]]}.obj'))

    return np.mean(accs)

def calculate_loss(X, model, loss_fn):

    ### code matrix
    batch_num = int(math.ceil(X.size(0) / config.batch_size))
    X_reconstructed = []

    EMD_losses = []
    for i in range(batch_num):
        X_data = X[i*config.batch_size:(i+1)*config.batch_size]
        X_temp = model(X_data).detach()
        X_reconstructed.append(X_temp)

        ### EMD loss
        EMD_loss = torch.mean(emd_approx(X_temp, X_data)).detach().cpu().numpy()
        EMD_losses.append(EMD_loss)

    X_reconstructed = torch.cat(X_reconstructed, dim = 0)

    testing_loss = loss_fn(X_reconstructed, X).detach().cpu().numpy()
    torch.cuda.empty_cache()
    CD_loss = CD_loss_avg(X_reconstructed, X).detach().cpu().numpy()

    return testing_loss, CD_loss, np.mean(EMD_losses)

def save_reconstructed_point(X, model, save_dir, label):
    batch_num = int(math.ceil(X.size(0) / config.batch_size))
    for i in range(batch_num):
        X_temp = model(X[i*config.batch_size:(i+1)*config.batch_size]).detach().cpu().numpy()
        for j in range(X_temp.shape[0]):
            write_point_cloud(X_temp[0], os.path.join(save_dir, f'{i*config.batch_size+j}_{label_names[label[i*config.batch_size+j]]}.obj'))

if __name__ == "__main__":

    debugger = MyDebugger('training_points_autoencoder', save_print_to_file = config.save_print_to_file)

    ### training data
    f = h5py.File('./data/train_data.h5')
    data_train = f['data'][:]  ### [9840, 2048, 3]
    X = torch.from_numpy(data_train).float()

    #### testing data + query data
    f = h5py.File('./data/test_data.h5')
    data_test = f['data'][:]  ###

    for i in range(data_test.shape[0]):
        write_point_cloud(data_test[i], debugger.file_path(f'testing_{i}.obj'))
    data_test = torch.from_numpy(data_test).float().to(device)
    label_test = np.squeeze(f['label'])

    f = h5py.File('./data/query_data.h5')
    data_query = f['data'][:]  ###
    data_query = torch.from_numpy(data_query).float().to(device)
    label_query =  np.squeeze(f['label'])
    f.close()

    model = config.current_model.to(device)
    loss_fn = config.loss_fn

    train_model(model = model,
                debugger = debugger,
                loss_fn = loss_fn,
                data_train = X,
                data_test = data_test,
                label_test = label_test,
                data_query = data_query,
                label_query = label_query,
                model_saving_epoch = config.model_saving_epoch,
                training_epoch = config.training_epoch,
                batch_size = config.batch_size,
                tolerance_epoch = config.tolerance_epoch
                )

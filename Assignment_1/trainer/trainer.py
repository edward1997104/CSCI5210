import torch
import torch.nn as nn
import torch.utils.data as Data
import numpy as np
import math
import os
from util.debugger import MyDebugger
from torch.optim import Adam, Optimizer
from inputs import config
import h5py

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
          optimizer_type = Adam):
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

        print(f"loss for epoch {epoch} : {np.mean(training_losses)}")

        if (epoch+1) % model_saving_epoch == 0:
            save_dir = debugger.file_path('models')
            if not os.path.isdir(save_dir):
                os.mkdir(save_dir)
            torch.save(model, os.path.join(save_dir, f'{epoch}_model.pth'))

            query_acc = caculate_query_acc(input_shapes = data_query,
                                           data_matrix = data_test,
                                           top_k = config.top_k,
                                           model = model,
                                           input_shapes_labels = label_query,
                                           data_labels = label_test)

            print(f"average query acc : {query_acc}")






    print("Training Done!!!")

def query_shape(input_shape: torch.Tensor, database_codes, top_k, model):

    input_shape = input_shape.unsqueeze(0)
    input_code = model[0](input_shape)[0]
    cos_distances = torch.mv(database_codes, input_code)

    ordered_indices = torch.argsort(cos_distances, descending = True)
    return ordered_indices[:top_k]

def caculate_query_acc(input_shapes : torch.Tensor, data_matrix : torch.Tensor,
                       top_k : int, model : torch.nn.Module, input_shapes_labels : np.array,
                       data_labels : np.array):

    accs = []

    ### code matrix
    batch_num = int(math.ceil(data_matrix.size(0) / config.batch_size))

    matrix_codes = []
    for i in range(batch_num):
        data = data_matrix[i*config.batch_size:(i+1)*config.batch_size]
        code = model[0](data)
        matrix_codes.append(code)

    matrix_codes = torch.stack(matrix_codes, dim = 0)

    for i in range(input_shapes.size(0)):
        queried_indices = query_shape(input_shapes[i], matrix_codes, top_k, model)
        acc = np.mean([input_shapes_labels[i] == data_labels[idx] for idx in queried_indices])
        accs.append(acc)

    return np.mean(accs)
if __name__ == "__main__":

    debugger = MyDebugger('training_points_autoencoder', save_print_to_file = config.save_print_to_file)

    ### training data
    f = h5py.File('./data/train_data.h5')
    data_train = f['data'][:]  ### [9840, 2048, 3]
    X = torch.from_numpy(data_train).float()

    #### testing data + query data
    f = h5py.File('./data/test_data.h5')
    data_test = f['data'][:]  ###
    data_test = torch.from_numpy(data_test).float().to(device)
    label_test = f['label']

    f = h5py.File('./data/query_data.h5')
    data_query = f['data'][:]  ###
    data_query = torch.from_numpy(data_query).float().to(device)
    label_query = f['label']

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
                batch_size = config.batch_size
                )

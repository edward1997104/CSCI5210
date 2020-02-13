import torch
import torch.nn as nn
import torch.utils.data as Data
import numpy as np
import os
from util.debugger import MyDebugger
from torch.optim import Adam, Optimizer
from inputs import config
import h5py

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_model(
          model : nn.Module,
          debugger: MyDebugger,
          X : torch.Tensor,
          lr : float = 5e-3 ,
          batch_size  : int = 32,
          training_epoch : int =200 ,
          model_saving_epoch : int = 20,
          data_loading_workers : int = 4,
          optimizer_type = Adam):
    print("Training Start!!!", flush=True)

    #### optimizer
    optimizer =  optimizer_type(model.parameters(), lr = lr)

    loss_fn = nn.MSELoss(reduce=True, size_average=True)

    ###### REFERENCE CODE from https://morvanzhou.github.io/tutorials/machine-learning/torch/3-05-train-on-batch/

    torch_dataset = Data.TensorDataset(X, X)

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

            print(f"loss for epoch {epoch} stpe {step} : {loss.item()}")

        print(f"loss for epoch {epoch} : {np.mean(training_losses)}")

        if epoch % model_saving_epoch == 0:
            save_dir = debugger.file_path('models')
            if not os.path.isdir(save_dir):
                os.mkdir(save_dir)
            torch.save(model, os.path.join(save_dir, f'{epoch}_model.pth'))




    print("Training Done!!!")

if __name__ == "__main__":

    debugger = MyDebugger('training_points_autoencoder', save_print_to_file =True)
    f = h5py.File('./data/train_data.h5')
    data_train = f['data'][:]  ### [9840, 2048, 3]

    X = torch.from_numpy(data_train).float()

    model = config.current_model.to(device)


    train_model(model = model,
                debugger = debugger,
                X = X)

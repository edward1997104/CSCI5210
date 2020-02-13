import torch as torch
import torch.nn as nn
import h5py

class PointNet_Encoder(nn.Module):

    def __init__(self,
                layers = [64, 128, 128, 256, 128],
                input_dimension = 3,
                activation = nn.ReLU()):
        super(PointNet_Encoder, self).__init__()

        #### layers
        self.feature_dims = [input_dimension] + layers
        self.activation = activation

        self.forward_layers = nn.ModuleList(
            [ nn.Linear(self.feature_dims[i], self.feature_dims[i+1] )for i in range(len(self.feature_dims) - 1)]
        )

    def forward(self, x):
        ### forward pass
        for i in range(len(self.forward_layers) - 1):
            x = self.forward_layers[i](x)
            x = self.activation(x)

        ### no activation for last layer
        x = self.forward_layers[-1](x)

        #### max pool as default --> can be changed later
        x, _ = torch.max(x , dim = 1)

        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__' :
    encoder = PointNet_Encoder()

    f = h5py.File('./data/train_data.h5')
    data_train = f['data'][:]  ### [9840, 2048, 3]

    X = torch.from_numpy(data_train[:32]).float().to(device)

    X_code = encoder(X)

    print(X.size(), X_code.size())



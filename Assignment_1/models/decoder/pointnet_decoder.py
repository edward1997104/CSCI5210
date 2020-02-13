import torch as torch
import torch.nn as nn
import h5py
from models.encoder.pointnet_encoder import PointNet_Encoder

class PointNet_Decoder(nn.Module):

    def __init__(self,
                layers = [256, 256],
                input_dimension = 128,
                activation = nn.ReLU(),
                number_of_points = 2048):
        super(PointNet_Decoder, self).__init__()

        #### layers
        self.feature_dims = [input_dimension] + layers + [3 * number_of_points]
        self.activation = activation
        self.number_of_points = number_of_points

        self.forward_layers = nn.ModuleList(
            [ nn.Linear(self.feature_dims[i], self.feature_dims[i+1] )for i in range(len(self.feature_dims) - 1)]
        )

    def forward(self, x):

        # batch_size
        batch_size = x.size(0)

        ### forward pass
        for i in range(len(self.forward_layers) - 1):
            x = self.forward_layers[i](x)
            x = self.activation(x)

        ### no activation for last layer
        x = self.forward_layers[-1](x)

        x = x.view(batch_size, self.number_of_points, 3)

        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__' :
    encoder = PointNet_Encoder()
    decoder = PointNet_Decoder()

    f = h5py.File('./data/train_data.h5')
    data_train = f['data'][:]  ### [9840, 2048, 3]

    X = torch.from_numpy(data_train[:32]).float().to(device)

    X_code = encoder(X)

    X_reconstructed = decoder(X_code)

    print(X_reconstructed.size())



import torch as torch
import torch.nn as nn
import numpy as np
import itertools
import h5py
from models.encoder.pointnet_encoder import PointNet_Encoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FoldingNet_Decoder(nn.Module):

    def __init__(self,
                layers = [256, 256],
                input_dimension = 128,
                activation = nn.ReLU(),
                number_of_points = 2048,
                number_of_fold=2,
                x_y_samples=(64, 32),
                sample_range = (-1, 1)):
        super(FoldingNet_Decoder, self).__init__()

        #### layers
        self.activation = activation
        self.number_of_points = number_of_points

        ###### special args for folding net
        self.number_of_fold = number_of_fold
        self.x_y_samples = x_y_samples
        self.sample_range = sample_range

        self.forward_layers = nn.ModuleList([])
        for i in range(number_of_fold):
            feature_dims = [input_dimension + 2] + layers + [3] if i == 0 else [input_dimension + 3] + layers + [3]
            forward_layer = nn.ModuleList(
                [ nn.Linear(feature_dims[i], feature_dims[i+1] )for i in range(len(feature_dims) - 1)]
            )
            self.forward_layers.append(forward_layer)

    def forward(self, input_code : torch.Tensor):

        # batch_size
        batch_size = input_code.size(0)

        # batch process to make (B, 128) -> (B, N, 128)
        expanded_code = input_code.unsqueeze(1).repeat((1, self.number_of_points, 1))

        #### set up initial input to be (B, N, 2)
        x_samples, y_samples = np.linspace(self.sample_range[0], self.sample_range[1], self.x_y_samples[0]), \
                               np.linspace(self.sample_range[0], self.sample_range[1], self.x_y_samples[1])
        samples = list(itertools.product(x_samples, y_samples))
        x = torch.from_numpy(np.array(samples)).float().to(device)
        x = x.unsqueeze(0).repeat((batch_size, 1, 1))

        assert x.size(1) == self.number_of_points


        ### forward pass
        for i in range(len(self.forward_layers)):

            #### combine with the code first
            x = torch.cat([x, expanded_code], dim = 2)
            ### for each folding
            for j in range(len(self.forward_layers[i]) - 1):
                x = self.forward_layers[i][j](x)
                x = self.activation(x)

            ### no activation for last layer
            x = self.forward_layers[i][-1](x)

        return x

if __name__ == '__main__' :
    encoder = PointNet_Encoder()
    decoder = FoldingNet_Decoder()
    print(f'decoder parameters : {list(decoder.parameters())}')

    f = h5py.File('./data/train_data.h5')
    data_train = f['data'][:]  ### [9840, 2048, 3]

    X = torch.from_numpy(data_train[:32]).float().to(device)

    X_code = encoder(X)

    X_reconstructed = decoder(X_code)

    print(X_reconstructed.size())



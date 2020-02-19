import torch as torch
import torch.nn as nn
import h5py

class PCN_encoder(nn.Module):

    def __init__(self,
                layers = [64, 128, 128, 256, 128],
                input_dimension = 3,
                activation = nn.ReLU()):
        super(PCN_encoder, self).__init__()
        self.activation = activation

        #### layers 1
        feature_dims_1 = [input_dimension] + layers

        self.forward_layers_1 = nn.ModuleList(
            [ nn.Linear(feature_dims_1[i], feature_dims_1[i+1] )for i in range(len(feature_dims_1) - 1)]
        )

        #### layers  2
        feature_dims_2 = [layers[-1] * 2] + layers

        self.forward_layers_2 = nn.ModuleList(
            [ nn.Linear(feature_dims_2[i], feature_dims_2[i+1] )for i in range(len(feature_dims_2) - 1)]
        )

    def forward(self, x : torch.Tensor):

        ### number of points
        number_points = x.size(1)

        ### forward pass 1
        for i in range(len(self.forward_layers_1) - 1):
            x = self.forward_layers_1[i](x)
            x = self.activation(x)

        ### no activation for last layer
        points_features_1 = self.forward_layers_1[-1](x)

        #### max pool as default --> can be changed later
        x, _ = torch.max(points_features_1 , dim = 1)
        x = x.unsqueeze(1).repeat((1, number_points, 1))
        x = torch.cat((x, points_features_1), dim = 2)

        ### forward pass 2
        for i in range(len(self.forward_layers_2) - 1):
            x = self.forward_layers_2[i](x)
            x = self.activation(x)

        ### no activation for last layer
        points_features_1 = self.forward_layers_2[-1](x)

        #### max pool as default --> can be changed later
        x, _ = torch.max(points_features_1 , dim = 1)


        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__' :
    encoder = PCN_encoder()

    f = h5py.File('./data/train_data.h5')
    data_train = f['data'][:]  ### [9840, 2048, 3]

    X = torch.from_numpy(data_train[:32]).float().to(device)

    X_code = encoder(X)

    print(X.size(), X_code.size())



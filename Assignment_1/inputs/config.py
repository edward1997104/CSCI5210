from models.encoder.pointnet_encoder import PointNet_Encoder
from models.encoder.pcn_encoder import PCN_encoder
from models.decoder.pointnet_decoder import PointNet_Decoder
from models.decoder.foldingnet_decoder import FoldingNet_Decoder
import torch.nn as nn

encoder_layers = [64, 128, 128, 256, 128]
decoder_layers = [256, 256]
activation = nn.LeakyReLU()
# loss_fn = nn.MSELoss(reduce=True, size_average=True)
from metrics.evaluation_metrics import CD_loss
loss_fn = CD_loss

######### ARGS for folding Net decoder
number_of_fold = 2
x_y_samples = (64, 64)
sample_range = (-1, 1)

######### Model set-up

current_model = nn.Sequential(
    PCN_encoder(layers=encoder_layers,
                     activation=activation),
    # PointNet_Encoder(layers = encoder_layers,
    #                  activation = activation),
    # PointNet_Decoder(layers = decoder_layers,
    #                  activation = activation)
    FoldingNet_Decoder(layers=decoder_layers,
                       activation=activation,
                       number_of_fold = number_of_fold,
                       x_y_samples = x_y_samples,
                       sample_range = sample_range)
)



####################### DEBUG
debug_base_folder = '../debug'
save_print_to_file = True
top_k = 5
model_saving_epoch = 20
training_epoch = 200
batch_size = 32
tolerance_epoch = 5

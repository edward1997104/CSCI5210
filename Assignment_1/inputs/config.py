from models.encoder.pointnet_encoder import PointNet_Encoder
from models.decoder.pointnet_decoder import PointNet_Decoder
import torch.nn as nn

encoder_layers = [64, 128, 128, 256, 128]
decoder_layers = [256, 256]
activation = nn.ReLU()
# loss_fn = nn.MSELoss(reduce=True, size_average=True)
from metrics.evaluation_metrics import CD_loss
loss_fn = CD_loss


current_model = nn.Sequential(
    PointNet_Encoder(layers = encoder_layers,
                     activation = activation),
    PointNet_Decoder(layers = decoder_layers,
                     activation = activation)
)



####################### DEBUG
debug_base_folder = '../debug'
save_print_to_file = True
top_k = 5
model_saving_epoch = 20

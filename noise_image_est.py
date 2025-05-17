import torch
import torch.nn as nn

def create_conv33relu(nin_channels = 1, nout_channels = 1):
        return nn.Sequential(
            nn.Conv2d(nin_channels, nout_channels, kernel_size=(3,3), padding=1),
            nn.ReLU(),
        )

def copy_n_crop(encoder_features, decoder_features):
    """
    Copy features from encoder and crop them to match decoder dimensions.

    Args:
        encoder_features: Features from the encoder/contracting path
        decoder_features: Features from the decoder/expansive path

    Returns:
        Concatenated features
    """
    # Get the height and width of decoder features
    _, _, h_decoder, w_decoder = decoder_features.size()

    # Center crop the encoder features to match decoder dimensions
    _, _, h_encoder, w_encoder = encoder_features.size()

    # Calculate offsets for cropping
    h_offset = (h_encoder - h_decoder) // 2
    w_offset = (w_encoder - w_decoder) // 2

    # Crop encoder features
    cropped_encoder_features = encoder_features[:, :,
                               h_offset:h_offset + h_decoder,
                               w_offset:w_offset + w_decoder]

    # Concatenate along the channel dimension
    return torch.cat([cropped_encoder_features, decoder_features], dim=1)

class NoiseImageEst(nn.Module):
    def __init__(self, t_samples, image_shape):
        super(NoiseImageEst, self).__init__()
        self.image_shape = image_shape
        self.emb = nn.Embedding(t_samples, torch.prod(torch.tensor(image_shape)))

        self.left_layer0_0 = create_conv33relu(1,4)
        # self.left_layer0_1 = Conv33Relu(4,4)
        self.left_layer0to1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.left_layer1_0 = create_conv33relu(4,8)
        # self.left_layer1_1 = Conv33Relu(8,8)
        self.left_layer1to2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.bot_layer2_0 = create_conv33relu(8,16)
        # self.bot_layer2_1 = Conv33Relu(8,8)

        self.right_layer2to1 = nn.ConvTranspose2d(16, 8, kernel_size=2, stride=2)
        # Due to concatenation n channels is doubled back to 16
        self.right_layer_1_1 = create_conv33relu(16,8)
        # self.right_layer1_0 = Conv33Relu(16,16)

        self.right_layer_1to0= nn.ConvTranspose2d(8, 4, kernel_size=2, stride=2)
        # Due to concatenation n channels is doubled back to 8
        self.right_layer_0_1 = create_conv33relu(8,4)
        # self.right_layer0_0 = Conv33Relu(32,32)

        self.dense_out = nn.Conv2d(4,1, kernel_size=1)

    def forward(self, xt, t):
        self.t_emb_out = self.emb(t)
        self.comb_image = self.t_emb_out.view(xt.shape) + xt
        self.ll00_out = self.left_layer0_0(self.comb_image)
        self.ll0to1_out = self.left_layer0to1(self.ll00_out)
        self.ll10_out = self.left_layer1_0(self.ll0to1_out)
        self.ll1to2_out = self.left_layer1to2(self.ll10_out)
        self.bl20_out = self.bot_layer2_0(self.ll1to2_out)

        self.rl2to1_out = self.right_layer2to1(self.bl20_out)
        self.rl2to1_out_comb = copy_n_crop(self.ll10_out, self.rl2to1_out)

        self.rl11_out = self.right_layer_1_1(self.rl2to1_out_comb)
        self.rl1to0_out = self.right_layer_1to0(self.rl11_out)
        self.rl1to0_out_comb = copy_n_crop(self.ll00_out ,self.rl1to0_out)

        self.rl01_out = self.right_layer_0_1(self.rl1to0_out_comb)
        self.out = self.dense_out(self.rl01_out)
        return self.out

class NoiseImageEstImg2(nn.Module):
    def __init__(self, t_samples, image_shape):
        super(NoiseImageEstImg2, self).__init__()
        self.image_shape = image_shape
        self.emb = nn.Embedding(t_samples, image_shape[0])
        self.dense = nn.Sequential(
            nn.Flatten(),
            nn.Linear(image_shape[0]*(image_shape[1] + 1), 400),
            nn.ReLU(),
            nn.Linear(400, 100),
            nn.ReLU(),
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, 100),
            nn.ReLU(),
            nn.Linear(100, 400),
            nn.ReLU(),
            nn.Linear(400, 28*28),
            nn.ReLU(),
        )


    def forward(self, xt, t):
        self.emb_t_out = self.emb(t.view(xt.shape[0],1,1))
        self.comb = torch.cat([xt, self.emb_t_out], dim=2)
        self.out = self.dense(self.comb)
        return self.out.view(self.out.shape[0], 1, 28, 28)


import torch
import torch.nn as nn
import torch.nn.functional as F

# helper conv function
def conv(in_channels, out_channels, kernel_size, stride=2, padding=1, normalization=True, norm_type='instance_norm'):
    """Creates a convolutional layer, with optional batch normalization.
    """
    layers = []
    conv_layer = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, 
                           kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
    
    layers.append(conv_layer)

    # optional normalization layer
    if normalization == True and norm_type == 'instance_norm':
        layers.append(nn.InstanceNorm2d(out_channels))
    elif normalization == True and norm_type == 'batch_norm':
        layers.append(nn.BatchNorm2d(out_channels))
    return nn.Sequential(*layers)

# helper deconv function
def deconv(in_channels, out_channels, kernel_size, stride=2, padding=1, normalization=True, norm_type='instance_norm'):
    """Creates a transpose convolutional layer, with optional batch normalization.
    """
    layers = []
    # append transpose conv layer
    layers.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False))

    # optional normalization layer
    if normalization == True and norm_type == 'instance_norm':
        layers.append(nn.InstanceNorm2d(out_channels))
    elif normalization == True and norm_type == 'batch_norm':
        layers.append(nn.BatchNorm2d(out_channels))
    return nn.Sequential(*layers)

# residual block class
class ResidualBlock(nn.Module):
    """Defines a residual block.
       This adds an input x to a convolutional layer (applied to x) with the same size input and output.
       These blocks allow a model to learn an effective transformation from one domain to another.
    """
    def __init__(self, conv_dim):
        super(ResidualBlock, self).__init__()
        # conv_dim = number of inputs
        
        # define two convolutional layers + batch normalization that will act as our residual function, F(x)
        # layers should have the same shape input as output; I suggest a kernel_size of 3
        
        self.conv_layer1 = conv(in_channels=conv_dim, out_channels=conv_dim, 
                                kernel_size=3, stride=1, padding=1, normalization=True)
        
        self.conv_layer2 = conv(in_channels=conv_dim, out_channels=conv_dim, 
                               kernel_size=3, stride=1, padding=1, normalization=True)

        # leaky relu function
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)

        # reset parameters
        self.reset_parameters()

    def init_weights(self, layer):
        if type(layer) == nn.Conv2d:
            torch.nn.init.xavier_uniform(layer.weight)

    def reset_parameters(self):
        self.conv_layer1.apply(self.init_weights)
        self.conv_layer2.apply(self.init_weights)
        
    def forward(self, x):
        # apply a ReLu activation the outputs of the first layer
        # return a summed output, x + resnet_block(x)
        out_1 = self.leaky_relu(self.conv_layer1(x))
        out_2 = x + self.conv_layer2(out_1)
        return out_2

# Define the Global Discriminator Architecture
class GlobalDiscriminator(nn.Module):
    
    def __init__(self, conv_dim=64):
        super(GlobalDiscriminator, self).__init__()
        self.image_size = (1024, 256)

        # Define all convolutional layers
        # Convolutional layers, increasing in depth
        # first layer has *no* batchnorm
        self.conv1 = conv(3, conv_dim, 4, normalization=False)
        self.conv2 = conv(conv_dim, conv_dim*2, 4) 
        self.conv3 = conv(conv_dim*2, conv_dim*4, 4) 
        self.conv4 = conv(conv_dim*4, conv_dim*4, 4)
        self.conv5 = conv(conv_dim*4, conv_dim*4, 4)
        self.conv6 = conv(conv_dim*4, conv_dim*4, 4)
        self.conv7 = conv(conv_dim*4, conv_dim*4, 4)
        self.conv8 = conv(conv_dim*4, conv_dim*4, 4)
        # ully-connected layers
        fc_in_size = int(conv_dim*4*self.image_size[0]*self.image_size[1] / ((2**8)*(2**8)))
        self.fc1 = nn.Linear(fc_in_size, 64)
        
        # Classification layer
        self.disc_out = nn.Linear(64, 1)

        # reset parameters
        self.reset_parameters()

        # flatten layer
        self.flatten = nn.Flatten()
        # leaky relu function
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)

    def init_weights(self, layer):
        if type(layer) == nn.Conv2d or type(layer) == nn.Linear:
            torch.nn.init.xavier_uniform(layer.weight)

    def reset_parameters(self):
        self.conv1.apply(self.init_weights)
        self.conv2.apply(self.init_weights)
        self.conv3.apply(self.init_weights)
        self.conv4.apply(self.init_weights)
        self.conv5.apply(self.init_weights)
        self.conv6.apply(self.init_weights)
        self.conv7.apply(self.init_weights)
        self.conv8.apply(self.init_weights)
        self.fc1.apply(self.init_weights)
        self.disc_out.apply(self.init_weights)

    def forward(self, x):
        # relu applied to all conv layers but last
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = F.relu(self.conv3(out))
        out = F.relu(self.conv4(out))
        out = F.relu(self.conv5(out))
        out = F.relu(self.conv6(out))
        out = F.relu(self.conv7(out))
        out = F.relu(self.conv8(out))
        out = self.flatten(out)
        out = F.relu(self.fc1(out))
        # last, classification layer
        out = self.disc_out(out)
        return out

# Define the Patch Discriminator Architecture
class PatchDiscriminator(nn.Module):
    
    def __init__(self, conv_dim=64):
        super(PatchDiscriminator, self).__init__()

        # Define all convolutional layers
        # Convolutional layers, increasing in depth
        # first layer has *no* batchnorm
        self.conv1 = conv(3, conv_dim, 4, normalization=False)
        self.conv2 = conv(conv_dim, conv_dim*2, 4) 
        self.conv3 = conv(conv_dim*2, conv_dim*4, 4) 
        self.conv4 = conv(conv_dim*4, conv_dim*8, 4)
        
        # Classification layer
        self.conv_final = conv(conv_dim*8, 1, 3, stride=1, normalization=False)

        # reset parameters
        self.reset_parameters()

        # leaky relu function
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)

    def init_weights(self, layer):
        if type(layer) == nn.Conv2d:
            torch.nn.init.xavier_uniform(layer.weight)

    def reset_parameters(self):
        self.conv1.apply(self.init_weights)
        self.conv2.apply(self.init_weights)
        self.conv3.apply(self.init_weights)
        self.conv4.apply(self.init_weights)
        self.conv_final.apply(self.init_weights)

    def forward(self, x):
        # relu applied to all conv layers but last
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = F.relu(self.conv3(out))
        out = F.relu(self.conv4(out))
        # last, classification layer
        out = self.conv_final(out)
        return out

# Define the Generator Architecture
class Generator(nn.Module):
    
    def __init__(self, conv_dim=64, n_res_blocks=6):
        super(Generator, self).__init__()

        # Define the encoder part of the generator
        self.conv1 = conv(3, conv_dim, 4)
        self.conv2 = conv(conv_dim, conv_dim*2, 4)
        self.conv3 = conv(conv_dim*2, conv_dim*4, 4)
        self.conv4 = conv(conv_dim*4, conv_dim*4, 4)

        # Define the resnet part of the generator
        # Residual blocks
        res_layers = []
        for layer in range(n_res_blocks):
            res_layers.append(ResidualBlock(conv_dim*4))
        # use sequential to create these layers
        self.res_blocks = nn.Sequential(*res_layers)

        # Define the decoder part of the generator
        self.deconv1 = deconv(conv_dim*4, conv_dim*4, 4)
        self.deconv2 = deconv(conv_dim*4, conv_dim*2, 4)
        self.deconv3 = deconv(conv_dim*2, conv_dim, 4)

        # no batch norm on last layer
        self.out_layer = deconv(conv_dim, 3, 4, normalization=False)

        # reset parameters
        self.reset_parameters()

        # leaky relu function
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)

    def init_weights(self, layer):
        if type(layer) == nn.Conv2d or type(layer) == nn.ConvTranspose2d:
            torch.nn.init.xavier_uniform(layer.weight)

    def reset_parameters(self):
        self.conv1.apply(self.init_weights)
        self.conv2.apply(self.init_weights)
        self.conv3.apply(self.init_weights)
        self.conv4.apply(self.init_weights)

        self.deconv1.apply(self.init_weights)
        self.deconv2.apply(self.init_weights)
        self.deconv3.apply(self.init_weights)

        self.out_layer.apply(self.init_weights)
        
    def forward(self, x):
        """Given an image x, returns a transformed image."""
        # define feedforward behavior, applying activations as necessary
        out = self.leaky_relu(self.conv1(x))
        out = self.leaky_relu(self.conv2(out))
        out = self.leaky_relu(self.conv3(out))
        out = self.leaky_relu(self.conv4(out))

        out = self.res_blocks(out)

        out = self.leaky_relu(self.deconv1(out))
        out = self.leaky_relu(self.deconv2(out))
        out = self.leaky_relu(self.deconv3(out))

        # tanh applied to last layer
        out = F.tanh(self.out_layer(out))
        out = torch.clamp(out, min=-0.5, max=0.5)

        return out

def CycleGAN(g_conv_dim=64, d_conv_dim=64, n_res_blocks=6):
    """Builds the generators and discriminators."""
    
    # Instantiate generators
    G_XtoY = Generator(conv_dim=g_conv_dim, n_res_blocks=n_res_blocks)
    G_YtoX = Generator(conv_dim=g_conv_dim, n_res_blocks=n_res_blocks)
    # Instantiate patch discriminators
    Dp_X = PatchDiscriminator(conv_dim=d_conv_dim)
    Dp_Y = PatchDiscriminator(conv_dim=d_conv_dim)
    # Instantiate global discriminators
    Dg_X = GlobalDiscriminator(conv_dim=d_conv_dim)
    Dg_Y = GlobalDiscriminator(conv_dim=d_conv_dim)

    # move models to GPU, if available
    cuda_available = torch.cuda.is_available()
    device = torch.device("cuda:0" if cuda_available else "cpu")

    device = torch.device(device)
    G_XtoY.to(device)
    G_YtoX.to(device)
    Dp_X.to(device)
    Dp_Y.to(device)
    Dg_X.to(device)
    Dg_Y.to(device)

    print('Using {}.'.format("GPU" if cuda_available else "CPU"))
    return G_XtoY, G_YtoX, Dp_X, Dp_Y, Dg_X, Dg_Y
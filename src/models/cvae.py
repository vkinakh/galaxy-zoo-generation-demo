from torch import cat
from torch.optim import Adam
from torch.nn import Sequential, ModuleList, \
                     Conv2d, Linear, \
                     LeakyReLU, Tanh, \
                     BatchNorm1d, BatchNorm2d, \
                     ConvTranspose2d, UpsamplingBilinear2d

from .neuralnetwork import NeuralNetwork


# parameters for cVAE
colors_dim = 3
labels_dim = 37
momentum = 0.99  # Batchnorm
negative_slope = 0.2  # LeakyReLU
optimizer = Adam
betas = (0.5, 0.999)

# hyperparameters
learning_rate = 2e-4
latent_dim = 128


def genUpsample(input_channels, output_channels, stride, pad):
   return Sequential(
        ConvTranspose2d(input_channels, output_channels, 4, stride, pad, bias=False),
        BatchNorm2d(output_channels),
        LeakyReLU(negative_slope=negative_slope))


def genUpsample2(input_channels, output_channels, kernel_size):
   return Sequential(
        Conv2d(input_channels, output_channels, kernel_size=kernel_size, stride=1, padding= (kernel_size-1) // 2),
        BatchNorm2d(output_channels),
        LeakyReLU(negative_slope=negative_slope),
        Conv2d(output_channels, output_channels, kernel_size=kernel_size, stride=1, padding= (kernel_size-1) // 2),
        BatchNorm2d(output_channels),
        LeakyReLU(negative_slope=negative_slope),
        UpsamplingBilinear2d(scale_factor=2))


class ConditionalDecoder(NeuralNetwork):
    def __init__(self, ll_scaling=1.0, dim_z=latent_dim):
        super(ConditionalDecoder, self).__init__()
        self.dim_z = dim_z
        ngf = 32
        self.init = genUpsample(self.dim_z, ngf * 16, 1, 0)
        self.embedding = Sequential(
            Linear(labels_dim, self.dim_z),
            BatchNorm1d(self.dim_z, momentum=momentum),
            LeakyReLU(negative_slope=negative_slope),
        )
        self.dense_init = Sequential(
            Linear(self.dim_z*2, self.dim_z),
            BatchNorm1d(self.dim_z, momentum=momentum),
            LeakyReLU(negative_slope=negative_slope),
        )
        self.m_modules = ModuleList()  # to 4x4
        self.c_modules = ModuleList()
        for i in range(4):
            self.m_modules.append(genUpsample2(ngf * 2**(4-i), ngf * 2**(3-i), 3))
            self.c_modules.append(Sequential(Conv2d(ngf * 2**(3-i), colors_dim, 3, 1, 1, bias=False), Tanh()))
        self.set_optimizer(optimizer, lr=learning_rate*ll_scaling, betas=betas)

    def forward(self, latent, labels, step=3):
        y = self.embedding(labels)
        out = cat((latent, y), dim=1)
        out = self.dense_init(out)
        out = out.unsqueeze(2).unsqueeze(3)
        out = self.init(out)
        for i in range(step):
            out = self.m_modules[i](out)
        out = self.c_modules[step](self.m_modules[step](out))
        return out

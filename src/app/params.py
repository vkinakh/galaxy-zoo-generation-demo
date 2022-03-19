"""
This file contains list of global parameters for the Galaxy Zoo generation app
"""

device = 'cpu'
size = 64   # generated image size
shape_label = 37  # shape of the input label
n_channels = 3  # number of color channels in image
upsample = True   # if true, generated images will be upsampled
path_labels = './data/training_solutions_rev1.csv'
noise_dim = 512  # noise size in InfoSCC-GAN
n_basis = 6   # size of additional z vectors in InfoSCC-GAN
y_type = 'real'  # type of labels in InfoSCC-GAN
dim_z = 128  # z vector size in BigGAN and cVAE

path_infoscc_gan = './models/InfoSCC-GAN/generator.pt'
drive_id_infoscc_gan = '1_kIujc497OH0ZJ7PNPwS5_otNlS7jMLI'

path_biggan = './models/BigGAN/generator.pth'
drive_id_biggan = ''

path_cvae = './models/CVAE/generator.pth'
drive_id_cvae = ''

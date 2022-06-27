from pathlib import Path
import math

import numpy as np
import streamlit as st

import torch
import torch.nn.functional as F

import src.app.params as params
from src.models import ConditionalGenerator as InfoSCC_GAN
from src.models.big.BigGAN2 import Generator as BigGAN2Generator
from src.models import ConditionalDecoder as cVAE
from src.data import get_labels_train
from src.utils import download_file, sample_labels


device = params.device
size = params.size
n_layers = int(math.log2(size) - 2)
bs = 12
lin_space = torch.linspace(0, 1, bs).unsqueeze(1)
captions = [f'label_a * {(1 - x):.02f} + label_b * {x:.02f}' for x in lin_space.squeeze().numpy()]


@st.cache(allow_output_mutation=True)
def load_model(model_type: str):

    print(f'Loading model: {model_type}')
    if model_type == 'InfoSCC-GAN':
        g = InfoSCC_GAN(size=params.size,
                        y_size=params.shape_label,
                        z_size=params.noise_dim)

        if not Path(params.path_infoscc_gan).exists():
            download_file(params.drive_id_infoscc_gan, params.path_infoscc_gan)

        ckpt = torch.load(params.path_infoscc_gan, map_location=torch.device('cpu'))
        g.load_state_dict(ckpt['g_ema'])
    elif model_type == 'BigGAN':
        g = BigGAN2Generator()

        if not Path(params.path_biggan).exists():
            download_file(params.drive_id_biggan, params.path_biggan)

        ckpt = torch.load(params.path_biggan, map_location=torch.device('cpu'))
        g.load_state_dict(ckpt)
    elif model_type == 'cVAE':
        g = cVAE()

        if not Path(params.path_cvae).exists():
            download_file(params.drive_id_cvae, params.path_cvae)

        ckpt = torch.load(params.path_cvae, map_location=torch.device('cpu'))
        g.load_state_dict(ckpt)
    else:
        raise ValueError('Unsupported model')
    g = g.eval().to(device=params.device)
    return g


@st.cache
def get_labels() -> torch.Tensor:
    path_labels = params.path_labels

    if not Path(path_labels).exists():
        download_file(params.drive_id_labels, path_labels)

    labels_train = get_labels_train(path_labels)
    return labels_train


def get_eps(n: int) -> torch.Tensor:
    eps = torch.randn((n, params.dim_z), device=device)
    return eps


def app():

    global lin_space, captions

    st.title('Interpolate Labels')
    st.markdown('This app allows the generation of the images with the labels that are interpolated between two labels.')
    st.markdown('In each row there are images generated with the same interpolated label by one of the models')

    biggan = load_model('BigGAN')
    infoscc_gan = load_model('InfoSCC-GAN')
    cvae = load_model('cVAE')
    labels_train = get_labels()

    # ==================== Labels ==============================================
    label_a = sample_labels(labels_train, n=1).repeat(bs, 1)
    label_b = sample_labels(labels_train, n=1).repeat(bs, 1)
    label_interpolated = (1 - lin_space) * label_a + lin_space * label_b

    sample_label = st.button('Sample label')
    if sample_label:
        label_a = sample_labels(labels_train, n=1).repeat(bs, 1)
        label_b = sample_labels(labels_train, n=1).repeat(bs, 1)
        label_interpolated = (1 - lin_space) * label_a + lin_space * label_b
    # ==================== Labels ==============================================

    # ==================== Noise ==============================================
    eps = get_eps(1).repeat(bs, 1)
    eps_infoscc = infoscc_gan.sample_eps(1).repeat(bs, 1)

    zs = np.array([[0.0] * params.n_basis] * n_layers, dtype=np.float32)
    zs_torch = torch.from_numpy(zs).unsqueeze(0).repeat(bs, 1, 1).to(device)

    st.subheader('Noise')
    st.markdown(r'Click on __Change eps__ button to change input $\varepsilon$ latent space')
    change_eps = st.button('Change eps')
    if change_eps:
        eps = get_eps(1).repeat(bs, 1)
        eps_infoscc = infoscc_gan.sample_eps(1).repeat(bs, 1)
    # ==================== Noise ==============================================

    with torch.no_grad():
        imgs_biggan = biggan(eps, label_interpolated).squeeze(0).cpu()
        imgs_infoscc = infoscc_gan(label_interpolated, eps_infoscc, zs_torch).squeeze(0).cpu()
        imgs_cvae = cvae(eps, label_interpolated).squeeze(0).cpu()

    if params.upsample:
        imgs_biggan = F.interpolate(imgs_biggan, (size * 4, size * 4), mode='bicubic')
        imgs_infoscc = F.interpolate(imgs_infoscc, (size * 4, size * 4), mode='bicubic')
        imgs_cvae = F.interpolate(imgs_cvae, (size * 4, size * 4), mode='bicubic')

    imgs_biggan = torch.clip(imgs_biggan, 0, 1)
    imgs_biggan = [(imgs_biggan[i].permute(1, 2, 0).numpy() * 255).astype(np.uint8) for i in range(bs)]
    imgs_infoscc = [(imgs_infoscc[i].permute(1, 2, 0).numpy() * 127.5 + 127.5).astype(np.uint8) for i in range(bs)]
    imgs_cvae = [(imgs_cvae[i].permute(1, 2, 0).numpy() * 127.5 + 127.5).astype(np.uint8) for i in range(bs)]

    c1, c2, c3 = st.columns(3)
    c1.header('BigGAN')
    c1.image(imgs_biggan, use_column_width=True, caption=captions)

    c2.header('InfoSCC-GAN')
    c2.image(imgs_infoscc, use_column_width=True, caption=captions)

    c3.header('cVAE')
    c3.image(imgs_cvae, use_column_width=True, caption=captions)

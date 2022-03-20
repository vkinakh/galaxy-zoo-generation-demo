from pathlib import Path
import math

import streamlit as st
import numpy as np

import torch
import torch.nn.functional as F

import src.app.params as params
from src.app.questions import q1, q1_options, q2, q2_options, q3, q3_options, q4, q4_options, q5, q5_options, \
    q6, q6_options, q7, q7_options, q8, q8_options, q9, q9_options, q10, q10_options, q11, q11_options
from src.models import ConditionalGenerator as InfoSCC_GAN
from src.models.big.BigGAN2 import Generator as BigGAN2Generator
from src.models import ConditionalDecoder as cVAE
from src.data import get_labels_train, make_galaxy_labels_hierarchical
from src.utils import download_file, sample_labels


device = params.device
bs = 10  # number of images to generate each model
n_cols = int(math.sqrt(bs))
size = params.size
n_layers = int(math.log2(size) - 2)

# manual labels
q1_out = [0] * len(q1_options)
q2_out = [0] * len(q2_options)
q3_out = [0] * len(q3_options)
q4_out = [0] * len(q4_options)
q5_out = [0] * len(q5_options)
q6_out = [0] * len(q6_options)
q7_out = [0] * len(q7_options)
q8_out = [0] * len(q8_options)
q9_out = [0] * len(q9_options)
q10_out = [0] * len(q10_options)
q11_out = [0] * len(q11_options)


def clear_out(elems=None):
    global q1_out, q2_out, q3_out, q4_out, q5_out, q6_out, q6_out, q7_out, q8_out, q9_out, q10_out, q11_out

    if elems is None:
        elems = list(range(1, 12))

    if 1 in elems:
        q1_out = [0] * len(q1_options)
    if 2 in elems:
        q2_out = [0] * len(q2_options)
    if 3 in elems:
        q3_out = [0] * len(q3_options)
    if 4 in elems:
        q4_out = [0] * len(q4_options)
    if 5 in elems:
        q5_out = [0] * len(q5_options)
    if 6 in elems:
        q6_out = [0] * len(q6_options)
    if 7 in elems:
        q7_out = [0] * len(q7_options)
    if 8 in elems:
        q8_out = [0] * len(q8_options)
    if 9 in elems:
        q9_out = [0] * len(q9_options)
    if 10 in elems:
        q10_out = [0] * len(q10_options)
    if 11 in elems:
        q11_out = [0] * len(q11_options)


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
    global q1_out, q2_out, q3_out, q4_out, q5_out, q6_out, q6_out, q7_out, q8_out, q9_out, q10_out, q11_out

    st.title('Compare models')
    st.markdown('This demo allows to compare BigGAN, InfoSCC-GAN and cVAE models for conditional galaxy generation.')
    st.markdown('In each row there are images generated with the same labels by each of the models')

    biggan = load_model('BigGAN')
    infoscc_gan = load_model('InfoSCC-GAN')
    cvae = load_model('cVAE')
    labels_train = get_labels()

    eps = get_eps(bs)  # for BigGAN and cVAE
    eps_infoscc = infoscc_gan.sample_eps(bs)

    zs = np.array([[0.0] * params.n_basis] * n_layers, dtype=np.float32)
    zs_torch = torch.from_numpy(zs).unsqueeze(0).repeat(bs, 1, 1).to(device)

    # ========================== Labels ================================
    st.subheader('Label')
    st.markdown(r'There are two types of selecting labels: __Random__ - sample random samples from the dataset;'
                r' __Manual__ - select labels manually (advanced use). When using __Manual__ all of the images will be'
                r' generated with tha same labels')
    label_type = st.radio('Label type', options=['Random', 'Manual (Advanced)'])
    if label_type == 'Random':
        labels = sample_labels(labels_train, bs).to(device)

        st.markdown(r'Click on __Sample labels__ button to sample random input labels')
        change_label = st.button('Sample label')

        if change_label:
            labels = sample_labels(labels_train, bs).to(device)
    elif label_type == 'Manual (Advanced)':
        st.markdown('Answer the questions below')

        q1_select_box = st.selectbox(q1, options=q1_options)
        clear_out()
        q1_out[q1_options.index(q1_select_box)] = 1
        # 1

        if q1_select_box == 'Smooth':
            q7_select_box = st.selectbox(q7, options=q7_options)
            clear_out([2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
            q7_out[q7_options.index(q7_select_box)] = 1
            # 1 - 7

            q6_select_box = st.selectbox(q6, options=q6_options)
            clear_out([2, 3, 4, 5, 6, 8, 9, 10, 11])
            q6_out[q6_options.index(q6_select_box)] = 1
            # 1 - 7 - 6

            if q6_select_box == 'Yes':
                q8_select_box = st.selectbox(q8, options=q8_options)
                clear_out([2, 3, 4, 5, 8, 9, 10, 11])
                q8_out[q8_options.index(q8_select_box)] = 1
                # 1 - 7 - 6 - 8 - end

        elif q1_select_box == 'Features or disk':
            q2_select_box = st.selectbox(q2, options=q2_options)
            clear_out([2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
            q2_out[q2_options.index(q2_select_box)] = 1
            # 1 - 2

            if q2_select_box == 'Yes':
                q9_select_box = st.selectbox(q9, options=q9_options)
                clear_out([3, 4, 5, 6, 7, 8, 9, 10, 11])
                q9_out[q9_options.index(q9_select_box)] = 1
                # 1 - 2 - 9

                q6_select_box = st.selectbox(q6, options=q6_options)
                clear_out([3, 4, 5, 6, 7, 8, 10, 11])
                q6_out[q6_options.index(q6_select_box)] = 1
                # 1 - 2 - 9 - 6

                if q6_select_box == 'Yes':
                    q8_select_box = st.selectbox(q8, options=q8_options)
                    clear_out([3, 4, 5, 7, 8, 10, 11])
                    q8_out[q8_options.index(q8_select_box)] = 1
                    # 1 - 2 - 9 - 6 - 8
            else:
                q3_select_box = st.selectbox(q3, options=q3_options)
                clear_out([3, 4, 5, 6, 7, 8, 9, 10, 11])
                q3_out[q3_options.index(q3_select_box)] = 1
                # 1 - 2 - 3

                q4_select_box = st.selectbox(q4, options=q4_options)
                clear_out([4, 5, 6, 7, 8, 9, 10, 11])
                q4_out[q4_options.index(q4_select_box)] = 1
                # 1 - 2 - 3 - 4

                if q4_select_box == 'Yes':
                    q10_select_box = st.selectbox(q10, options=q10_options)
                    clear_out([5, 6, 7, 8, 9, 10, 11])
                    q10_out[q10_options.index(q10_select_box)] = 1
                    # 1 - 2 - 3 - 4 - 10

                    q11_select_box = st.selectbox(q11, options=q11_options)
                    clear_out([5, 6, 7, 8, 9, 11])
                    q11_out[q11_options.index(q11_select_box)] = 1
                    # 1 - 2 - 3 - 4 - 10 - 11

                    q5_select_box = st.selectbox(q5, options=q5_options)
                    clear_out([5, 6, 7, 8, 9])
                    q5_out[q5_options.index(q5_select_box)] = 1
                    # 1 - 2 - 3 - 4 - 10 - 11 - 5

                    q6_select_box = st.selectbox(q6, options=q6_options)
                    clear_out([6, 7, 8, 9])
                    q6_out[q6_options.index(q6_select_box)] = 1
                    # 1 - 2 - 3 - 4 - 10 - 11 - 5 - 6

                    if q6_select_box == 'Yes':
                        q8_select_box = st.selectbox(q8, options=q8_options)
                        clear_out([7, 8, 9])
                        q8_out[q8_options.index(q8_select_box)] = 1
                        # 1 - 2 - 3 - 4 - 10 - 11 - 5 - 6 - 8 - End
                else:
                    q5_select_box = st.selectbox(q5, options=q5_options)
                    clear_out([5, 6, 7, 8, 9, 10, 11])
                    q5_out[q5_options.index(q5_select_box)] = 1
                    # 1 - 2 - 3 - 4 - 5

                    q6_select_box = st.selectbox(q6, options=q6_options)
                    clear_out([6, 7, 8, 9, 10, 11])
                    q6_out[q6_options.index(q6_select_box)] = 1
                    # 1 - 2 - 3 - 4 - 5 - 6

                    if q6_select_box == 'Yes':
                        q8_select_box = st.selectbox(q8, options=q8_options)
                        clear_out([7, 8, 9, 10, 11])
                        q8_out[q8_options.index(q8_select_box)] = 1
                        # 1 - 2 - 3 - 4 - 5 - 6 - 8 - End

        labels = [*q1_out, *q2_out, *q3_out, *q4_out, *q5_out, *q6_out, *q7_out, *q8_out, *q9_out, *q10_out, *q11_out]
        labels = torch.Tensor(labels).to(device)
        labels = labels.unsqueeze(0).repeat(bs, 1)
        labels = make_galaxy_labels_hierarchical(labels)
        clear_out()
    # ========================== Labels ================================

    st.subheader('Noise')
    st.markdown(r'Click on __Change eps__ button to change input $\varepsilon$ latent space')
    change_eps = st.button('Change eps')
    if change_eps:
        eps = get_eps(bs)  # for BigGAN and cVAE
        eps_infoscc = infoscc_gan.sample_eps(bs)

    with torch.no_grad():
        imgs_biggan = biggan(eps, labels).squeeze(0).cpu()
        imgs_infoscc = infoscc_gan(labels, eps_infoscc, zs_torch).squeeze(0).cpu()
        imgs_cvae = cvae(eps, labels).squeeze(0).cpu()

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
    c1.image(imgs_biggan, use_column_width=True)

    c2.header('InfoSCC-GAN')
    c2.image(imgs_infoscc, use_column_width=True)

    c3.header('cVAE')
    c3.image(imgs_cvae, use_column_width=True)

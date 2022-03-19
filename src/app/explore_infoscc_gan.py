from pathlib import Path
import math

import numpy as np
from google_drive_downloader import GoogleDriveDownloader as gdd
import streamlit as st

import torch
import torch.nn.functional as F

import src.app.params as params
from src.models import ConditionalGenerator
from src.data import get_labels_train, make_galaxy_labels_hierarchical

# global parameters
device = params.device
size = params.size
y_size = shape_label = params.shape_label
n_channels = params.n_channels
upsample = params.upsample
z_size = noise_dim = 512
n_layers = int(math.log2(size) - 2)
n_basis = 6
y_type = 'real'
bs = 16  # number of samples to generate
n_cols = int(math.sqrt(bs))
model_path = './models/InfoSCC-GAN/generator.pt'  # path to the model
drive_id = '1_kIujc497OH0ZJ7PNPwS5_otNlS7jMLI'   # google drive id of the model
path_labels = './data/training_solutions_rev1.csv'

# manual labels, question values
q1 = 'Is the object a smooth galaxy, a galaxy with features/disk or a star?'
q1_options = ['Smooth', 'Features or disk', 'Star or artifact']
q1_out = [0] * len(q1_options)

q2 = 'Is it edge-on? '
q2_options = ['Yes', 'No']
q2_out = [0] * len(q2_options)

q3 = 'Is there a bar?'
q3_options = ['Yes', 'No']
q3_out = [0] * len(q3_options)

q4 = 'Is there a spiral pattern?'
q4_options = ['Yes', 'No']
q4_out = [0] * len(q4_options)

q5 = 'How prominent is the central bulge?'
q5_options = ['No bulge', 'Just noticeable', 'Obvious', 'Dominant']
q5_out = [0] * len(q5_options)

q6 = 'Is there anything "odd" about the galaxy?'
q6_options = ['Yes', 'No']
q6_out = [0] * len(q6_options)

q7 = 'How round is the smooth galaxy?'
q7_options = ['Completely round', 'In between', 'Cigar-shaped']
q7_out = [0] * len(q7_options)

q8 = 'What is the odd feature?'
q8_options = ['Ring', 'Lens or are', 'Disturbed', 'Irregular', 'Other', 'Merger', 'Dust lane']
q8_out = [0] * len(q8_options)

q9 = 'What shape is the bulge in the edge-on galaxy?'
q9_options = ['Rounded', 'Boxy', 'No bulge']
q9_out = [0] * len(q9_options)

q10 = 'How tightly wound are the spiral arms?'
q10_options = ['Tight', 'Medium', 'Loose']
q10_out = [0] * len(q10_options)

q11 = 'How many spiral arms are there?'
q11_options = ['1', '2', '3', '4', 'more than four', 'can`t tell']
q11_out = [0] * len(q11_options)


def download_model(file_id: str, output_path: str):
    gdd.download_file_from_google_drive(file_id=file_id,
                                        dest_path=output_path)


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
def load_model(model_path: str) -> ConditionalGenerator:

    print(f'Loading model: {model_path}')
    g_ema = ConditionalGenerator(size, y_size, z_size, n_channels, n_basis, noise_dim)
    ckpt = torch.load(model_path, map_location=torch.device('cpu'))
    g_ema.load_state_dict(ckpt['g_ema'])
    g_ema.eval().to(device)
    return g_ema


@st.cache(allow_output_mutation=True)
def get_eps(model: ConditionalGenerator, n: int) -> torch.Tensor:
    eps = model.sample_eps(n)
    return eps.to(device)


def get_labels(n: int) -> torch.Tensor:
    labels_train = get_labels_train(path_labels)
    high = int(labels_train.shape[0])
    idx = np.random.randint(0, high, size=n)
    return labels_train[idx]


def app():
    global q1_out, q2_out, q3_out, q4_out, q5_out, q6_out, q6_out, q7_out, q8_out, q9_out, q10_out, q11_out

    st.title('Explore InfoSCC-GAN')
    st.markdown('This demo shows *Stochastic Contrastive Conditional Generative Adversarial Network* (InfoSCC-GAN) '
                'for conditional galaxy generation')

    st.subheader(r'<- Use sidebar to explore $z_1, ..., z_k$ latent variables')

    if not Path(model_path).exists():
        download_model(drive_id, model_path)

    model = load_model(model_path)
    eps = get_eps(model, bs)

    # get zs
    zs = np.array([[0.0] * n_basis] * n_layers, dtype=np.float32)

    for l in range(n_layers):
        st.sidebar.markdown(f'## Layer: {l}')
        for d in range(n_basis):
            zs[l][d] = st.sidebar.slider(f'Dimension: {d}', key=f'{l}{d}',
                                         min_value=-5., max_value=5., value=0., step=0.1)

    # ========================== Labels ================================
    st.subheader('Label')
    st.markdown(r'There are two types of selecting labels: __Random__ - sample random samples from the dataset;'
                r' __Manual__ - select labels manually (advanced use). When using __Manual__ all of the images will be'
                r' generated with tha same labels')
    label_type = st.radio('Label type', options=['Random', 'Manual (Advanced)'])
    if label_type == 'Random':
        labels = get_labels(bs).to(device)

        st.markdown(r'Click on __Sample labels__ button to sample random input labels')
        change_label = st.button('Sample label')

        if change_label:
            labels = get_labels(bs).to(device)
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
        eps = model.sample_eps(bs).to(device)

    zs_torch = torch.from_numpy(zs).unsqueeze(0).repeat(bs, 1, 1).to(device)

    with torch.no_grad():
        imgs = model(labels, eps, zs_torch).squeeze(0).cpu()

    if upsample:
        imgs = F.interpolate(imgs, (size * 4, size * 4), mode='bicubic')

    imgs = [(imgs[i].permute(1, 2, 0).numpy() * 127.5 + 127.5).astype(np.uint8) for i in range(bs)]

    counter = 0
    for r in range(bs // n_cols):
        cols = st.columns(n_cols)

        for c in range(n_cols):
            cols[c].image(imgs[counter])
            counter += 1

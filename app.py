"""
Aplicación Streamlit para generar imágenes usando GANs entrenadas
y visualizar muestras del dataset Fashion-MNIST.
"""
import os
import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import FashionMNIST
import numpy as np

# ================= CONFIGURACIÓN =================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
Z_DIM = 100
IMAGE_SIZE = 28
# Clases de Fashion-MNIST
CLASS_NAMES = [
    'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
]

# ================= MODELOS =================
class MLPGenerator(nn.Module):
    """Generador MLP para Fashion-MNIST"""
    def __init__(self, z_dim=Z_DIM, img_dim=IMAGE_SIZE * IMAGE_SIZE):
        super(MLPGenerator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, img_dim),
            nn.Tanh()
        )

    def forward(self, z):
        return self.model(z)


class DCGenerator(nn.Module):
    """Generador DC-GAN para Fashion-MNIST"""
    def __init__(self, z_dim=Z_DIM, img_channels=1, features_g=64):
        super(DCGenerator, self).__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(z_dim, features_g * 4, 4, 1, 0, bias=False),
            nn.BatchNorm2d(features_g * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(features_g * 4, features_g * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features_g * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(features_g * 2, features_g, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features_g),
            nn.ReLU(True),
            nn.ConvTranspose2d(features_g, img_channels, 4, 2, 2, bias=False),
            nn.Tanh()
        )

    def forward(self, z):
        return self.model(z)


# ================= FUNCIONES =================
@st.cache_resource
def load_model(model_name):
    """Carga el generador entrenado desde 'results/'"""
    if model_name == 'MLP-GAN':
        model = MLPGenerator()
        path = os.path.join('results', 'mlp_generator.pth')
    else:
        model = DCGenerator()
        path = os.path.join('results', 'dc_generator.pth')
    if not os.path.isfile(path):
        st.error(f"Modelo no encontrado: {path}")
        st.stop()
    state = torch.load(path, map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    return model


@st.cache_data
def load_dataset():
    """Carga Fashion-MNIST (modo sólo lectura)"""
    transform = transforms.ToTensor()
    ds = FashionMNIST(root='data', train=True, download=False, transform=transform)
    return ds


def generate_images(model, num_images, seed):
    """Genera imágenes a partir del modelo y ruido aleatorio."""
    torch.manual_seed(seed)
    noise = torch.randn(num_images, Z_DIM, device=DEVICE)
    # Para DC-GAN usamos ruido con forma (batch, Z_DIM, 1, 1)
    if model_option == 'DC-GAN':
        noise = noise.view(num_images, Z_DIM, 1, 1)
    with torch.no_grad():
        out = model(noise).cpu()
    # Dar forma adecuada para la GAN clásica (MLP)
    if model_option == 'MLP-GAN':
        out = out.view(num_images, 1, IMAGE_SIZE, IMAGE_SIZE)
    imgs = (out + 1) / 2  # escala [-1,1] -> [0,1]
    return imgs.numpy()


def sample_real_images(dataset, label, num_samples):
    """Selecciona aleatoriamente muestras reales del dataset."""
    if label is None:
        idxs = np.random.choice(len(dataset), size=num_samples, replace=False)
    else:
        idxs_all = [i for i, (_, lbl) in enumerate(dataset) if lbl == label]
        idxs = np.random.choice(idxs_all, size=min(num_samples, len(idxs_all)), replace=False)
    imgs = [dataset[i][0].numpy().squeeze() for i in idxs]
    return imgs


# ================= INTERFAZ =================
st.title("GANs en Fashion-MNIST 🌟")
st.write("Genera imágenes con GAN clásica (MLP-GAN) o DC-GAN y compara con muestras reales del dataset.")

# Sidebar - configuración
st.sidebar.header("Configuración")
model_option = st.sidebar.selectbox("Modelo pre-entrenado", ['MLP-GAN', 'DC-GAN'])
num_gen = st.sidebar.slider("Número de imágenes a generar", 1, 25, 9)
num_real = st.sidebar.slider("Número de imágenes reales para mostrar", 1, 25, 9)
class_option = st.sidebar.selectbox("Clase de Fashion-MNIST", ['Aleatorio'] + CLASS_NAMES)
seed = st.sidebar.number_input("Semilla aleatoria", 0, 9999, value=42, step=1)
run = st.sidebar.button("Ejecutar generación")

# Cargar datos y modelo
dataset = load_dataset()
model = load_model(model_option)

if run:
    # Generar imágenes
    gen_imgs = generate_images(model, num_gen, seed)
    st.subheader("Imágenes generadas")
    imgs_to_show = []
    for i in range(gen_imgs.shape[0]):
        img = gen_imgs[i]
        # Asegurar que la imagen tenga forma (H, W) para escala de grises
        while img.ndim > 2:
            img = img.squeeze()
        # Si queda como 1D, reshape a 2D
        if img.ndim == 1:
            img = img.reshape(IMAGE_SIZE, IMAGE_SIZE)
        imgs_to_show.append(img)
    if imgs_to_show:
        st.image(imgs_to_show, width=100)

    # Mostrar imágenes reales
    st.subheader("Imágenes reales del dataset")
    label = None if class_option == 'Aleatorio' else CLASS_NAMES.index(class_option)
    real_imgs = sample_real_images(dataset, label, num_real)
    if real_imgs:
        st.image(real_imgs, width=100)
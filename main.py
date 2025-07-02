import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime

# Configuración de reproducibilidad
torch.manual_seed(42)
np.random.seed(42)

# Configuración de dispositivo
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Usando dispositivo: {device}")

# Hiperparámetros globales
BATCH_SIZE = 64
LEARNING_RATE = 0.0002
NUM_EPOCHS = 100
Z_DIM = 100
IMAGE_SIZE = 28
BETA1 = 0.5


class MLPGenerator(nn.Module):
    """Generador basado en capas completamente conectadas (MLP)"""

    def __init__(self, z_dim=100, img_dim=784):
        super(MLPGenerator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, img_dim),
            nn.Tanh()  # Salida en rango [-1, 1]
        )

    def forward(self, z):
        return self.model(z)


class MLPDiscriminator(nn.Module):
    """Discriminador basado en capas completamente conectadas (MLP)"""

    def __init__(self, img_dim=784):
        super(MLPDiscriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(img_dim, 1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        return self.model(img)


class DCGenerator(nn.Module):
    """Generador DC-GAN con capas convolucionales transpuestas"""

    def __init__(self, z_dim=100, img_channels=1, features_g=64):
        super(DCGenerator, self).__init__()
        self.model = nn.Sequential(
            # Entrada: z_dim x 1 x 1
            nn.ConvTranspose2d(z_dim, features_g * 4, 4, 1, 0, bias=False),
            nn.BatchNorm2d(features_g * 4),
            nn.ReLU(True),
            # Estado: (features_g*4) x 4 x 4
            nn.ConvTranspose2d(features_g * 4, features_g * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features_g * 2),
            nn.ReLU(True),
            # Estado: (features_g*2) x 8 x 8
            nn.ConvTranspose2d(features_g * 2, features_g, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features_g),
            nn.ReLU(True),
            # Estado: (features_g) x 16 x 16
            nn.ConvTranspose2d(features_g, img_channels, 4, 2, 2, bias=False),
            nn.Tanh()
            # Salida: img_channels x 28 x 28
        )

    def forward(self, z):
        return self.model(z)


class DCDiscriminator(nn.Module):
    """Discriminador DC-GAN con capas convolucionales"""

    def __init__(self, img_channels=1, features_d=64):
        super(DCDiscriminator, self).__init__()
        self.model = nn.Sequential(
            # Entrada: img_channels x 28 x 28
            nn.Conv2d(img_channels, features_d, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # Estado: features_d x 14 x 14
            nn.Conv2d(features_d, features_d * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features_d * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # Estado: (features_d*2) x 7 x 7
            nn.Conv2d(features_d * 2, features_d * 4, 4, 1, 1, bias=False),
            nn.BatchNorm2d(features_d * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # Estado: (features_d*4) x 6 x 6
            nn.Conv2d(features_d * 4, 1, 6, 1, 0, bias=False),
            nn.Sigmoid()
            # Salida: 1 x 1 x 1
        )

    def forward(self, img):
        return self.model(img).view(-1, 1).squeeze(1)


def weights_init(m):
    """Inicialización de pesos según paper DC-GAN"""
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def load_fashion_mnist():
    """Carga y prepara el dataset Fashion-MNIST"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Normalizar a [-1, 1]
    ])

    dataset = torchvision.datasets.FashionMNIST(
        root='./data', train=True, download=True, transform=transform
    )

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    return dataloader


def train_gan(generator, discriminator, dataloader, model_name, epochs=NUM_EPOCHS):
    """Función de entrenamiento para ambos tipos de GAN"""
    print(f"\n=== Entrenando {model_name} ===")

    # Optimizadores
    optim_g = optim.Adam(generator.parameters(), lr=LEARNING_RATE, betas=(BETA1, 0.999))
    optim_d = optim.Adam(discriminator.parameters(), lr=LEARNING_RATE, betas=(BETA1, 0.999))

    # Función de pérdida
    criterion = nn.BCELoss()

    # Listas para almacenar pérdidas
    g_losses = []
    d_losses = []

    # Vector de ruido fijo para generar imágenes de seguimiento
    fixed_noise = torch.randn(64, Z_DIM, device=device)
    if model_name == "DC-GAN":
        fixed_noise = fixed_noise.view(64, Z_DIM, 1, 1)

    for epoch in range(epochs):
        for i, (real_imgs, _) in enumerate(dataloader):
            batch_size = real_imgs.shape[0]
            real_imgs = real_imgs.to(device)

            # Etiquetas
            real_label = torch.ones(batch_size, device=device)
            fake_label = torch.zeros(batch_size, device=device)

            # ============ Entrenar Discriminador ============
            discriminator.zero_grad()

            # Pérdida con imágenes reales
            if model_name == "MLP-GAN":
                real_imgs_flat = real_imgs.view(batch_size, -1)
                output_real = discriminator(real_imgs_flat).view(-1)
            else:
                output_real = discriminator(real_imgs).view(-1)

            loss_d_real = criterion(output_real, real_label)

            # Pérdida con imágenes falsas
            noise = torch.randn(batch_size, Z_DIM, device=device)
            if model_name == "DC-GAN":
                noise = noise.view(batch_size, Z_DIM, 1, 1)

            fake_imgs = generator(noise)

            if model_name == "MLP-GAN":
                fake_imgs_flat = fake_imgs.view(batch_size, -1)
                output_fake = discriminator(fake_imgs_flat.detach()).view(-1)
            else:
                output_fake = discriminator(fake_imgs.detach()).view(-1)

            loss_d_fake = criterion(output_fake, fake_label)

            # Pérdida total del discriminador
            loss_d = loss_d_real + loss_d_fake
            loss_d.backward()
            optim_d.step()

            # ============ Entrenar Generador ============
            generator.zero_grad()

            if model_name == "MLP-GAN":
                fake_imgs_flat = fake_imgs.view(batch_size, -1)
                output = discriminator(fake_imgs_flat).view(-1)
            else:
                output = discriminator(fake_imgs).view(-1)

            loss_g = criterion(output, real_label)
            loss_g.backward()
            optim_g.step()

            # Guardar pérdidas
            if i % 100 == 0:
                g_losses.append(loss_g.item())
                d_losses.append(loss_d.item())

        # Imprimir progreso
        if epoch % 10 == 0:
            print(f'Época [{epoch}/{epochs}] - Loss D: {loss_d.item():.4f}, Loss G: {loss_g.item():.4f}')

    # Generar imágenes finales
    with torch.no_grad():
        fake_imgs = generator(fixed_noise)
        if model_name == "MLP-GAN":
            fake_imgs = fake_imgs.view(-1, 1, 28, 28)

    return g_losses, d_losses, fake_imgs


def save_images(images, model_name, num_images=10):
    """Guarda las imágenes generadas"""
    os.makedirs(f'results/{model_name}', exist_ok=True)

    # Desnormalizar imágenes de [-1, 1] a [0, 1]
    images = (images + 1) / 2

    # Crear figura
    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    fig.suptitle(f'Imágenes generadas por {model_name}', fontsize=16)

    for i in range(num_images):
        row = i // 5
        col = i % 5
        axes[row, col].imshow(images[i].cpu().squeeze(), cmap='gray')
        axes[row, col].axis('off')
        axes[row, col].set_title(f'Img {i + 1}')

    plt.tight_layout()
    plt.savefig(f'results/{model_name}/generated_images.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_losses(g_losses, d_losses, model_name):
    """Grafica las curvas de pérdida"""
    plt.figure(figsize=(10, 5))
    plt.plot(g_losses, label='Pérdida Generador')
    plt.plot(d_losses, label='Pérdida Discriminador')
    plt.xlabel('Iteraciones (cada 100 batches)')
    plt.ylabel('Pérdida')
    plt.title(f'Curvas de Pérdida - {model_name}')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'results/{model_name}/loss_curves.png', dpi=300, bbox_inches='tight')
    plt.show()


def main():
    """Función principal de entrenamiento y evaluación"""
    print("=== Proyecto GAN vs DC-GAN en Fashion-MNIST ===")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Dispositivo: {device}")

    # Cargar datos
    dataloader = load_fashion_mnist()
    print(f"Dataset cargado: {len(dataloader.dataset)} imágenes")

    # Crear directorio de resultados
    os.makedirs('results', exist_ok=True)

    # ================ ENTRENAR MLP-GAN ================
    print("\n" + "=" * 50)
    print("ENTRENANDO GAN CLÁSICA (MLP)")
    print("=" * 50)

    mlp_gen = MLPGenerator(Z_DIM, IMAGE_SIZE * IMAGE_SIZE).to(device)
    mlp_disc = MLPDiscriminator(IMAGE_SIZE * IMAGE_SIZE).to(device)

    # Aplicar inicialización de pesos
    mlp_gen.apply(weights_init)
    mlp_disc.apply(weights_init)

    mlp_g_losses, mlp_d_losses, mlp_fake_imgs = train_gan(
        mlp_gen, mlp_disc, dataloader, "MLP-GAN"
    )

    # Guardar resultados MLP-GAN
    save_images(mlp_fake_imgs, "MLP-GAN")
    plot_losses(mlp_g_losses, mlp_d_losses, "MLP-GAN")

    # ================ ENTRENAR DC-GAN ================
    print("\n" + "=" * 50)
    print("ENTRENANDO DC-GAN")
    print("=" * 50)

    dc_gen = DCGenerator(Z_DIM).to(device)
    dc_disc = DCDiscriminator().to(device)

    # Aplicar inicialización de pesos
    dc_gen.apply(weights_init)
    dc_disc.apply(weights_init)

    dc_g_losses, dc_d_losses, dc_fake_imgs = train_gan(
        dc_gen, dc_disc, dataloader, "DC-GAN"
    )

    # Guardar resultados DC-GAN
    save_images(dc_fake_imgs, "DC-GAN")
    plot_losses(dc_g_losses, dc_d_losses, "DC-GAN")

    # ================ COMPARACIÓN FINAL ================
    print("\n" + "=" * 50)
    print("COMPARACIÓN DE RESULTADOS")
    print("=" * 50)

    # Crear comparación visual
    fig, axes = plt.subplots(2, 10, figsize=(20, 6))
    fig.suptitle('Comparación: MLP-GAN vs DC-GAN', fontsize=16)

    # Desnormalizar imágenes
    mlp_imgs_norm = (mlp_fake_imgs[:10] + 1) / 2
    dc_imgs_norm = (dc_fake_imgs[:10] + 1) / 2

    for i in range(10):
        # MLP-GAN (fila superior)
        axes[0, i].imshow(mlp_imgs_norm[i].cpu().squeeze(), cmap='gray')
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_ylabel('MLP-GAN', fontsize=12)

        # DC-GAN (fila inferior)
        axes[1, i].imshow(dc_imgs_norm[i].cpu().squeeze(), cmap='gray')
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_ylabel('DC-GAN', fontsize=12)

    plt.tight_layout()
    plt.savefig('results/comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Guardar modelos entrenados
    torch.save(mlp_gen.state_dict(), 'results/mlp_generator.pth')
    torch.save(dc_gen.state_dict(), 'results/dc_generator.pth')

    print("\n✅ Entrenamiento completado!")
    print("📁 Resultados guardados en la carpeta 'results/'")
    print("🖼️  Imágenes generadas y curvas de pérdida disponibles")


if __name__ == "__main__":
    main()
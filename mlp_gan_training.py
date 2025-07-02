# MLP-GAN Training Notebook
# Implementación y entrenamiento de GAN clásica en Fashion-MNIST

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm

# Configuración
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Dispositivo: {device}")

# Hiperparámetros
BATCH_SIZE = 64
LEARNING_RATE = 0.0002
NUM_EPOCHS = 100
Z_DIM = 100
IMAGE_SIZE = 28
BETA1 = 0.5

# Reproducibilidad
torch.manual_seed(42)
np.random.seed(42)


class MLPGenerator(nn.Module):
    """
    Generador MLP para Fashion-MNIST
    Convierte vector de ruido z en imagen 28x28
    """

    def __init__(self, z_dim=100, img_dim=784):
        super(MLPGenerator, self).__init__()
        self.img_dim = img_dim

        self.model = nn.Sequential(
            # Primera capa: expandir desde z_dim a 256
            nn.Linear(z_dim, 256),
            nn.LeakyReLU(0.2, inplace=True),

            # Segunda capa: 256 -> 512
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2, inplace=True),

            # Tercera capa: 512 -> 1024
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2, inplace=True),

            # Capa de salida: 1024 -> img_dim
            nn.Linear(1024, img_dim),
            nn.Tanh()  # Salida en rango [-1, 1]
        )

    def forward(self, z):
        """
        Forward pass del generador
        z: tensor de ruido (batch_size, z_dim)
        return: tensor de imagen (batch_size, img_dim)
        """
        return self.model(z)


class MLPDiscriminator(nn.Module):
    """
    Discriminador MLP para Fashion-MNIST
    Clasifica si una imagen es real o generada
    """

    def __init__(self, img_dim=784):
        super(MLPDiscriminator, self).__init__()

        self.model = nn.Sequential(
            # Primera capa con dropout
            nn.Linear(img_dim, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),

            # Segunda capa
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),

            # Tercera capa
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),

            # Capa de salida
            nn.Linear(256, 1),
            nn.Sigmoid()  # Probabilidad de ser real
        )

    def forward(self, img):
        """
        Forward pass del discriminador
        img: tensor de imagen aplanada (batch_size, img_dim)
        return: probabilidad de ser real (batch_size, 1)
        """
        return self.model(img)


def load_fashion_mnist():
    """Carga el dataset Fashion-MNIST"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Normalizar a [-1, 1]
    ])

    dataset = torchvision.datasets.FashionMNIST(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2
    )

    return dataloader


def train_mlp_gan():
    """Función principal de entrenamiento"""
    print("=== Entrenando MLP-GAN ===")

    # Cargar datos
    dataloader = load_fashion_mnist()
    print(f"Dataset cargado: {len(dataloader.dataset)} imágenes")

    # Crear modelos
    generator = MLPGenerator(Z_DIM, IMAGE_SIZE * IMAGE_SIZE).to(device)
    discriminator = MLPDiscriminator(IMAGE_SIZE * IMAGE_SIZE).to(device)

    print(f"Generador - Parámetros: {sum(p.numel() for p in generator.parameters()):,}")
    print(f"Discriminador - Parámetros: {sum(p.numel() for p in discriminator.parameters()):,}")

    # Optimizadores
    opt_gen = optim.Adam(generator.parameters(), lr=LEARNING_RATE, betas=(BETA1, 0.999))
    opt_disc = optim.Adam(discriminator.parameters(), lr=LEARNING_RATE, betas=(BETA1, 0.999))

    # Función de pérdida
    criterion = nn.BCELoss()

    # Tracking de pérdidas
    gen_losses = []
    disc_losses = []

    # Vector de ruido fijo para visualización
    fixed_noise = torch.randn(16, Z_DIM, device=device)

    print("Iniciando entrenamiento...")

    for epoch in range(NUM_EPOCHS):
        gen_loss_epoch = 0
        disc_loss_epoch = 0

        # Barra de progreso
        pbar = tqdm(dataloader, desc=f'Época {epoch + 1}/{NUM_EPOCHS}')

        for batch_idx, (real_imgs, _) in enumerate(pbar):
            batch_size = real_imgs.shape[0]
            real_imgs = real_imgs.view(batch_size, -1).to(device)

            # Etiquetas
            real_labels = torch.ones(batch_size, 1, device=device)
            fake_labels = torch.zeros(batch_size, 1, device=device)

            # ===============================
            # Entrenar Discriminador
            # ===============================
            discriminator.zero_grad()

            # Pérdida con imágenes reales
            real_pred = discriminator(real_imgs)
            real_loss = criterion(real_pred, real_labels)

            # Generar imágenes falsas
            noise = torch.randn(batch_size, Z_DIM, device=device)
            fake_imgs = generator(noise)

            # Pérdida con imágenes falsas
            fake_pred = discriminator(fake_imgs.detach())
            fake_loss = criterion(fake_pred, fake_labels)

            # Pérdida total del discriminador
            disc_loss = real_loss + fake_loss
            disc_loss.backward()
            opt_disc.step()

            # ===============================
            # Entrenar Generador
            # ===============================
            generator.zero_grad()

            # El generador quiere que el discriminador clasifique
            # las imágenes falsas como reales
            fake_pred = discriminator(fake_imgs)
            gen_loss = criterion(fake_pred, real_labels)

            gen_loss.backward()
            opt_gen.step()

            # Actualizar métricas
            gen_loss_epoch += gen_loss.item()
            disc_loss_epoch += disc_loss.item()

            # Actualizar barra de progreso
            pbar.set_postfix({
                'D_loss': f'{disc_loss.item():.4f}',
                'G_loss': f'{gen_loss.item():.4f}'
            })

        # Promediar pérdidas de la época
        gen_loss_epoch /= len(dataloader)
        disc_loss_epoch /= len(dataloader)

        gen_losses.append(gen_loss_epoch)
        disc_losses.append(disc_loss_epoch)

        # Imprimir estadísticas cada 10 épocas
        if (epoch + 1) % 10 == 0:
            print(f'Época [{epoch + 1}/{NUM_EPOCHS}]')
            print(f'  Pérdida Discriminador: {disc_loss_epoch:.4f}')
            print(f'  Pérdida Generador: {gen_loss_epoch:.4f}')

            # Generar y mostrar imágenes de muestra
            with torch.no_grad():
                fake_sample = generator(fixed_noise)
                fake_sample = fake_sample.view(-1, 1, 28, 28)
                show_sample_images(fake_sample, epoch + 1)

    return generator, discriminator, gen_losses, disc_losses


def show_sample_images(images, epoch):
    """Muestra una grilla de imágenes generadas"""
    # Desnormalizar de [-1, 1] a [0, 1]
    images = (images + 1) / 2

    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    fig.suptitle(f'Imágenes Generadas - Época {epoch}', fontsize=14)

    for i in range(16):
        row = i // 4
        col = i % 4
        axes[row, col].imshow(images[i].cpu().squeeze(), cmap='gray')
        axes[row, col].axis('off')

    plt.tight_layout()
    plt.show()


def plot_training_curves(gen_losses, disc_losses):
    """Grafica las curvas de entrenamiento"""
    plt.figure(figsize=(12, 5))

    # Pérdidas
    plt.subplot(1, 2, 1)
    plt.plot(gen_losses, label='Generador', color='blue')
    plt.plot(disc_losses, label='Discriminador', color='red')
    plt.xlabel('Épocas')
    plt.ylabel('Pérdida')
    plt.title('Curvas de Pérdida - MLP-GAN')
    plt.legend()
    plt.grid(True)

    # Suavizado con media móvil
    plt.subplot(1, 2, 2)
    window = 5
    gen_smooth = np.convolve(gen_losses, np.ones(window) / window, mode='valid')
    disc_smooth = np.convolve(disc_losses, np.ones(window) / window, mode='valid')

    plt.plot(gen_smooth, label='Generador (suavizado)', color='blue')
    plt.plot(disc_smooth, label='Discriminador (suavizado)', color='red')
    plt.xlabel('Épocas')
    plt.ylabel('Pérdida')
    plt.title('Curvas Suavizadas')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def evaluate_model(generator):
    """Evalúa el modelo generador entrenado"""
    generator.eval()

    print("=== Evaluación del Modelo ===")

    # Generar batch de imágenes
    with torch.no_grad():
        noise = torch.randn(64, Z_DIM, device=device)
        fake_imgs = generator(noise)
        fake_imgs = fake_imgs.view(-1, 1, 28, 28)
        fake_imgs = (fake_imgs + 1) / 2  # Desnormalizar

    # Mostrar galería de imágenes generadas
    fig, axes = plt.subplots(8, 8, figsize=(12, 12))
    fig.suptitle('Galería de Imágenes Generadas - MLP-GAN', fontsize=16)

    for i in range(64):
        row = i // 8
        col = i % 8
        axes[row, col].imshow(fake_imgs[i].cpu().squeeze(), cmap='gray')
        axes[row, col].axis('off')

    plt.tight_layout()
    plt.show()

    # Estadísticas básicas
    pixel_mean = fake_imgs.mean().item()
    pixel_std = fake_imgs.std().item()

    print(f"Estadísticas de píxeles generados:")
    print(f"  Media: {pixel_mean:.4f}")
    print(f"  Desviación estándar: {pixel_std:.4f}")
    print(f"  Rango: [{fake_imgs.min().item():.4f}, {fake_imgs.max().item():.4f}]")


def main():
    """Función principal del notebook"""
    print("🚀 Iniciando entrenamiento MLP-GAN en Fashion-MNIST")
    print(f"Configuración:")
    print(f"  - Dispositivo: {device}")
    print(f"  - Batch size: {BATCH_SIZE}")
    print(f"  - Learning rate: {LEARNING_RATE}")
    print(f"  - Épocas: {NUM_EPOCHS}")
    print(f"  - Dimensión z: {Z_DIM}")

    # Entrenar modelo
    generator, discriminator, gen_losses, disc_losses = train_mlp_gan()

    # Visualizar curvas de entrenamiento
    plot_training_curves(gen_losses, disc_losses)

    # Evaluar modelo final
    evaluate_model(generator)

    # Guardar modelos
    torch.save(generator.state_dict(), 'mlp_generator.pth')
    torch.save(discriminator.state_dict(), 'mlp_discriminator.pth')

    print("✅ Entrenamiento completado!")
    print("📁 Modelos guardados como 'mlp_generator.pth' y 'mlp_discriminator.pth'")


# Ejecutar solo si es el script principal
if __name__ == "__main__":
    main()
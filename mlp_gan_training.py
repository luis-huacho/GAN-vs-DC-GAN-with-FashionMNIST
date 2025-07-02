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


def weights_init(m):
    """Inicialización de pesos mejorada para ambos tipos de red"""
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        # Para capas lineales, usar inicialización Xavier
        nn.init.xavier_normal_(m.weight.data, gain=0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)
    elif classname.find('Conv') != -1:
        # Para futuras extensiones con capas convolucionales
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


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
    """Función principal de entrenamiento con mejoras en estabilidad"""
    print("=== Entrenando MLP-GAN con Mejoras ===")

    # Cargar datos
    dataloader = load_fashion_mnist()
    print(f"Dataset cargado: {len(dataloader.dataset)} imágenes")

    # Crear modelos
    generator = MLPGenerator(Z_DIM, IMAGE_SIZE * IMAGE_SIZE).to(device)
    discriminator = MLPDiscriminator(IMAGE_SIZE * IMAGE_SIZE).to(device)

    # Aplicar inicialización mejorada
    print("Aplicando inicialización de pesos...")
    generator.apply(weights_init)
    discriminator.apply(weights_init)

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
        discriminator_too_strong = 0  # Contador para diagnóstico

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

            # MEJORA: Evitar entrenamiento excesivo del discriminador
            if disc_loss.item() > 0.1:  # Solo entrenar si no es demasiado fuerte
                disc_loss.backward()
                opt_disc.step()
            else:
                discriminator_too_strong += 1

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
                'G_loss': f'{gen_loss.item():.4f}',
                'D_strong': discriminator_too_strong
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
            print(f'  Discriminador demasiado fuerte: {discriminator_too_strong}/{len(dataloader)} batches')

            # Análisis de equilibrio
            loss_ratio = gen_loss_epoch / max(disc_loss_epoch, 1e-8)
            if loss_ratio > 5:
                print(f'  ⚠️ Discriminador dominando (ratio G/D: {loss_ratio:.2f})')
            elif loss_ratio < 0.2:
                print(f'  ⚠️ Generador dominando (ratio G/D: {loss_ratio:.2f})')
            else:
                print(f'  ✅ Entrenamiento balanceado (ratio G/D: {loss_ratio:.2f})')

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
    fig.suptitle(f'MLP-GAN - Imágenes Generadas - Época {epoch}', fontsize=14)

    for i in range(16):
        row = i // 4
        col = i % 4
        axes[row, col].imshow(images[i].cpu().squeeze(), cmap='gray')
        axes[row, col].axis('off')

    plt.tight_layout()
    plt.show()


def plot_training_curves(gen_losses, disc_losses):
    """Grafica las curvas de entrenamiento con análisis mejorado"""
    plt.figure(figsize=(15, 5))

    # Pérdidas
    plt.subplot(1, 3, 1)
    plt.plot(gen_losses, label='Generador', color='blue', alpha=0.7)
    plt.plot(disc_losses, label='Discriminador', color='red', alpha=0.7)
    plt.xlabel('Épocas')
    plt.ylabel('Pérdida')
    plt.title('Curvas de Pérdida - MLP-GAN')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Suavizado con media móvil
    plt.subplot(1, 3, 2)
    window = 5
    if len(gen_losses) >= window:
        gen_smooth = np.convolve(gen_losses, np.ones(window) / window, mode='valid')
        disc_smooth = np.convolve(disc_losses, np.ones(window) / window, mode='valid')

        plt.plot(gen_smooth, label='Generador (suavizado)', color='blue')
        plt.plot(disc_smooth, label='Discriminador (suavizado)', color='red')
        plt.xlabel('Épocas')
        plt.ylabel('Pérdida')
        plt.title('Curvas Suavizadas')
        plt.legend()
        plt.grid(True, alpha=0.3)

    # Ratio de pérdidas (indicador de equilibrio)
    plt.subplot(1, 3, 3)
    ratios = [g / max(d, 1e-8) for g, d in zip(gen_losses, disc_losses)]
    plt.plot(ratios, color='green', alpha=0.7)
    plt.axhline(y=1, color='black', linestyle='--', alpha=0.5, label='Equilibrio ideal')
    plt.xlabel('Épocas')
    plt.ylabel('Ratio G/D')
    plt.title('Equilibrio de Entrenamiento')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, max(10, max(ratios) * 1.1))

    plt.tight_layout()
    plt.show()


def evaluate_model(generator):
    """Evalúa el modelo generador entrenado con métricas mejoradas"""
    generator.eval()

    print("=== Evaluación del Modelo MLP-GAN ===")

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

    # Estadísticas detalladas
    pixel_mean = fake_imgs.mean().item()
    pixel_std = fake_imgs.std().item()
    pixel_min = fake_imgs.min().item()
    pixel_max = fake_imgs.max().item()

    print(f"Estadísticas de píxeles generados:")
    print(f"  Media: {pixel_mean:.4f}")
    print(f"  Desviación estándar: {pixel_std:.4f}")
    print(f"  Rango: [{pixel_min:.4f}, {pixel_max:.4f}]")

    # Análisis de calidad
    if pixel_std < 0.1:
        print(f"  ⚠️ Advertencia: Baja variación en píxeles (posible colapso)")
    elif pixel_std > 0.4:
        print(f"  ⚠️ Advertencia: Alta variación en píxeles (posible ruido)")
    else:
        print(f"  ✅ Variación de píxeles en rango saludable")

    # Análisis de diversidad simple
    img_flat = fake_imgs.view(64, -1)
    distances = torch.cdist(img_flat, img_flat)
    mean_distance = distances.mean().item()

    print(f"  Diversidad promedio: {mean_distance:.4f}")
    if mean_distance < 0.1:
        print(f"  ⚠️ Advertencia: Baja diversidad (posible mode collapse)")
    else:
        print(f"  ✅ Diversidad aceptable")


def save_training_results(generator, discriminator, gen_losses, disc_losses):
    """Guarda los resultados del entrenamiento"""
    # Crear directorio si no existe
    os.makedirs('results/MLP-GAN', exist_ok=True)

    # Guardar modelos
    torch.save(generator.state_dict(), 'results/MLP-GAN/mlp_generator.pth')
    torch.save(discriminator.state_dict(), 'results/MLP-GAN/mlp_discriminator.pth')

    # Guardar pérdidas
    import json
    losses_data = {
        'generator_losses': gen_losses,
        'discriminator_losses': disc_losses,
        'hyperparameters': {
            'batch_size': BATCH_SIZE,
            'learning_rate': LEARNING_RATE,
            'num_epochs': NUM_EPOCHS,
            'z_dim': Z_DIM,
            'beta1': BETA1
        }
    }

    with open('results/MLP-GAN/training_losses.json', 'w') as f:
        json.dump(losses_data, f, indent=2)

    print("💾 Resultados guardados en results/MLP-GAN/")


def main():
    """Función principal del notebook"""
    print("🚀 Iniciando entrenamiento MLP-GAN mejorado en Fashion-MNIST")
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

    # Guardar resultados
    save_training_results(generator, discriminator, gen_losses, disc_losses)

    # Análisis final
    print("\n" + "=" * 50)
    print("ANÁLISIS FINAL DEL ENTRENAMIENTO")
    print("=" * 50)

    final_g_loss = gen_losses[-1]
    final_d_loss = disc_losses[-1]
    final_ratio = final_g_loss / max(final_d_loss, 1e-8)

    print(f"Pérdida final del Generador: {final_g_loss:.4f}")
    print(f"Pérdida final del Discriminador: {final_d_loss:.4f}")
    print(f"Ratio final G/D: {final_ratio:.4f}")

    # Estabilidad en las últimas épocas
    last_10_g = gen_losses[-10:] if len(gen_losses) >= 10 else gen_losses
    last_10_d = disc_losses[-10:] if len(disc_losses) >= 10 else disc_losses

    g_stability = np.std(last_10_g)
    d_stability = np.std(last_10_d)

    print(f"Estabilidad del Generador (últimas épocas): {g_stability:.4f}")
    print(f"Estabilidad del Discriminador (últimas épocas): {d_stability:.4f}")

    if g_stability < 0.1 and d_stability < 0.1:
        print("✅ Entrenamiento estable conseguido")
    else:
        print("⚠️ Entrenamiento todavía muestra inestabilidad")

    print("✅ Entrenamiento completado!")
    print("📁 Modelos guardados como 'results/MLP-GAN/mlp_generator.pth' y 'mlp_discriminator.pth'")


# Ejecutar solo si es el script principal
if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Script de configuración para el proyecto GAN vs DC-GAN
Instala dependencias y configura el entorno
"""

import os
import sys
import subprocess
import pkg_resources
from pathlib import Path


# Verificar versión de Python
def check_python_version():
    """Verifica que se esté usando Python 3.12+"""
    version = sys.version_info
    if version < (3, 12):
        print(f"❌ Se requiere Python 3.12+, pero se está usando {version.major}.{version.minor}")
        print("Por favor, actualiza Python antes de continuar.")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} detectado")
    return True


# Dependencias requeridas
REQUIRED_PACKAGES = {
    'torch': '>=2.0.0',
    'torchvision': '>=0.15.0',
    'matplotlib': '>=3.7.0',
    'numpy': '>=1.24.0',
    'scipy': '>=1.10.0',
    'scikit-learn': '>=1.3.0',
    'seaborn': '>=0.12.0',
    'tqdm': '>=4.65.0',
    'Pillow': '>=10.0.0'
}


def check_gpu_support():
    """Verifica soporte de GPU/CUDA"""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            print(f"✅ CUDA disponible - {gpu_count} GPU(s) detectada(s)")
            print(f"   GPU Principal: {gpu_name}")
            return True
        else:
            print("⚠️  CUDA no disponible - se usará CPU")
            print("   Para mejor rendimiento, instala PyTorch con soporte CUDA")
            return False
    except ImportError:
        print("⚠️  PyTorch no instalado - no se puede verificar GPU")
        return False


def install_package(package, version_spec=""):
    """Instala un paquete usando pip"""
    try:
        if version_spec:
            package_spec = f"{package}{version_spec}"
        else:
            package_spec = package

        print(f"📦 Instalando {package_spec}...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install",
            package_spec, "--upgrade", "--quiet"
        ])
        print(f"✅ {package} instalado correctamente")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error instalando {package}: {e}")
        return False


def check_and_install_dependencies():
    """Verifica e instala dependencias faltantes"""
    print("\n🔍 Verificando dependencias...")

    missing_packages = []

    for package, version_spec in REQUIRED_PACKAGES.items():
        try:
            pkg_resources.require(f"{package}{version_spec}")
            print(f"✅ {package} ya instalado")
        except (pkg_resources.DistributionNotFound, pkg_resources.VersionConflict):
            missing_packages.append((package, version_spec))

    if missing_packages:
        print(f"\n📥 Instalando {len(missing_packages)} paquete(s) faltante(s)...")

        for package, version_spec in missing_packages:
            if not install_package(package, version_spec):
                print(f"❌ No se pudo instalar {package}")
                return False

    print("✅ Todas las dependencias están instaladas")
    return True


def create_project_structure():
    """Crea la estructura de directorios del proyecto"""
    print("\n📁 Creando estructura del proyecto...")

    directories = [
        'data',
        'results',
        'results/MLP-GAN',
        'results/DC-GAN',
        'models',
        'notebooks',
        'utils',
        'checkpoints'
    ]

    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Directorio creado: {directory}/")

    return True


def create_requirements_file():
    """Crea archivo requirements.txt"""
    requirements_content = """# Proyecto GAN vs DC-GAN - Fashion-MNIST
# Compatibilidad: Python 3.12+

torch>=2.0.0
torchvision>=0.15.0
matplotlib>=3.7.0
numpy>=1.24.0
scipy>=1.10.0
scikit-learn>=1.3.0
seaborn>=0.12.0
tqdm>=4.65.0
Pillow>=10.0.0

# Opcional para análisis avanzado
jupyterlab>=4.0.0
tensorboard>=2.13.0
wandb>=0.15.0
"""

    with open('requirements.txt', 'w') as f:
        f.write(requirements_content)

    print("✅ Archivo requirements.txt creado")


def create_config_file():
    """Crea archivo de configuración"""
    config_content = """# Configuración del proyecto GAN vs DC-GAN

[DEFAULT]
# Configuración de entrenamiento
batch_size = 64
learning_rate = 0.0002
num_epochs = 100
z_dim = 100
beta1 = 0.5

# Configuración de datos
data_dir = ./data
results_dir = ./results
checkpoints_dir = ./checkpoints

# Configuración de dispositivo
device = auto  # auto, cpu, cuda

# Configuración de reproducibilidad
random_seed = 42

# Configuración de logging
log_level = INFO
save_frequency = 10  # Guardar cada N épocas

[MLP_GAN]
# Configuración específica para MLP-GAN
hidden_dims = [256, 512, 1024]
dropout_rate = 0.3

[DC_GAN]
# Configuración específica para DC-GAN
features_g = 64
features_d = 64
use_batch_norm = true

[EVALUATION]
# Configuración de evaluación
num_samples_eval = 1000
fid_batch_size = 64
is_splits = 10
"""

    with open('config.ini', 'w') as f:
        f.write(config_content)

    print("✅ Archivo config.ini creado")


def create_readme():
    """Crea archivo README.md"""
    readme_content = """# GAN vs DC-GAN en Fashion-MNIST

Comparación de arquitecturas GAN clásica y DC-GAN aplicadas al dataset Fashion-MNIST.

## 🚀 Inicio Rápido

### Instalación
```bash
# Clonar repositorio
git clone <url-repositorio>
cd gan-dcgan-fashion-mnist

# Ejecutar configuración automática
python setup.py

# O instalar manualmente
pip install -r requirements.txt
```

### Entrenamiento
```bash
# Entrenar ambos modelos
python main.py

# Solo MLP-GAN
python -c "from main import train_mlp_gan; train_mlp_gan()"

# Solo DC-GAN  
python -c "from main import train_dc_gan; train_dc_gan()"
```

### Evaluación
```bash
# Generar reporte comparativo
python -c "from utils.evaluation import main_evaluation; main_evaluation()"
```

## 📊 Resultados

Los resultados se guardan en:
- `results/MLP-GAN/` - Imágenes y gráficos de MLP-GAN
- `results/DC-GAN/` - Imágenes y gráficos de DC-GAN  
- `results/comparison.png` - Comparación visual
- `training_report.html` - Reporte completo

## 🔧 Configuración

Edita `config.ini` para modificar hiperparámetros:
- Batch size, learning rate, épocas
- Dimensiones de las redes
- Configuración de dispositivo (CPU/GPU)

## 📁 Estructura del Proyecto

```
gan-dcgan-fashion-mnist/
├── main.py              # Script principal
├── setup.py             # Configuración automática
├── config.ini           # Configuración
├── requirements.txt     # Dependencias
├── models/              # Definiciones de modelos
├── utils/               # Utilidades de evaluación
├── results/             # Resultados de entrenamiento
├── notebooks/           # Jupyter notebooks
└── checkpoints/         # Modelos guardados
```

## 🎯 Objetivos del Proyecto

1. **Implementar** GAN clásica (MLP) y DC-GAN
2. **Entrenar** ambos modelos en Fashion-MNIST
3. **Comparar** calidad visual y estabilidad
4. **Evaluar** métricas objetivas (FID, IS)
5. **Analizar** ventajas y limitaciones

## 📖 Documentación

Ver `informe.md` para análisis detallado y conclusiones.

## ⚙️ Requisitos del Sistema

- **Python**: 3.12+
- **RAM**: 8GB (16GB recomendado)
- **GPU**: NVIDIA GTX 1060+ (opcional pero recomendado)
- **Espacio**: 2GB libres

## 🤝 Contribuciones

1. Fork del repositorio
2. Crear rama: `git checkout -b feature/nueva-funcionalidad`
3. Commit: `git commit -am 'Agregar nueva funcionalidad'`
4. Push: `git push origin feature/nueva-funcionalidad`
5. Pull Request

## 📄 Licencia

Este proyecto es para fines educativos. Ver LICENSE para detalles.
"""

    with open('README.md', 'w') as f:
        f.write(readme_content)

    print("✅ Archivo README.md creado")


def run_system_diagnostics():
    """Ejecuta diagnósticos del sistema"""
    print("\n🔧 Ejecutando diagnósticos del sistema...")

    # Información del sistema
    print(f"Sistema operativo: {os.name}")
    print(f"Arquitectura: {os.uname().machine if hasattr(os, 'uname') else 'N/A'}")

    # Memoria disponible (aproximada)
    try:
        import psutil
        memory = psutil.virtual_memory()
        print(f"RAM total: {memory.total / (1024 ** 3):.1f} GB")
        print(f"RAM disponible: {memory.available / (1024 ** 3):.1f} GB")
    except ImportError:
        print("RAM: No se pudo determinar (instalar psutil para detalles)")

    # Verificar GPU
    gpu_available = check_gpu_support()

    return gpu_available


def main():
    """Función principal de configuración"""
    print("🎯 CONFIGURACIÓN DEL PROYECTO GAN vs DC-GAN")
    print("=" * 50)

    # Verificar Python
    if not check_python_version():
        sys.exit(1)

    # Diagnósticos del sistema
    gpu_available = run_system_diagnostics()

    # Instalar dependencias
    if not check_and_install_dependencies():
        print("❌ Error instalando dependencias")
        sys.exit(1)

    # Crear estructura del proyecto
    create_project_structure()

    # Crear archivos de configuración
    create_requirements_file()
    create_config_file()
    create_readme()

    # Verificaciones finales
    print("\n🧪 Verificaciones finales...")

    try:
        import torch
        import torchvision
        import matplotlib
        print("✅ Importaciones principales exitosas")

        # Test simple de PyTorch
        x = torch.randn(2, 3)
        y = torch.nn.Linear(3, 1)(x)
        print("✅ Test básico de PyTorch exitoso")

    except Exception as e:
        print(f"❌ Error en verificación final: {e}")
        sys.exit(1)

    # Resumen final
    print("\n" + "=" * 50)
    print("🎉 CONFIGURACIÓN COMPLETADA EXITOSAMENTE!")
    print("=" * 50)

    print("\n📋 Resumen:")
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} configurado")
    print("✅ Dependencias instaladas")
    print("✅ Estructura del proyecto creada")
    print("✅ Archivos de configuración generados")
    print(f"{'✅' if gpu_available else '⚠️ '} GPU: {'Disponible' if gpu_available else 'No disponible (usará CPU)'}")

    print("\n🚀 Próximos pasos:")
    print("1. Ejecutar: python main.py")
    print("2. Revisar resultados en ./results/")
    print("3. Leer informe completo en informe.md")

    if not gpu_available:
        print("\n💡 Consejo: Para mejor rendimiento, considera usar Google Colab o una GPU local")

    print("\n📚 Documentación disponible:")
    print("- README.md: Guía de uso")
    print("- config.ini: Configuración de hiperparámetros")
    print("- requirements.txt: Lista de dependencias")


def create_colab_notebook():
    """Crea notebook para Google Colab"""
    colab_content = '''{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "header"
   },
   "source": [
    "# 🎯 GAN vs DC-GAN en Fashion-MNIST\\n",
    "\\n",
    "Comparación de arquitecturas GAN clásica y DC-GAN aplicadas al dataset Fashion-MNIST.\\n",
    "\\n",
    "**Objetivos:**\\n",
    "- Implementar y entrenar ambas arquitecturas\\n",
    "- Comparar calidad visual y estabilidad\\n",
    "- Evaluar métricas objetivas\\n",
    "\\n",
    "**Tiempo estimado:** 30-45 minutos en GPU"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "setup"
   },
   "source": [
    "## 🔧 Configuración Inicial"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "id": "install"
   },
   "outputs": [],
   "source": [
    "# Instalar dependencias\\n",
    "!pip install torch torchvision matplotlib numpy scipy scikit-learn seaborn tqdm\\n",
    "\\n",
    "# Verificar GPU\\n",
    "import torch\\n",
    "print(f'GPU disponible: {torch.cuda.is_available()}')\\n",
    "if torch.cuda.is_available():\\n",
    "    print(f'GPU: {torch.cuda.get_device_name(0)}')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "id": "imports"
   },
   "outputs": [],
   "source": [
    "# Importaciones principales\\n",
    "import torch\\n",
    "import torch.nn as nn\\n",
    "import torch.optim as optim\\n",
    "import torchvision\\n",
    "import torchvision.transforms as transforms\\n",
    "from torch.utils.data import DataLoader\\n",
    "import matplotlib.pyplot as plt\\n",
    "import numpy as np\\n",
    "from tqdm import tqdm\\n",
    "\\n",
    "# Configuración\\n",
    "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\\n",
    "torch.manual_seed(42)\\n",
    "np.random.seed(42)\\n",
    "\\n",
    "print(f'Dispositivo: {device}')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "data"
   },
   "source": [
    "## 📊 Carga de Datos"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "id": "load_data"
   },
   "outputs": [],
   "source": [
    "# Cargar Fashion-MNIST\\n",
    "transform = transforms.Compose([\\n",
    "    transforms.ToTensor(),\\n",
    "    transforms.Normalize((0.5,), (0.5,))\\n",
    "])\\n",
    "\\n",
    "dataset = torchvision.datasets.FashionMNIST(\\n",
    "    root='./data', train=True, download=True, transform=transform\\n",
    ")\\n",
    "\\n",
    "dataloader = DataLoader(dataset, batch_size=64, shuffle=True)\\n",
    "\\n",
    "print(f'Dataset cargado: {len(dataset)} imágenes')\\n",
    "\\n",
    "# Mostrar muestras\\n",
    "examples = iter(dataloader)\\n",
    "example_data, example_targets = next(examples)\\n",
    "\\n",
    "fig, axes = plt.subplots(2, 5, figsize=(12, 6))\\n",
    "for i in range(10):\\n",
    "    row, col = i // 5, i % 5\\n",
    "    img = (example_data[i] + 1) / 2  # Desnormalizar\\n",
    "    axes[row, col].imshow(img.squeeze(), cmap='gray')\\n",
    "    axes[row, col].set_title(f'Clase: {example_targets[i]}')\\n",
    "    axes[row, col].axis('off')\\n",
    "plt.suptitle('Muestras de Fashion-MNIST')\\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "models"
   },
   "source": [
    "## 🧠 Definición de Modelos"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "id": "mlp_models"
   },
   "outputs": [],
   "source": [
    "# Pegue aquí el código de las clases MLPGenerator y MLPDiscriminator del archivo principal"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "id": "dc_models"
   },
   "outputs": [],
   "source": [
    "# Pegue aquí el código de las clases DCGenerator y DCDiscriminator del archivo principal"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "training"
   },
   "source": [
    "## 🏋️ Entrenamiento"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "id": "train_mlp"
   },
   "outputs": [],
   "source": [
    "# Entrenar MLP-GAN\\n",
    "print('=== Entrenando MLP-GAN ===')\\n",
    "# Pegue aquí la función de entrenamiento adaptada"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "id": "train_dc"
   },
   "outputs": [],
   "source": [
    "# Entrenar DC-GAN\\n",
    "print('=== Entrenando DC-GAN ===')\\n",
    "# Pegue aquí la función de entrenamiento adaptada"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "results"
   },
   "source": [
    "## 📈 Resultados y Comparación"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "id": "comparison"
   },
   "outputs": [],
   "source": [
    "# Comparación visual\\n",
    "# Generar y mostrar imágenes de ambos modelos"
   ]
  }
 ],
 "metadata": {
  "accelerator": "GPU",
  "colab": {
   "gpuType": "T4",
   "provenance": []
  },
  "kernelspec": {
   "display_name": "Python 3",
   "name": "python3"
  },
  "language_info": {
   "name": "python"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 0
}'''

    with open('GAN_vs_DCGAN_Colab.ipynb', 'w') as f:
        f.write(colab_content)

    print("✅ Notebook de Google Colab creado: GAN_vs_DCGAN_Colab.ipynb")


def create_quick_test():
    """Crea script de prueba rápida"""
    test_content = """#!/usr/bin/env python3
\"\"\"
Prueba rápida del entorno de GAN vs DC-GAN
Verifica que todo funcione correctamente
\"\"\"

import torch
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def quick_test():
    \"\"\"Ejecuta prueba rápida del entorno\"\"\"
    print("🧪 PRUEBA RÁPIDA DEL ENTORNO")
    print("=" * 40)

    # Test 1: PyTorch básico
    print("Test 1: PyTorch básico...")
    try:
        x = torch.randn(5, 3)
        y = torch.randn(5, 3)
        z = x + y
        print(f"✅ Operaciones básicas: {z.shape}")
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

    # Test 2: Red neuronal simple
    print("Test 2: Red neuronal...")
    try:
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 1)
        )
        x = torch.randn(32, 10)
        y = model(x)
        print(f"✅ Red neuronal: entrada {x.shape} -> salida {y.shape}")
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

    # Test 3: GPU (si disponible)
    print("Test 3: GPU...")
    if torch.cuda.is_available():
        try:
            device = torch.device('cuda')
            x = torch.randn(10, 10).to(device)
            y = x * 2
            print(f"✅ GPU funcional: {torch.cuda.get_device_name(0)}")
        except Exception as e:
            print(f"⚠️  GPU disponible pero con error: {e}")
    else:
        print("⚠️  GPU no disponible, usando CPU")

    # Test 4: Descarga de datos
    print("Test 4: Descarga de Fashion-MNIST...")
    try:
        transform = torchvision.transforms.ToTensor()
        dataset = torchvision.datasets.FashionMNIST(
            root='./test_data', train=True, download=True, transform=transform
        )
        print(f"✅ Dataset descargado: {len(dataset)} imágenes")

        # Limpiar datos de test
        import shutil
        shutil.rmtree('./test_data', ignore_errors=True)

    except Exception as e:
        print(f"❌ Error descargando datos: {e}")
        return False

    # Test 5: Matplotlib
    print("Test 5: Visualización...")
    try:
        plt.figure(figsize=(4, 3))
        plt.plot([1, 2, 3, 4], [1, 4, 2, 3])
        plt.title('Test Plot')
        plt.close()  # No mostrar
        print("✅ Matplotlib funcional")
    except Exception as e:
        print(f"❌ Error con matplotlib: {e}")
        return False

    # Test 6: Estructura de directorios
    print("Test 6: Estructura de proyecto...")
    required_dirs = ['data', 'results', 'models', 'utils']
    for dir_name in required_dirs:
        if Path(dir_name).exists():
            print(f"✅ Directorio {dir_name}/ existe")
        else:
            print(f"⚠️  Directorio {dir_name}/ faltante")

    print("\\n" + "=" * 40)
    print("🎉 PRUEBA COMPLETADA EXITOSAMENTE")
    print("El entorno está listo para entrenar GANs!")
    print("\\nPara comenzar, ejecuta: python main.py")

    return True

def memory_test():
    \"\"\"Prueba de memoria disponible\"\"\"
    print("\\n🧠 Prueba de memoria...")

    try:
        # Crear tensor grande para probar memoria
        size_mb = 100  # 100 MB
        elements = (size_mb * 1024 * 1024) // 4  # 4 bytes por float32

        x = torch.randn(elements)
        print(f"✅ Memoria CPU: Puede crear tensor de {size_mb}MB")

        if torch.cuda.is_available():
            x_gpu = x.cuda()
            print(f"✅ Memoria GPU: Puede crear tensor de {size_mb}MB")

        del x
        if torch.cuda.is_available():
            del x_gpu
            torch.cuda.empty_cache()

    except Exception as e:
        print(f"⚠️  Advertencia de memoria: {e}")

if __name__ == "__main__":
    success = quick_test()
    memory_test()

    if success:
        print("\\n🚀 ¡Todo listo para comenzar!")
    else:
        print("\\n❌ Hay problemas con el entorno")
        print("Revisa la instalación y vuelve a ejecutar setup.py")
"""

    with open('quick_test.py', 'w') as f:
        f.write(test_content)

    print("✅ Script de prueba rápida creado: quick_test.py")


if __name__ == "__main__":
    try:
        main()

        # Crear archivos adicionales
        print("\n📄 Creando archivos adicionales...")
        create_colab_notebook()
        create_quick_test()

        print("\n🎯 Configuración completa!")
        print("Ejecuta 'python quick_test.py' para verificar el entorno")

    except KeyboardInterrupt:
        print("\n⏹️  Configuración cancelada por el usuario")
    except Exception as e:
        print(f"\n❌ Error inesperado: {e}")
        print("Por favor, reporta este error si persiste")
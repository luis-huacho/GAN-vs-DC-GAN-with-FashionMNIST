#!/usr/bin/env python3
"""
Prueba rápida del entorno de GAN vs DC-GAN
Verifica que todo funcione correctamente
"""

import torch
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def quick_test():
    """Ejecuta prueba rápida del entorno"""
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

    print("\n" + "=" * 40)
    print("🎉 PRUEBA COMPLETADA EXITOSAMENTE")
    print("El entorno está listo para entrenar GANs!")
    print("\nPara comenzar, ejecuta: python main.py")

    return True

def memory_test():
    """Prueba de memoria disponible"""
    print("\n🧠 Prueba de memoria...")

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
        print("\n🚀 ¡Todo listo para comenzar!")
    else:
        print("\n❌ Hay problemas con el entorno")
        print("Revisa la instalación y vuelve a ejecutar setup.py")

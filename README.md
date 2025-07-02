# GAN vs DC-GAN en Fashion-MNIST

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

# Clasificación Sonar: Minas vs. Rocas con PyTorch 🧠

![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)

Este proyecto utiliza una **Red Neuronal Artificial (ANN)** para resolver un problema de clasificación binaria: distinguir si una señal de sonar ha rebotado en un cilindro metálico (una **mina**) o en una roca cilíndrica.

[Image of sonar signal processing and underwater detection]

## 🎯 Descripción del Proyecto

El **Sonar Dataset** es un reto clásico de Machine Learning que consiste en 208 registros. Cada registro contiene **60 variables** que representan la energía en diferentes bandas de frecuencia.

### Desafíos Técnicos:
* **Clases Desbalanceadas:** El dataset contiene 111 patrones de minas y 97 de rocas. Al no estar perfectamente equilibrado, el modelo debe evaluarse con métricas más allá del simple *Accuracy*.
* **Alta Dimensionalidad:** Contamos con 60 características de entrada, lo que requiere una arquitectura de red capaz de generalizar sin caer en el sobreajuste (*overfitting*).

## 🧠 Arquitectura de la Red (MLP)

He implementado un **Perceptrón Multicapa (MLP)** robusto en PyTorch:
* **Entrada:** 60 neuronas (una por cada banda de frecuencia).
* **Capas Ocultas:** Capas densas (Fully Connected) con funciones de activación **ReLU**.
* **Salida:** 1 neurona con activación **Sigmoid** para determinar la probabilidad de pertenecer a la clase "Mina".

## 💻 Acceso Rápido al Código

Puedes explorar el código completo, las explicaciones paso a paso y las gráficas generadas directamente en el cuaderno principal del proyecto haciendo clic en el siguiente enlace:
  
👉 **[Abrir el Jupyter Notebook: `red_neuronal_sonar.ipynb`](/notebooks/red_neuronal_sonar.ipynb)**

> 💡 **Nota:** GitHub renderiza los archivos `.ipynb` de forma nativa, por lo que puedes visualizar todo el código, los comentarios y los resultados de las ejecuciones directamente desde tu navegador sin necesidad de descargar nada.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Alexisfpy/sonar-ai-classification-pytorch/blob/master/notebooks/red_neuronal_sonar.ipynb)

## 🛠️ Instalación y Configuración

El proyecto está gestionado con `pyproject.toml`, siguiendo los estándares modernos de empaquetado de Python en 2026.

1. **Clonar el repositorio:**
   ```bash
   git clone [https://github.com/Alexisfpy/sonar-ai-classification-pytorch.git](https://github.com/Alexisfpy/sonar-ai-classification-pytorch.git)
   cd sonar-ai-classification-pytorch
2. **Crear y activar el entorno virtual**
   Para este proyecto usaremos -> Python 3.12
   ```bash
   uv python pin 3.12
   ```
   Una vez instalado o si ya tenías Python 3, siguiente comando:
    ```bash
    python -m venv .venv
    ```
    En Windows
    ```bash
    .venv\Scripts\activate
    ```
    En Linux/macOs
    ```bash
    source .venv/bin/activate
    ```
4. **Instalar dependencias**

    Este proyecto utiliza un archivo pyproject.toml para gestionar sus paquetes. Con el entorno virtual activado, instala todas las dependencias automáticamente ejecutando:
    ```bash
    pip install .
    ```
    o
    ```bash
    uv sync
    ```
## 🚀 Cómo ejecutarlo
En Visual Studio Code (Recomendado)
1. Abre la carpeta del proyecto en VS Code.

2. Abre el archivo notebooks/sonar_classification.ipynb.

3. Haz clic en "Select Kernel" (arriba a la derecha) y elige el entorno virtual .venv.

4. Ejecuta las celdas para ver el entrenamiento y los resultados en tiempo real.

### Uso de Modelo entrenado 
Puedes cargar los pesos del modelo (.pth) para realizar inferencia rápida:
```python
import torch
model.load_state_dict(torch.load('modelos/modelo_sonar_entrenado.pth'))
model.eval()
```

## 📊 Métricas de Evaluación
Para validar el rendimiento frente al desbalanceo de clases, se incluyen:
 - **Matriz de Confusión**: Para monitorizar Falsos Negativos (minas no detectadas).
 - **Curvas de Loss/Accuracy**: Visualización del proceso de aprendizaje.
 - **F1-Score**: Como métrica principal de equilibrio entre precisión y exhaustividad.

## 📂 Estructura del Repositorio
```text
├── data/               # Dataset sonar.all-data
├── modelos/            # Pesos del modelo guardados (.pth)
├── notebooks/          # Notebook interactivo con la solución
├── pyproject.toml      # Configuración de dependencias
└── README.md           # Documentación del proyecto
```

## 📄 Licencia

Este proyecto está bajo la Licencia MIT - mira el archivo [LICENSE](LICENSE) para más detalles.

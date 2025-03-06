# README - Redes neuronales con keras

---

## Descripción General

Este proyecto implementa una red neuronal utilizando **TensorFlow** y **Keras** para clasificar imágenes de dígitos escritos a mano del conjunto de datos MNIST. El objetivo es entrenar un modelo que pueda reconocer con alta precisión los dígitos del 0 al 9. Este proyecto fue realizado con Python 3.12

---

## Requisitos Previos

Antes de ejecutar el código, asegúrate de tener instaladas las siguientes bibliotecas:

```bash
pip install numpy keras tensorflow matplotlib
```

Opcionalmente, puedes usar un entorno virtual para aislar dependencias:

```bash
python -m venv env
source env/bin/activate  # En Linux/macOS
env\Scripts\activate  # En Windows
pip install -r requirements.txt
```

---

## Estructura del Proyecto

```
📂 neural_network_keras/
├── 📂 src/                      
│   ├── neural_network_keras.py   # Código fuente
├── main.py                       # Script principal para ejecutar el modelo
├── requirements.txt              # Dependencias del proyecto
├── README.md                     # Documentación del proyecto
```

---

## Implementación del Modelo

### Carga de Datos

El código utiliza el conjunto de datos MNIST, que contiene 60,000 imágenes para entrenamiento y 10,000 para prueba.

```python
from keras.datasets import mnist
(train_x, train_y), (test_x, test_y) = mnist.load_data()
```

### Preprocesamiento

Las imágenes se normalizan y se convierten a formato de red neuronal.

```python
train_x = train_x.reshape(60000, 28*28).astype('float32') / 255
test_x = test_x.reshape(10000, 28*28).astype('float32') / 255
```

Las etiquetas se convierten a **one-hot encoding**:

```python
from keras.utils import to_categorical
train_y = to_categorical(train_y, 10)
test_y = to_categorical(test_y, 10)
```

### Definición del Modelo

El modelo consta de:

- Una capa de entrada con 784 nodos (28x28 píxeles aplanados)
- Una capa oculta con 512 neuronas y activación ReLU
- Una capa de salida con 10 neuronas y activación Softmax

```python
from keras.models import Sequential
from keras.layers import Dense, Input

model = Sequential([
    Input(shape=(28*28,)),
    Dense(512, activation='relu'),
    Dense(10, activation='softmax')
])
```

### Compilación del Modelo

Se usa el optimizador RMSprop y la función de pérdida categorical crossentropy.

```python
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
```

### Entrenamiento del Modelo

El modelo se entrena con 8 épocas y lotes de 128 imágenes.

```python
model.fit(train_x, train_y, epochs=8, batch_size=128)
```

### Evaluación del Modelo

Una vez entrenado, se evalúa su rendimiento en el conjunto de prueba:

```python
loss, accuracy = model.evaluate(test_x, test_y)
print(f"Precisión en el conjunto de prueba: {accuracy:.4f}")
```

---

## Ejecución

Para entrenar y evaluar el modelo, ejecuta en la terminal:

```bash
python main.py
```

---

## Conclusión

Este proyecto muestra cómo se puede utilizar Keras para entrenar una red neuronal simple en un problema clásico de clasificación de imágenes.

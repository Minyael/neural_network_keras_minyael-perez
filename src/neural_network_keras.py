import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Input
from keras.utils import to_categorical
from keras.datasets import mnist
import matplotlib.pyplot as plt

def train_neural_network():
    # Cargar datos de entrenamiento y prueba desde MNIST
    (train_data_x, train_labels_y), (test_data_x, test_labels_y) = mnist.load_data()
    
    # Mostrar información sobre los datos cargados
    print("Forma de los datos de entrenamiento:", train_data_x.shape)
    print("Etiqueta del primer ejemplo de entrenamiento:", train_labels_y[1])
    print("Forma de los datos de prueba:", test_data_x.shape)
    
    # Visualizar un ejemplo de imagen de entrenamiento
    plt.imshow(train_data_x[1], cmap="gray")
    plt.title("Ejemplo de imagen de entrenamiento")
    plt.show()
    
    # Normalización de datos:
    # Convertir las imágenes en vectores de 28x28 y normalizar dividiendo entre 255
    x_train = train_data_x.reshape(60000, 28*28).astype('float32') / 255
    y_train = to_categorical(train_labels_y)
    x_test = test_data_x.reshape(10000, 28*28).astype('float32') / 255
    y_test = to_categorical(test_labels_y)
    
    # Definir la arquitectura de la red neuronal
    model = Sequential([
        Input(shape=(28*28,)),
        Dense(512, activation='relu'),
        Dense(10, activation='softmax')
    ])
    
    # Compilación del modelo
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    
    # Mostrar resumen del modelo
    model.summary()
    
    # Entrenamiento del modelo
    model.fit(x_train, y_train, epochs=8, batch_size=128)
    
    # Evaluación del modelo
    loss, accuracy = model.evaluate(x_test, y_test)
    print(f"Precisión en datos de prueba: {accuracy:.4f}")
    
    return model

if __name__ == "__main__":
    train_neural_network()
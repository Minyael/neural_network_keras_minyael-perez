import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Input
from keras.utils import to_categorical
from keras.datasets import mnist
import matplotlib.pyplot as plt

def train_mnist_model():
    # Cargar datos de entrenamiento
    (train_data_x, train_labels_y), (test_data_x, test_labels_y) = mnist.load_data()
    
    # Mostrar la forma de los datos de entrenamiento
    print(train_data_x.shape)
    print(train_labels_y[0])
    
    # Definir la arquitectura de la red
    model = Sequential([
        Input(shape=(28*28,)),
        Dense(512, activation='relu'),
        Dense(10, activation='softmax')
    ])
    
    # Compilar el modelo
    model.compile(
        optimizer='rmsprop', 
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Mostrar resumen del modelo
    model.summary()
    
    # Normalización de los datos
    x_train = train_data_x.reshape(60000, 28*28).astype('float32') / 255
    y_train = to_categorical(train_labels_y)
    
    x_test = test_data_x.reshape(10000, 28*28).astype('float32') / 255
    y_test = to_categorical(test_labels_y)
    
    # Entrenamiento del modelo
    model.fit(x_train, y_train, epochs=8, batch_size=128)
    
    # Evaluación del modelo
    model.evaluate(x_test, y_test)

if __name__ == "__main__":
    train_mnist_model()
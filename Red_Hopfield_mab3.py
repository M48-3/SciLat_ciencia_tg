"""
Implementación en Python de una red Hopfield para reconocimiento de patrones
En este ejemplo, n_units es el número de unidades de procesamiento de la red Hopfield, y patterns es una matriz numpy que contiene los patrones de entrenamiento. En el método train, la red Hopfield aprende los patrones de entrenamiento actualizando los pesos de las conexiones.
En el método predict, se realiza la predicción para un nuevo patrón de entrada input_pattern. La red Hopfield actualiza iterativamente los estados de las unidades de procesamiento hasta que se alcanza un estado de equilibrio estable. El resultado final es el patrón reconocido.
"""

import numpy as np

class HopfieldNetwork:
    def __init__(self, n_units):
        self.n_units = n_units
        self.weights = np.zeros((n_units, n_units))
    
    def train(self, patterns):
        for pattern in patterns:
            self.weights += np.outer(pattern, pattern)
        np.fill_diagonal(self.weights, 0)
    
    def predict(self, input_pattern, n_iters=100):
        state = np.copy(input_pattern)
        for _ in range(n_iters):
            h = np.dot(self.weights, state)
            state = np.where(h > 0, 1, -1)
            if np.all(state == h):
                break
        return state

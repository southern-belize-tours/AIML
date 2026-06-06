import numpy as np

INPUT_SIZE: int = 2


class Perceptron:

    def __init__(self, input_size: int = INPUT_SIZE, learning_rate: float = 0.1):
        self.input_size = input_size
        self.learning_rate = learning_rate
        self.initialize_weights()
        self.bias: float = 0

    def __str__(self):
        weights_str = ", ".join(f"{w:.3f}" for w in self.weights)
        lines = [
            '-' * 20,
            'Perceptron',
            '-' * 20,
            f"Weights: [{weights_str}]",
            f"Bias:    {self.bias:.3f}",
            '-' * 20,
        ]
        return "\n".join(lines)

    def initialize_weights(self):
        self.weights: np.ndarray = np.random.uniform(-1, 1, size=self.input_size)

    def activation_function(self, z: float) -> int:
        return 1 if z >= 0 else 0

    def iterate(self, x: np.ndarray) -> int:
        return self.activation_function(np.dot(x, self.weights) + self.bias)

    def train(self, x: np.ndarray, target: int):
        error = target - self.iterate(x)
        self.weights += self.learning_rate * error * x
        self.bias += self.learning_rate * error

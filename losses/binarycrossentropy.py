import numpy as np

class BinaryCrossEntropy:
    def __init__(self) -> None:
        pass

    def compute(self, y_hat: np.ndarray, y: np.ndarray) -> float:
        """
        Computes the binary cross entropy loss.
            args:
                y: true labels (n_classes, batch_size)
                y_hat: predicted labels (n_classes, batch_size)
            returns:
                binary cross entropy loss
        """
        # TODO: Implement binary cross entropy loss
        batch_size = None
        cost = None
        return np.squeeze(cost)

    def backward(self, y_hat: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Computes the derivative of the binary cross entropy loss.
            args:
                y: true labels (n_classes, batch_size)
                y_hat: predicted labels (n_classes, batch_size)
            returns:
                derivative of the binary cross entropy loss
        """
        # hint: use the np.divide function
        # TODO: Implement backward pass for binary cross entropy loss
        return None


import numpy as np

def sigmoid(Z: np.ndarray) -> np.ndarray:
    """
    Sigmoid activation function.
        args:
           x: input to the activation function
        returns:
            sigmoid(x)
    """
    # TODO: Implement sigmoid activation function
    A = None
    return A

def sigmoid_backward(dA: np.ndarray, Z: np.ndarray) -> np.ndarray:
    """
    Backward pass for sigmoid activation function.
        args:
            dA: derivative of the cost with respect to the activation
            Z: input to the activation function
        returns:
            derivative of the cost with respect to Z
    """
    A = sigmoid(Z)
    # TODO: Implement backward pass for sigmoid activation function
    dZ = None
    return dZ

def relu(Z: np.ndarray) -> np.ndarray:
    """
    ReLU activation function.
        args:
            x: input to the activation function
        returns:
            relu(x)
    """
    # TODO: Implement ReLU activation function
    A = None
    return A

def relu_backward(dA: np.ndarray, Z: np.ndarray) -> np.ndarray:
    """
    Backward pass for ReLU activation function.
        args:
            dA: derivative of the cost with respect to the activation
            Z: input to the activation function
        returns:
            derivative of the cost with respect to Z
    """
    # TODO: Implement backward pass for ReLU activation function
    dZ = None
    dZ[Z <= 0] = 0

    return dZ

def tanh(Z: np.ndarray) -> np.ndarray:
    """
    Tanh activation function.
        args:
            x: input to the activation function
        returns:
            tanh(x)
    """
    # TODO: Implement tanh activation function
    A = None
    return A

def tanh_backward(dA: np.ndarray, Z: np.ndarray) -> np.ndarray:
    """
    Backward pass for tanh activation function.
        args:
            dA: derivative of the cost with respect to the activation
            Z: input to the activation function
        returns:
            derivative of the cost with respect to Z
    """
    A = tanh(Z)
    # TODO: Implement backward pass for tanh activation function
    dZ = None
    return dZ

def linear(Z: np.ndarray) -> np.ndarray:
    """
    Linear activation function.
        args:
            x: input to the activation function
        returns:
            x
    """
    # TODO: Implement linear activation function
    A = None
    return A

def linear_backward(dA: np.ndarray, Z: np.ndarray) -> np.ndarray:
    """
    Backward pass for linear activation function.
        args:
            dA: derivative of the cost with respect to the activation
            Z: input to the activation function
        returns:
            derivative of the cost with respect to Z
    """
    # TODO: Implement backward pass for linear activation function
    dZ = None
    return dZ

def get_activation(activation: str) -> tuple:
    """
    Returns the activation function and its derivative.
        args:
            activation: activation function name
        returns:
            activation function and its derivative
    """
    if activation == 'sigmoid':
        return sigmoid, sigmoid_backward
    elif activation == 'relu':
        return relu, relu_backward
    elif activation == 'tanh':
        return tanh, tanh_backward
    elif activation == 'linear':
        return linear, linear_backward
    else:
        raise ValueError('Activation function not supported')


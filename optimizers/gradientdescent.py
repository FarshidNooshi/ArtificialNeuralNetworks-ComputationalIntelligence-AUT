# TODO: Implement the gradient descent optimizer
class GD:
    def __init__(self, layers_list: dict, learning_rate: float):
        """
        Gradient Descent optimizer.
            args:
                layers_list: dictionary of layers name and layer object
                learning_rate: learning rate
        """
        self.learning_rate = learning_rate
        self.layers = layers_list
    
    def update(self, grads, name):
        """
        Update the parameters of the layer.
            args:
                grads: list of gradients for the weights and bias
                name: name of the layer
            returns:
                params: list of updated parameters
        """
        layer = self.layers[None]
        params = []
        #TODO: Implement gradient descent update
        for None in range(len(grads)):
            params.append(None - None * None)
        return params
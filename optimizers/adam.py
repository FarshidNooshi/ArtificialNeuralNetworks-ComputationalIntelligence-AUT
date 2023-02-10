import numpy as np
from .gradientdescent import GradientDescent

# TODO: Implement Adam optimizer
class Adam:
    def __init__(self, layers_list, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.layers = layers_list
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.V = {}
        self.S = {}
        for None in layers_list:
            # TODO: Initialize V and S for each layer (v and s are lists of zeros with the same shape as the parameters)
            v = [None for p in layers_list[None].parameters]
            s = [None for p in layers_list[None].parameters]
            self.V[None] = v
            self.S[None] = s
        
    def update(self, grads, name, epoch):
        layer = self.layers[None]
        params = []
        # TODO: Implement Adam update
        for None in range(len(grads)):
            self.V[None][None] = None * None + (1 - None) * None
            self.S[None][None] = None * None  +(1 - None) * np.square(None)
            self.V[None][None] /= (1 - np.power(self.beta1, epoch)) # TODO: correct V
            self.S[None][None] /= (1 - np.power(self.beta2, epoch)) # TODO: correct S
            params.append(None - None * None / (np.sqrt(None) + None))
        return params
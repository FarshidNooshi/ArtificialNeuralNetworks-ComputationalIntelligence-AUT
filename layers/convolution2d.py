import numpy as np

class Conv2D:
    def __init__(self, in_channels, out_channels, name, kernel_size=(1, 1), stride=(1, 1), padding=(1, 1), initialize_method="random"):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.name = name
        self.initialize_method = initialize_method

        self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        self.stride = (stride, stride) if isinstance(stride, int) else stride
        self.padding = (padding, padding) if isinstance(padding, int) else padding
        self.parameters = [self.initialize_weights(), self.initialize_bias()]


    def initialize_weights(self):
        """
        Initialize weights.
        returns:
            weights: initialized kernel with shape: (kernel_size[0], kernel_size[1], in_channels, out_channels)
        """
        # TODO: Implement initialization of weights
        
        if self.initialize_method == "random":
            return None * 0.01
        if self.initialize_method == "xavier":
            return None
        if self.initialize_method == "he":
            return None
        else:
            raise ValueError("Invalid initialization method")
    
    def initialize_bias(self):
        """
        Initialize bias.
        returns:
            bias: initialized bias with shape: (1, 1, 1, out_channels)
        
        """
        # TODO: Implement initialization of bias
        return None
    
    def target_shape(self, input_shape):
        """
        Calculate the shape of the output of the convolutional layer.
        args:
            input_shape: shape of the input to the convolutional layer
        returns:
            target_shape: shape of the output of the convolutional layer
        """
        # TODO: Implement calculation of target shape
        H = None
        W = None
        return (H, W)
    
    def pad(self, A, padding, pad_value=0):
        """
        Pad the input with zeros.
        args:
            A: input to be padded
            padding: tuple of padding for height and width
            pad_value: value to pad with
        returns:
            A_padded: padded input
        """
        A_padded = np.pad(A, ((0, 0), (padding[0], padding[0]), (padding[1], padding[1]), (0, 0)), mode="constant", constant_values=(pad_value, pad_value))
        return A_padded
    
    def single_step_convolve(self, a_slic_prev, W, b):
        """
        Convolve a slice of the input with the kernel.
        args:
            a_slic_prev: slice of the input data
            W: kernel
            b: bias
        returns:
            Z: convolved value
        """
        # TODO: Implement single step convolution
        Z = None    # hint: element-wise multiplication
        Z = None    # hint: sum over all elements
        Z = None    # hint: add bias as type float using np.float(None)
        return Z

    def forward(self, A_prev):
        """
        Forward pass for convolutional layer.
            args:
                A_prev: activations from previous layer (or input data)
                A_prev.shape = (batch_size, H_prev, W_prev, C_prev)
            returns:
                A: output of the convolutional layer
        """
        # TODO: Implement forward pass
        W, b = None
        (batch_size, H_prev, W_prev, C_prev) = None
        (kernel_size_h, kernel_size_w, C_prev, C) = None
        stride_h, stride_w = None
        padding_h, padding_w = None
        H, W = None
        Z = None
        A_prev_pad = None # hint: use self.pad()
        for i in range(None):
            for h in range(None):
                h_start = None
                h_end = h_start + None
                for w in range(None):
                    w_start = None
                    w_end = w_start + None
                    for c in range(None):
                        a_slice_prev = A_prev_pad[i, h_start:h_end, w_start:w_end, :]
                        Z[i, h, w, c] = None # hint: use self.single_step_convolve()
        return Z

    def backward(self, dZ, A_prev):
        """
        Backward pass for convolutional layer.
        args:
            dZ: gradient of the cost with respect to the output of the convolutional layer
            A_prev: activations from previous layer (or input data)
            A_prev.shape = (batch_size, H_prev, W_prev, C_prev)
        returns:
            dA_prev: gradient of the cost with respect to the input of the convolutional layer
            gradients: list of gradients with respect to the weights and bias
        """
        # TODO: Implement backward pass
        W, b = None
        (batch_size, H_prev, W_prev, C_prev) = None
        (kernel_size_h, kernel_size_w, C_prev, C) = None
        stride_h, stride_w = None
        padding_h, padding_w = None
        H, W = None
        dA_prev = None  # hint: same shape as A_prev
        dW = None    # hint: same shape as W
        db = None    # hint: same shape as b
        A_prev_pad = None # hint: use self.pad()
        dA_prev_pad = None # hint: use self.pad()
        for i in range(None):
            a_prev_pad = A_prev_pad[i]
            da_prev_pad = dA_prev_pad[i]
            for h in range(None):
                for w in range(None):
                    for c in range(None):
                        h_start = None
                        h_end = h_start + None
                        w_start = None
                        w_end = w_start + None
                        a_slice = a_prev_pad[h_start:h_end, w_start:w_end, :]
                        da_prev_pad += None # hint: use element-wise multiplication of dZ and W
                        dW[..., c] += None # hint: use element-wise multiplication of dZ and a_slice
                        db[..., c] += None # hint: use dZ
            dA_prev[i, :, :, :] = None # hint: remove padding (trick: pad:-pad)
        grads = [dW, db]
        return dA_prev, grads
    
    def update_parameters(self, optimizer, grads):
        """
        Update parameters of the convolutional layer.
        args:
            optimizer: optimizer to use for updating parameters
            grads: list of gradients with respect to the weights and bias
        """
        self.parameters = optimizer.update(grads, self.name)
import numpy as np

class MaxPool2D:
    def __init__(self, kernel_size=(3, 3), stride=(1, 1), mode="max"):
        """
        Max pooling layer.
            args:
                kernel_size: size of the kernel
                stride: stride of the kernel
                mode: max or average
            Question:Why we don't need to set name for the layer?
        """
        self.stride = (stride, stride) if isinstance(stride, int) else stride
        self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        self.mode = mode
    
    def target_shape(self, input_shape):
        """
        Calculate the shape of the output of the layer.
            args:
                input_shape: shape of the input
            returns:
                output_shape: shape of the output
        """
        # TODO: Implement target_shape
        H = None
        W = None
        return H, W
    
    def forward(self, A_prev):
        """
        Forward pass for max pooling layer.
            args:
                A_prev: activations from previous layer (or input data)
            returns:
                A: output of the max pooling layer
        """
        # TODO: Implement forward pass for max pooling layer
        (batch_size, H_prev, W_prev, C_prev) = None
        (f_h, f_w) = None
        strideh, stridew = None
        H, W = None
        A = np.zeros((None, None, None, None))
        for i in range(None):
            for h in range(None):
                h_start = None
                h_end = h_start + None
                for w in range(None):
                    w_start = None
                    w_end = w_start + None
                    for c in range(None):
                        a_prev_slice = A_prev[i, h_start:h_end, w_start:w_end, c]
                        if self.mode == "max":
                            A[i, h, w, c] = None
                        elif self.mode == "average":
                            A[i, h, w, c] = None
                        else:
                            raise ValueError("Invalid mode")

        return A
    
    def create_mask_from_window(self, x):
        """
        Create a mask from an input matrix x, to identify the max entry of x.
            args:
                x: numpy array
            returns:
                mask: numpy array of the same shape as window, contains a True at the position corresponding to the max entry of x.
        """
        # TODO: Implement create_mask_from_window
        mask = x == None
        return mask
    
    def distribute_value(self, dz, shape):
        """
        Distribute the input value in the matrix of dimension shape.
            args:
                dz: input scalar
                shape: the shape (n_H, n_W) of the output matrix for which we want to distribute the value of dz
            returns:
                a: distributed value
        """
        # TODO: Implement distribute_value
        (n_H, n_W) = shape
        average = None
        a = np.ones(shape) * None
        return a
    
    def backward(self, dZ, A_prev):
        """
        Backward pass for max pooling layer.
            args:
                dA: gradient of cost with respect to the output of the max pooling layer
                A_prev: activations from previous layer (or input data)
            returns:
                dA_prev: gradient of cost with respect to the input of the max pooling layer
        """
        # TODO: Implement backward pass for max pooling layer
        (f_h, f_w) = self.kernel_size
        strideh, stridew = self.stride
        batch_size, H_prev, W_prev, C_prev = None
        batch_size, H, W, C = None
        dA_prev = np.zeros((None, None, None, None))
        for i in range(None):
            for h in range(None):
                for w in range(None):
                    for c in range(None):
                        h_start = None
                        h_end = h_start + None
                        w_start = None
                        w_end = w_start + None
                        if self.mode == "max":
                            a_prev_slice = A_prev[i, h_start:h_end, w_start:w_end, c]
                            mask = self.create_mask_from_window(None)
                            dA_prev[i, h_start:h_end, w_start:w_end, c] += np.multiply(None, None)
                        elif self.mode == "average":
                            dz = dZ[i, h, w, c]
                            dA_prev[i, h_start:h_end, w_start:w_end, c] += self.distribute_value(None, None)
                        else:
                            raise ValueError("Invalid mode")
        # Don't change the return
        return dA_prev, None
    
    

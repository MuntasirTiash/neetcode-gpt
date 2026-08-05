import numpy as np
from numpy.typing import NDArray
from typing import Tuple


class Solution:
    def backward(self, x: NDArray[np.float64], w: NDArray[np.float64], b: float, y_true: float) -> Tuple[NDArray[np.float64], float]:
        # x: 1D input array
        # w: 1D weight array
        # b: scalar bias
        # y_true: true target value
        #
        # Forward: z = dot(x, w) + b, y_hat = sigmoid(z)
        # Loss: L = 0.5 * (y_hat - y_true)^2
        # Return: (dL_dw rounded to 5 decimals, dL_db rounded to 5 decimals)
        n = len(w)

        gradient = [0]*n
        z = np.dot(x,w) + b
        predictions = 1 / (1+ np.exp(-z))

        for i in range(n):
            gradient[i] = np.round(np.dot(np.dot((predictions - y_true),predictions*(1-predictions)),x[i]),5)


        b_gradient = np.round(np.dot((predictions - y_true),predictions*(1-predictions)),5)

        return (gradient,b_gradient)

    

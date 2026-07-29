import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array of logits
        # Hint: subtract max(z) for numerical stability before computing exp
        # return np.round(your_answer, 4)
        maximum = max(z)

        z = z - maximum
    
        total = sum(np.exp(z))

        res = np.round(np.exp(z)/total,4)

        return res

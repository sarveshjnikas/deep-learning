import numpy as np
from common import sigmoid

class Model1:
    def __init__(self, nin, seed=7021):
        rng = np.random.default_rng(seed)
        self.w = rng.uniform(-1,1,nin)
        self.b = rng.uniform(-1,1)
        
    def __call__(self, X):
        self.X = X
        self.z = X @ self.w + self.b
        self.a = sigmoid(self.z)
        return self.a
    
    def compute_gradients(self, X, Ypred, Y):
        # THE CHAIN: X -> [w,b] -> z -> sigmoid() -> Ypred -> Loss
        self.dz = (Ypred - Y) / len(X)
        self.dw = self.X.T @ self.dz
        self.db = self.dz.sum()
    
    def update_weights(self, lr):
        self.w -= lr * self.dw
        self.b -= lr * self.db
        
    def parameters(self, ):
        return { 
            "w": self.w,
            "b": self.b
        }
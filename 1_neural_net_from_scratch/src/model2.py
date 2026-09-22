import numpy as np
from common import sigmoid

class Model2:
    def __init__(self, nin, nout=32, seed=7021):
        # INCOMING WOULD BE (N,4) AND OUT WOULD BE (32,1) SO THE LAYER IN BETWEEN SHOULD BE W(4,32) AND B(1,32)
        rng = np.random.default_rng(seed)
        self.w1 = rng.uniform(-1,1,(nin,nout)) # (4,32) WEIGHTS
        self.b1 = rng.uniform(-1,1, nout) # 32 BIASES FOR EACH NEURON
        
        self.w2 = rng.uniform(-1,1, nout) # 32 WEIGHTS
        self.b2 = rng.uniform(-1,1) # 1 SCALAR
        
    def __call__(self, X):
        self.X = X
        self.z1 = self.X @ self.w1 + self.b1
        self.a1 = sigmoid(self.z1)
        self.z2 = self.a1 @ self.w2 + self.b2
        out = sigmoid(self.z2)
        return out
    
    def compute_gradients(self, X, Ypred, Y):
        # THE CHAIN: X -> [w1,b1] -> z1 -> sigmoid() -> a1 -> [w2,b2] -> z2 -> sigmoid() -> Ypred -> Loss
        # THE z2 ONWARDS PART OF THE CHAIN IS IDENTICAL TO MODEL 1. 
        dz2 = (Ypred - Y) / len(X) # (N,)
        self.dw2 = self.a1.T @ dz2 # (32,) 
        self.db2 = dz2.sum()
        
        da1 = dz2[:,None] @ self.w2[None, :] # (N,32) =  (N,) (32,)
        dz1 = da1 * (self.a1) * (np.ones_like(self.a1)-self.a1) # (N,32)
        
        self.dw1 = self.X.T @ dz1 # (4,32)
        self.db1 = dz1.sum(axis=0)   # (32,)
        return
    
    def update_weights(self, lr):
        self.w1 -= lr * self.dw1
        self.b1 -= lr * self.db1
        
        self.w2 -= lr * self.dw2
        self.b2 -= lr * self.db2
        
    def parameters(self, ):
        return { 
            "w1": self.w1,
            "b1": self.b1,
            "w2": self.w2,
            "b2": self.b2
        }
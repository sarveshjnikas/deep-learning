import numpy as np
from common import sigmoid

class Model3:
    def __init__(self, nin, n1=32, nout=16, seed=7021):
        rng = np.random.default_rng(seed)
        self.w1 = rng.uniform(-1, 1, (nin, n1)) # (4,32) WEIGHTS FOR FIRST HIDDEN LAYER
        self.b1 = rng.uniform(-1, 1, (n1)) # 32 BIASES FOR EACH HIDDEN LAYER 1 NEURON
        
        self.w2 = rng.uniform(-1, 1, (n1, nout)) # (32,16) WEIGHTS FOR SECOND HIDDEN LAYER INPUT
        self.b2 = rng.uniform(-1, 1, (nout)) # 16 BIASES FOR EACH HIDDEN LAYER 2 NEURON
        
        self.w3 = rng.uniform(-1, 1, nout) # 16 WEIGHTS FOR OUTPUT FROM EACH NEURON IN THE SECOND HIDDEN LAYER 
        self.b3 = rng.uniform(-1, 1) # 1 SCALAR BIAS FOR THE FINAL OUTPUT

    def __call__(self, X):
        self.X = X # (N,4)
        self.z1 = self.X @ self.w1 + self.b1 # (N,32)
        self.a1 = sigmoid(self.z1) # (N,32)
        self.z2 = self.a1 @ self.w2 + self.b2 # (N,16)
        self.a2 = sigmoid(self.z2) # (N,16)
        self.z3  = self.a2 @ self.w3 + self.b3 # (N,)
        out = sigmoid(self.z3) # (N,)
        return out

    def compute_gradients(self, X, Ypred, Y):
        # DETAILED MANUALLY COMPUTED BACKPROP:
        # THE CHAIN: X -> [w1,b1] -> z1 -> sigmoid -> a1 -> [w2,b2] -> z2 -> sigmoid -> a2 -> [w3,b3] -> z3 -> sigmoid -> Ypred -> Loss
        # ypred = sig(z3) => dypred/dz3 = sig(z3)*(1-sig(z3)) 
        dz3 = (Ypred - Y) / len(X) # (N,): dL/dz3 = dL/dypred * dypred/d(z3) => dz3 = (Ypred - Y)/N
        self.dw3 = self.a2.T @ dz3 # (16,) = (16,N) * (N,)
        self.db3 = dz3.sum() # ALL dz3 SQUASHED INTO ONE SCALAR
       
        da2 = dz3[:, None] @ self.w3[None,:] # WE WANT (N,16) BUT dz3 is (N,) and w3 is (16,) RESHAPE THEM AS (N,1) AND (1,16)
        dz2 = da2 * (self.a2) * (np.ones_like(self.a2)-self.a2) # (N,16)
        self.dw2 = self.a1.T @ dz2 # (32,N) (N,16)
        self.db2 = dz2.sum(axis=0) # (16,) 
        
        da1 = dz2 @ self.w2.T # WE WANT (N,32) = dz2(N,16) * w2(32,16)
        dz1 = da1 * (self.a1) * (np.ones_like(self.a1)-self.a1) # (N,32)
        self.dw1 = self.X.T @ dz1 # (4,32) = (4,N) * (N,32)
        self.db1 = dz1.sum(axis=0) # (32,) = (N,32) COLLAPSED AND SUMMED ALONG ROWS

    def update_weights(self, lr):
        self.w1 -= lr * self.dw1
        self.b1 -= lr * self.db1
        
        self.w2 -= lr * self.dw2
        self.b2 -= lr * self.db2
        
        self.w3 -= lr * self.dw3
        self.b3 -= lr * self.db3
        
    def parameters(self, ):
        return { 
            "w1": self.w1,
            "b1": self.b1,
            "w2": self.w2,
            "b2": self.b2,
            "w3": self.w3,
            "b3": self.b3
        }
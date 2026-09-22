import numpy as np
import os

FEATURES = ['variance', 'skewness', 'curtosis', 'entropy']
INPUT_FEATURE_COUNT = len(FEATURES)
MAX_EPOCHS = 20
MODEL1_LEARNING_RATE = 5 # FOUND USING TUNING AND CHECKING VAL LOSS

MODEL2_LEARNING_RATE = 8 # FOUND USING TUNING AND CHECKING VAL LOSS
MODEL2_HIDDEN_NEURONS = 32

MODEL3_LEARNING_RATE = 8 # FOUND USING TUNING AND CHECKING VAL LOSS
MODEL3_HIDDEN_NEURONS_L1 = 32
MODEL3_HIDDEN_NEURONS_L2 = 16

EPS = 1e-15

def sigmoid(x):
    sig = 1/ (1+ np.exp(-1*x))
    return sig
    
def build_dataset(df):
    X = df[FEATURES].to_numpy(dtype=np.float32) # CONVERT TO PLAIN NP ARRAY AND CAST AS FLOAT
    Y = df['class'].to_numpy(dtype=np.int32) # CONVERT TO PLAIN NP ARRAY AND CAST AS INT
    return X, Y

def compute_loss(Yact, Ypred):
    l = np.clip(Ypred, EPS, 1-EPS)
    loss = -np.mean(Yact*np.log(l) + (1-Yact)*np.log(1-l))
    return loss

def calculate_f1_score(yact, ypred):
    tp = np.sum((yact ==1) & (ypred ==1))
    fp = np.sum((yact ==0) & (ypred ==1))
    fn = np.sum((yact ==1) & (ypred ==0))

    precision = tp /(tp+fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    
    if precision +recall == 0:
        return 0.0
    f1_score = 2 * precision *recall /(precision+recall)
    
    return f1_score

def params_standardizer_fitting(X):
    mean, std = X.mean(axis=0), X.std(axis=0)
    return mean, std

def standardize_data(X, mean, std):
    return (X-mean) / std

def write_log(path, predictions):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w') as f:
        for i, p in enumerate(predictions, start=1):
            f.write(f"Sample_{i} {int(p)}\n")
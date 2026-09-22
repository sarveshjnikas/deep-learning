import pandas as pd
import numpy as np

import pickle
from common import INPUT_FEATURE_COUNT, MODEL2_HIDDEN_NEURONS, MODEL2_LEARNING_RATE, MODEL3_HIDDEN_NEURONS_L1, MODEL3_HIDDEN_NEURONS_L2, MODEL3_LEARNING_RATE, build_dataset, MAX_EPOCHS, MODEL1_LEARNING_RATE, compute_loss, calculate_f1_score, params_standardizer_fitting, standardize_data, write_log
from model1 import Model1
from model2 import Model2
from model3 import Model3

# SEED
S = 20922302228

# DATASET READ AND SPLIT
df = pd.read_csv("train.csv")

rng = np.random.default_rng(S)
df = df.iloc[rng.permutation(len(df))]

n1 = int(0.8*len(df))
n2 = int(0.9*len(df))
Xtr, Ytr = build_dataset(df[:n1]) # TRAINING SPLIT
Xte, Yte = build_dataset(df[n1:n2]) # TESTING SPLIT
Xval, Yval = build_dataset(df[n2:]) # VALIDATION SPLIT

# STANDARDIZER: EQUAL FEATURE SCALES => ONE LEARNING RATE WORKS FOR ALL WEIGHTS, AND z STAYS NEAR 0 WHERE SIGMOID HAS SLOPE. IF LARGE |z| --> GRADIENT VANISHES 
mean, std = params_standardizer_fitting(Xtr)

Xtr = standardize_data(Xtr, mean, std)
Xte = standardize_data(Xte, mean, std)
Xval = standardize_data(Xval, mean, std)

# GNERALISED TRAINING FUNCTION. ALL THREE MODELS SHARE THE SAME INTERFACE
def train(model, X, Y, Xval, Yval, lr, batch_size=32, seed=S):
    rng = np.random.default_rng(seed)
    for epoch in range(MAX_EPOCHS):
        shuffled_indices = rng.permutation(len(X)) # SHUFFLED ROW INDICES FOR BATCHING
        splits = np.array_split(shuffled_indices, len(X)//batch_size) # NUMBER OF SPLITS
        
        for batch_indices in splits:
            xb, yb = X[batch_indices], Y[batch_indices]
            Ypred = model(xb)  # FORWARD PASS
            model.compute_gradients(xb, Ypred, yb) # BACKWARD PASS    
            model.update_weights(lr) # WEIGHT UPDATE
            
        train_loss = compute_loss(Y, model(X)) # LOSS COMPUTATION ON TRAINING
        val_loss = compute_loss(Yval, model(Xval)) # LOSS COMPUTATION ON VALIDATION : I USED THIS FOR LEARNING RATE TUNING. 
        val_f1 = calculate_f1_score(Yval, (model(Xval) >= 0.5).astype(int)) # F1 ON VALIDATION

        print(f"Epoch {epoch:2d}  train_loss {train_loss:.4f} val_loss {val_loss:.4f} val_f1 {val_f1:.4f}")
    return train_loss

# THE F1 SCORE EVALUATED ON THE TEST SPLIT
def evaluate(model, X, Y):
    return calculate_f1_score(Y, (model(X) >= 0.5).astype(int))

# MODEL 1: SINGLE OUTPUT NEURON
m1 = Model1(INPUT_FEATURE_COUNT, seed=S)
train(m1, Xtr, Ytr, Xval, Yval, lr=MODEL1_LEARNING_RATE)
write_log("logs/train_log_model_1.txt", (m1(Xte) >= 0.5).astype(int))

# MODEL 2: SINGLE HIDDEN LAYER OF 32 OUTPUT NEURONS
m2 = Model2(INPUT_FEATURE_COUNT, MODEL2_HIDDEN_NEURONS, seed=S)
train(m2, Xtr, Ytr, Xval, Yval, lr=MODEL2_LEARNING_RATE)
write_log("logs/train_log_model_2.txt", (m2(Xte) >= 0.5).astype(int))

# MODEL 3: TWO HIDDEN LAYERS OF 32 AND 16 OUTPUT NEURONS
m3 = Model3(INPUT_FEATURE_COUNT, MODEL3_HIDDEN_NEURONS_L1, MODEL3_HIDDEN_NEURONS_L2, seed=S)
train(m3, Xtr, Ytr, Xval, Yval, lr=MODEL3_LEARNING_RATE)
write_log("logs/train_log_model_3.txt", (m3(Xte) >= 0.5).astype(int))

print("All models Trained.")
print(f"Model 1 test F1: {evaluate(m1, Xte, Yte):.4f}")
print(f"Model 2 test F1: {evaluate(m2, Xte, Yte):.4f}")
print(f"Model 3 test F1: {evaluate(m3, Xte, Yte):.4f}")

# SAVE MODELS
# train.py: before the pickle dumps
os.makedirs('models', exist_ok=True)

model1_payload = {**m1.parameters(), 'mean': mean, 'std': std}
with open('models/model_1.pkl', 'wb') as f:
    pickle.dump(model1_payload, f)
    
model2_payload = {**m2.parameters(), 'mean': mean, 'std': std}
with open('models/model_2.pkl', 'wb') as f:
    pickle.dump(model2_payload, f)
    
model3_payload = {**m3.parameters(), 'mean': mean, 'std': std}
with open('models/model_3.pkl', 'wb') as f:
    pickle.dump(model3_payload, f)
    
print("All models Saved.")
print("="*60)
import torch
import toch.nn as nn
from torch.nn import functional as F

# HYPERPARAMETERS
batch_size = 32
block_size = 8
max_iters = 5000
eval_iters = 200 # EVALUATION USING 200 BATCHES INSTEAD OF 1 RANDOM BATCH
learning_rate = 3e-4
device = "cuda" if torch.cuda.is_available() else "cpu"
n_embd = 384 # EMBEDDING DIMENSION
n_head = 6 # NUMBER OF ATTENTION HEADS
n_layer = 6 # THE NUMBER OF TRANSFORMERS BLOCK 
dropout = 0.2
# ------------

torch.manual_seed(1337)
with open("tinyshakespeare.txt", "r") as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

# ENCODING AND DECODING FUNCTIONS
stoi = {s:i for i, s in enumerate(chars)}
itos = {i: s for i, s in enumerate(chars)}
encode = lambda s: [stoi[ch] for ch in s]
decode = lambda l: "".join([itos[i] for i in l])

# TRAIN AND TEST SPLITS
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data))
train_data = data[n:]
val_data = data[:n]

# DATA LOADING
def get_batch(split):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data)- block_size, (batch_size,)) # WE NEED BATCH SIZE RANDOM OFFSETS. (batch_size,) - AN 1D TENSOR CONTAINING batch_size INTEGERS
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    model.eval()
    out = {}
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters): 
            X, Y = get_batch(split)
            logits, loss = model(X,Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


"""
HEAD:

Self-attention lets each token gather information from in its context.
A single head learns one way of determining relevance.
Like, "Head-1" might learn patterns about vowels.

Our input is x which is of the shape B, T, C.

B: Number of sequences in a batch
T: Number of tokens per sequence
C: Embedding dimensions per token

Three transformations are learned:
Q: Query:   What information is this token looking for?
K: Key:     What information does this token offer for matching?
V: Value:   What does this token contribute if attended to?

Q and K are compared to calculate attention weights. 
Those weights are then applied to values which provide representation of each token.


"""
# ONE HEAD OF SELF-ATTENTON 
class Head(nn.Module): # nn.Module: A pytorch COMPONENT THAT LETS US MANAGE AND TRAIN OUR NEURAL NETWORK COMPONENTS
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False) # head_size = n_embd // n_head WE WANT n_head DIFFERENT PERSPECTIVES WITHOUT INCREASING THE REPRESENTATION SIZE.
        self.query = nn.Linear(n_embd, head_size, bias=False) 
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones((block_size, block_size)))) # REGISTER BUFFER MEANS STORE THIS IN THE MODEL BUT THIS IS NOT A LEARNABLE PARAMETER. THE MATRIX IS UPPER TRIANGULAR AND MEANS EACH TOKEN CAN ATTEND TO ITSELF AND TOKENS BEFORE IT BUT NEVER TO FUTURE TOKENS
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # INPUT: (B=batch_size, T=time-step, C=channels=n_embd)
        # OUTPUT: (B=batch_size, T=time-step, HS=head_size)
        B, T, C =  x.shape
        k = self.key(x) # (B,T,HS)
        q = self.query(x) # (B,T,HS)
        # COMPUTE ATTENTION SCORES ("AFFINITIES")
        wei = q @ k.transpose(-2,-1) # (B,T,HS) @ (B,HS,T) = (B,T,T) CALCULATE ALL PAIRWISE SCORES. HOW MUCH EACH TOKEN MATCHES WITH EVERY OTHER TOKEN
        wei = wei * (k.shape[-1]**-0.5) # FOR EACH DOT PRODUCT WE ARE SUMMING OVER HS TERMS. SO VARIANCE OF SCORES CHANGES WITH HS. TO KEEP SCORE VARIANCE STABLE, WE DIVIDE BY SQRT(HS)
        wei = wei.masked_fill(
            self.tril[:T, :T] == 0, # WHY CUT THE TRIL MATRIX TO LIMIT IT TO T? TRIL IS MADE AS BLOCK_SIZE BY BLOCK_SIZE. WE DO NOT ALWAYS HAVE T=BLOCK_SIZE, IN GENERATION WE START FROM AS LITTLE AS 1 TOKEN. 
            float('-inf') # JUST SO THAT SOFTMAX GIVES "0" PROB IN THE NEXT STEP
            ) # (B,T,T)
        wei = F.softmax(wei, dim=-1) # OR dim=2. MEANS TAKE EACH ROW AND NORMALISE IT. EACH ROW SUMS TO ONE.
        wei = self.dropout(wei) # THIS PREVENTS THE MODEL FROM RELYING TOO HEAVILY OF PARTICULAR ATTENTION CONNECTIONS
        # PERFORM WEIGHTED AGGREGATION OF THE VALUES
        v = self.value(x)
        out = wei @ v # (B,T,T) @ (B,T,HS) = (B,T,HS)
        return out

# MULTIPLE HEADS OF SELF-ATTENTION IN PARALLEL       
class MultiHeadAttention(nn.module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)]) # DOUBT: WHY NOT PLAIN PYTHON LIST? HOW DOES THIS DIFFER FROM NN.SEQUENTIAL ETC. NEED TO SPEND SOME TIME HERE. 
        self.proj = nn.Linear(head_size * num_heads, n_embd) # THE HEADS HAVE WORKED INDEPENDENTLY. THIS LEARNS HOW TO MIX INFORMATION FROM DIFFERENT HEADS.
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1) # HERE EACH H(X) IS (B,T,HS=64) SO WE CONCAT 64 FEATURES OF 6 HEADS = head_size * num_heads FEATURES
        out = self.dropout(self.proj(out))
        return out # (B,T,n_embd)
    
# THINK ABOUT FEEDFORWARD LAYER THIS WAY: ATTENTION GATHERS INFORMATION, THIS LAYERS "THINKS" ABOUT THAT INFORMATION.
class FeedForward(nn.module):
    def __init__(self, n_embd):
        super().__init()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4*n_embd), # EXPAND TO EXPLORE RICHER FEATURE COMBINATIONS
            nn.ReLU(), # A NON LINEARITY ALLOWING TO LEARN MORE COMPLEX PATTERNS THAN ONLY LINEAR LAYERS WOULD ALLOW
            nn.Linear(4*n_embd, n_embd), # SHRINK BACK TO KEEP MODEL'S REPRESENTATION SIZE CONSISTENT
            nn.Dropout(dropout) 
        )
        
    def forward(self, x):
        return self.net(x)

# TRANSFORMER BLOCK: COMBINES MULTI HEAD ATTENTION WITH FEED FORWARD PROCESSING 
class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size) # COMMUNICATION FIRST
        self.ffwd = FeedForward(n_embd) # THEN FOLLOWS COMPUTATION
        self.ln1 = nn.LayerNorm(n_embd) # THIS IS APPLIED ON THE 384 FEATURES FOR EACH TOKEN TO KEEP FEATURE VALUES STABLE
        self.ln2 = nn.LayerNorm(n_embd)
    
    def forward(self, x):
        x += self.sa(self.ln1(x))
        x += self.ffwd(self.ln2(x))
        return x
        
# THE LANGUAGE MODEL
class GPTLanguageModel(nn.module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential()
        self.apply(self.__init_weights)
        
    def __init_weights(self):
        
                
    def __init_weights(self, module):
        if isinstance()
        
    def forward():
        logits = 
        loss = 
        return logits, loss
        
    def generate():
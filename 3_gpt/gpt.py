import torch
import torch.nn as nn
from torch.nn import functional as F

# HYPERPARAMETERS
batch_size = 32
block_size = 8
max_iters = 5000
eval_interval = 500 # HOW OFTEN TO PRINT TRAIN/VAL LOSS
eval_iters = 200 # EVALUATION USING 200 BATCHES INSTEAD OF 1 RANDOM BATCH
learning_rate = 3e-4
device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
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
train_data = data[:n] # FIRST 90% FOR TRAINING
val_data = data[n:] # LAST 10% FOR VALIDATION

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
        self.register_buffer('tril', torch.tril(torch.ones((block_size, block_size)))) # REGISTER BUFFER MEANS STORE THIS IN THE MODEL BUT THIS IS NOT A LEARNABLE PARAMETER. THE MATRIX IS LOWER TRIANGULAR AND MEANS EACH TOKEN CAN ATTEND TO ITSELF AND TOKENS BEFORE IT BUT NEVER TO FUTURE TOKENS
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
class MultiHeadAttention(nn.Module):
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
class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
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
        x = x + self.sa(self.ln1(x)) # NOT x += ...: IN-PLACE ADD WOULD OVERWRITE x THAT LAYERNORM SAVED FOR BACKPROP
        x = x + self.ffwd(self.ln2(x)) # RESIDUAL: EACH SUBLAYER ADDS TO x INSTEAD OF REPLACING IT
        return x
        
# THE LANGUAGE MODEL
class GPTLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd) # WHAT EACH CHARACTER IS
        self.position_embedding_table = nn.Embedding(block_size, n_embd) # WHERE IT SITS IN THE CONTEXT
        self.blocks = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # FINAL LAYER NORM
        self.lm_head = nn.Linear(n_embd, vocab_size) # MAPS EACH TOKEN'S FEATURES TO A SCORE FOR EVERY CHARACTER
        self.apply(self._init_weights)

    def _init_weights(self, module):
        # SMALL RANDOM WEIGHTS (STD 0.02) KEEP EARLY ACTIVATIONS AND GRADIENTS STABLE
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C) BROADCAST ADDS THE SAME POSITION VECTORS TO EVERY SEQUENCE
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B*T, C), targets.view(B*T)) # CROSS ENTROPY WANTS (N,C) AND (N,)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx: (B,T) CONTEXT. APPEND ONE SAMPLED CHARACTER AT A TIME
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:] # POSITION TABLE ONLY HAS block_size ROWS, SO CROP THE CONTEXT
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] # ONLY THE LAST TIME STEP PREDICTS THE NEXT CHARACTER (B,vocab_size)
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1) # SAMPLE, NOT ARGMAX, SO TEXT IS NOT REPETITIVE (B,1)
            idx = torch.cat((idx, idx_next), dim=1) # (B,T+1)
        return idx

model = GPTLanguageModel().to(device)
print(sum(p.numel() for p in model.parameters())/1e6, "M parameters")
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for it in range(max_iters):
    if it % eval_interval == 0 or it == max_iters - 1:
        losses = estimate_loss()
        print(f"step {it}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
    xb, yb = get_batch("train")
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# GENERATE FROM A SINGLE "0" TOKEN (NEWLINE)
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(model.generate(context, max_new_tokens=500)[0].tolist()))

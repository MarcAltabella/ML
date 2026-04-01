import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 64 # how many independent sequences will we process in parallel?
block_size = 256 # what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 300
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embed = 384 #how many tokens will represent each char individually
n_head = 6 #each attention layer looks at the sequence through 6 parallel lenses simultaneusly 384//6 = 64 dim each. So instead of 384 dim at the same time you get 6 parallel 64 dim
n_layer = 6 # stack 6 full blocks of attention
dropout = 0.2 # 20% of the calculations are disabled and dropped to 0

torch.manual_seed(1337)

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text))) # remove duplicate chars, convert into a list, order
vocab_size = len(chars) # size of the chars list

#Tokenizing

# char to integer
stoi = {}
for i, ch in enumerate(chars):
    stoi[ch] = i # stoi = {'a': 0, 'b': 1, 'c': 2}

# integer to char lookup
itos = {}
for i, ch in enumerate(chars):
    itos[i] = ch #itos = {0: 'a', 1: 'b', 2: 'c'}

def encode(s):
    result = []
    for c in s:
        result.append(stoi[c]) # appends the equivalent number to the char 
    return result

def decode(l):
    result = []
    for i in l:
        result.append(itos[i])
    return ''.join(result)

# Encoding
data = torch.tensor(encode(text), dtype = torch.long)
n = int(0.9*len(data))
train_data = data[:n] #90% train
val_data = data[n:] #10% test data

# Data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    if split == 'train':
        data = train_data
    else:
        data = val_data
    
    ix = torch.randint(len(data) - block_size, (batch_size,)) # 4 random positions in the data
    
    x_list = []
    for i in ix:
        chunk = data[i : i + block_size]
        x_list.append(chunk)
    x = torch.stack(x_list).to(device)

    y_list = []
    for i in ix:
        chunk = data[i + 1 : i + block_size + 1]
        y_list.append(chunk)
    y = torch.stack(y_list).to(device)

    return x, y

@torch.no_grad() # everything that happens inside the function we wont use backward
def estimate_loss():
    out = {}
    model.eval()

    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        
        for k in range(eval_iters):
            X, Y = get_batch(split) #calls get_batch with the train split or the test split
            logits, loss = model(X, Y) # calculates the loss for every step up until eval_iters which is 300
            losses[k] = loss.item() # store the loss
        out[split] = losses.mean() # compute the mean of the loss ant return the value, do this for train and test

    model.train()
    return out


class Head(nn.Module):
    
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False) 
        self.query = nn.Linear(n_embed, head_size, bias=False) 
        self.value = nn.Linear(n_embed, head_size, bias=False) 
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape

        # Every token produces a query and a key by projecting x through a linear layer
        k = self.key(x) # (B, T, head_size) --> what am I looking for
        q = self.query(x) # (B, T, head_size) --> what do I contain

        wei = q @ k.transpose(-2, -1) * C**-0.5 # Compute attention scores (B,T,C) @ (B, C, T) --> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B,T,T)
        wei = F.softmax(wei, dim = -1) # softmax for every single row to obtain normalized values
        wei = self.dropout(wei)
        v = self.value(x) # (B, T, C)
        out = wei @ v # (B, T, T) @ (B, T, C) --> (B, T, C)
        #out = wei @ x # multiply it times x to obtain the box of words. X is the information of a token
        return out

class MultiHeadAttention(nn.Module):

    # Multiple heads of self attention in parallel

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)]) # creating multiple heads
        self.proj = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        head_outputs = [h(x) for h in self.heads] # run the heads in paralel
        out =  torch.cat(head_outputs, dim=-1) # concatenate the outputs in the channel dimension
        out = self.proj(out)
        return out

class FeedForward(nn.Module):
    
    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed), # each token is a vector of 32 numbers, each vector gets mutliplied by the weight matrix (32, 32) out[0] = x[0]*w00 + x[1]*w01 + x[2]*w02 + ... + x[31]*w031. Growing the layer of the residual pathway by 4
            nn.ReLU(), # set all negative values to 0
            nn.Linear(4 * n_embed, n_embed), # projection layer going back into the residual pathway
            nn.Dropout(dropout),
        )
    
    def forward(self, x):
        return self.net(x)

class Block(nn.Module):

    # Transformer block: communication followed by computation

    def __init__(self, n_embed, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embed // n_head # calculate head size
        self.sa = MultiHeadAttention(n_head, head_size) # communication
        self.ffwd = FeedForward(n_embed) # computation
        self.lnl = nn.LayerNorm(n_embed)
        self.lnl2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        x = x + self.sa(self.lnl(x)) # this allows to propagate the gradient through addition
        x = x + self.ffwd(self.lnl2(x))
        return x


class BigramLanguageModel(nn.Module): #pred of the next char
    
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed) # vocab_size * n_embed number of embedded dimensions vocab_size → n_embed   (e.g. 65 → 32)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        '''
        self.blocks = nn.Sequential(
            Block(n_embed, n_head=4),
            Block(n_embed, n_head=4),
            Block(n_embed, n_head=4),
            nn.LayerNorm(n_embed),
        )
        '''
        # Scaling the model from 3 blocks to n blocks
        self.blocks = nn.Sequential(*[Block(n_embed, n_head=n_head) for _ in range(n_layer)])
        self.lm_f = nn.LayerNorm(n_embed) # final layer norm
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(self, idx, targets=None): #Input batch of size (B, T)
        B, T = idx.shape # idx shape is B*T

        token_emb = self.token_embedding_table(idx) # (B,T,C) -> B = 4 sequences, T = 8 positions, C = 65 channels, each channel is now represented as a row of 65 numbers called logits
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T, C) integers from 0 to t-1 get embedded through the table to create T, C
        x = token_emb + pos_emb #(B, T, C), each batch has B * T tokens with C channels dimensions, one for each token
        x = self.blocks(x) # self attention head (B, T, C)
        logits = self.lm_head(x) # (B, T, vocab_size)

        if targets is None:
            loss = None

        else: 
            B, T, C = logits.shape
            logits = logits.view(B*T, C) # convert the B,T,C into a 2D array
            targets = targets.view(B*T) # convert the targets into a 1D array
            loss = F.cross_entropy(logits, targets) # it takes the logits, softmax converts them to probabilities and finally pushes the -log(probab) to punish low confidence and to obtain the loss value
        '''
        BEFORE
        logits[0][0] = [0.2, 0.8, 0.1, ...]  # 65 values — sequence 0, position 0
        logits[0][1] = [0.5, 0.1, 0.9, ...]  # 65 values — sequence 0, position 1
        logits[1][0] = [0.9, 0.2, 0.4, ...]  # 65 values — sequence 1, position 0
        ...
        AFTER
        row 0  = [0.2, 0.8, 0.1, ...]  # was logits[0][0]
        row 1  = [0.5, 0.1, 0.9, ...]  # was logits[0][1]
        row 8  = [0.9, 0.2, 0.4, ...]  # was logits[1][0]
        '''
        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):

            idx_cond = idx[:, -block_size:] # crop the context that we are going to feed to self
            # get the predictions
            logits, loss = self(idx_cond) # call forward()
            
            # focus only on the last time step
            logits = logits[:, -1, :] # you predict the last position because thats where the next char lives (B, C)
            
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # dim=-1 operates on the last dimension of the tensor (B, C), so in C (across the 65 scores)
            
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1) -> this makes that instead of picking the highest probability character, it picks random samples based on the probabilities, so a character with 0.6 probab gets picked more often than one with .1 but not always, in order to make the output more creative
            
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1) concatenate the new character into the end of the sequence
        return idx

model = BigramLanguageModel()
m = model.to(device)

optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)

for steps in range(max_iters):

    if steps % eval_interval == 0:
        losses = estimate_loss()
        print(f"step: {steps} | train loss: {losses['train']:.4f} | val loss: {losses['val']:.4f}")


    #sample batch of data
    xb, yb = get_batch('train')

    # eval loss
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    ''' 
    iteration 1:   [0.51, 0.48, 0.50, ...]  # random start
                        ↓ saw 'h' followed by 'e' in Shakespeare
    iteration 2:   [0.51, 0.52, 0.50, ...]  # 'e' score nudged up slightly
                        ↓ saw it again
    iteration 3:   [0.51, 0.56, 0.50, ...]  # 'e' score nudged up again
                        ↓ thousands of times later
    iteration N:   [0.1,  4.2,  0.3,  ...]  # 'e' score now much higher
    '''

context = torch.zeros((1,1), dtype=torch.long).to(device) # (1, 1) = [[0]] first char, 0
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
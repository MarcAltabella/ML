import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 32 # how many independent sequences will we process in parallel?
block_size = 8 # what is the maximum context length for predictions?
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200

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



class BigramLanguageModel(nn.Module):
    
    def __init__(self, vocab_size):
        
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size) # create a lookup table size vocab_size * vocab_size, every char gets its own embedding during training

    def forward(self, idx, targets=None): #Input batch of size (B, T)

        logits = self.token_embedding_table(idx) # (B,T,C) -> B = 4 sequences, T = 8 positions, C = 65 channels, each channel is now represented as a row of 65 numbers called logits
        
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
            # get the predictions
            logits, loss = self(idx) # call forward()
            
            # focus only on the last time step
            logits = logits[:, -1, :] # you predict the last position because thats where the next char lives (B, C)
            
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # dim=-1 operates on the last dimension of the tensor (B, C), so in C (across the 65 scores)
            
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1) -> this makes that instead of picking the highest probability character, it picks random samples based on the probabilities, so a character with 0.6 probab gets picked more often than one with .1 but not always, in order to make the output more creative
            
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1) concatenate the new character into the end of the sequence
        return idx

model = BigramLanguageModel(vocab_size)
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
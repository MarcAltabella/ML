# Creating X:
nn.Embedding(vocab_size, n_embed)  # shape: (65, 32)

A table with 65 rows (one per character) and 32 columns. Given a character index, returns its 32-dim vector:

index 0  ("a")  →  [0.2, 0.5, 0.1, ...]  32 numbers
index 1  ("b")  →  [0.8, 0.1, 0.9, ...]  32 numbers
...
index 64 ("z")  →  [0.6, 0.4, 0.8, ...]  32 numbers


nn.Embedding(block_size, n_embed)  # shape: (8, 32)

Same idea but for **positions** — a table with 8 rows (one per position in the sequence):

position 0  →  [0.3, 0.7, 0.2, ...]  32 numbers
position 1  →  [0.1, 0.4, 0.6, ...]  32 numbers
...
position 7  →  [0.9, 0.2, 0.5, ...]  32 numbers

# Creating logits:
1. 
x = self.sa_head(x)  # (B, T, 32) → (B, T, 32)
x goes into the attention head. Each token gathers information from previous tokens. Shape stays the same — but now each token's 32 numbers contain context from the whole sequence, not just its own embedding.

2. 
logits = self.lm_head(x)  # (B, T, 32) → (B, T, 65)
`lm_head` is `nn.Linear(32, 65)` — it takes each token's 32-dim contextual vector and projects it up to 65 numbers, one per character in the vocabulary:

The reestreching is done token vector (32,)  @  W (32, 65)  →  logits (65,), where weight is 

### What the 65 numbers mean

Each number is a **raw score** for how likely that character is to come next:

logits for position 4:
  "a" → 0.2
  "b" → 4.1   ← model thinks "b" is likely next
  "c" → 0.8
  ...
  "z" → 0.1

They're not probabilities yet — just raw scores. Softmax converts them to probabilities during loss calculation or generation.


### Why position embedding is needed

Attention has no built-in sense of order — without it, the model can't distinguish:

"cat sat"  vs  "sat cat"

x = token_emb + pos_emb  # (B, T, 32)

Each token final vector, what it is + where it is

# Attention
Imagine you're reading the sentence:

"The animal didn't cross the street because it was too tired"

When you hit the word "it", your brain immediately looks back at "animal" — not "street", not "cross". You're paying attention to the relevant word.
That's exactly what the attention mechanism does, but learned automatically.

The three players
Every token produces three vectors:

Query — "what am I looking for?"
Key — "what do I contain?"
Value — "what do I actually give you if you pick me?"

The process is:

Your query dots with everyone else's keys → produces a score (how relevant is each token to me?)
Scores go through softmax → become weights that sum to 1
Multiply those weights by everyone's values → weighted average of information

"it" query  ·  "animal" key  =  high score  ✓
"it" query  ·  "street"  key  =  low score   ✗

# Self attention heads

Key & Query calculation:
W_k and W_q are random at first but during training they start improving
self.key   = nn.Linear(n_embed, head_size, bias=False)  # W_k initialised randomly
self.query = nn.Linear(n_embed, head_size, bias=False)  # W_q initialised randomly
self.value = nn.Linear(n_embed, head_size, bias=False)  # W_v initialised randomly

token vector (32,)  @  W_k (32, 16)  →  key (16,)
token vector (32,)  @  W_q (32, 16)  →  query (16,)

wei = q @ k.transpose(-2, -1) * C**-0.5  # (B, T, T)

Every token scores every other token — *how relevant are you to me?* The `* C**-0.5` just stops the values from getting too large before softmax (scaling trick).

The result is a `(T, T)` matrix of raw scores:

        the  cat  sat  on
the  [  1.2  0.3  0.1  0.8 ]
cat  [  0.4  1.5  0.9  0.2 ]
sat  [  0.1  0.8  1.1  0.3 ]
on   [  0.6  0.2  0.4  1.0 ]

wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))

`tril` is a lower triangular matrix of ones:

1 0 0 0
1 1 0 0
1 1 1 0
1 1 1 1

Wherever there's a `0` (upper triangle = future tokens), the score becomes `-inf`. After softmax, `-inf` becomes `0` — those tokens get **zero attention weight**.

        the  cat  sat  on
the  [  1.2  -inf -inf -inf ]   ← "the" can only see itself
cat  [  0.4  1.5  -inf -inf ]   ← "cat" can see "the" and itself
sat  [  0.1  0.8  1.1  -inf ]   ← "sat" can't see "on" yet
on   [  0.6  0.2  0.4  1.0  ]   ← "on" sees everything

wei = F.softmax(wei, dim=-1)
Converts each row into probabilities that sum to 1. The -inf scores become exactly 0.

v = self.value(x)
out = wei @ v  # (B, T, T) @ (B, T, head_size) → (B, T, head_size)

Values are what tokens actually *share*. The attention weights tell you *how much* to take from each token's value. The result for each token is a **weighted average** of all the values it's allowed to see.

Loss is calculated by comparing the matrix with the logits with the correct next char

x = data[i : i + block_size]      # "the cat sa"
y = data[i+1 : i + block_size+1]  # "he cat sat"

So for every token position, the model must predict the next one, and the correct answer is already sitting in `y`.

input:  "t"
logits: [0.1, 0.05, 4.2, 0.3, ...]   ← 65 scores, one per possible next char
                 ↑
            "h" has highest score  ✓  (correct next char is "h")

Cross entropy:
1. Runs softmax on logits → probabilities
2. Looks up the probability assigned to the **correct** next character
3. Computes `-log(p)` — low confidence in the right answer = high loss

model assigns "h" probability 0.9  →  loss = -log(0.9) = 0.10  ✓ low loss
model assigns "h" probability 0.1  →  loss = -log(0.1) = 2.30  ✗ high loss

### The shape of information flow

token 1 → can only use token 1's value
token 2 → blends token 1 and 2's values
token 3 → blends token 1, 2 and 3's values
token 4 → blends all values

# Multi-head attention
A single head produces one (T, T) attention map, one pattern of "who attends to whom". It might learn that pronouns point to nouns. Great. But language has dozens of relationships happening simultaneously in the same sentence:

"The tired animal didn't cross the street because it was too scared"

"it" → "animal" (pronoun resolution)
"tired" → "animal" (adjective agreement)
"cross" → "animal" (subject-verb)
"scared" → "it" (predicate)
"because" → connects two clauses
One head can't capture all of that at once — its single (T, T) map can only express one weighting pattern per token.

Multiple heads each produce their own independent attention map, so the model can track all those relationships simultaneously. Then concatenation merges all that information back into one rich representation.

Essentially what they do is focus on different relationships. They diverge during training because they have different W_q, W_k and W_v

Why they diverge during training
Imagine two heads both randomly initialise. By chance, head 1's weights happen to produce slightly higher scores when pronouns attend to nouns. Backprop notices this is useful and reinforces it:
head 1 accidentally gets pronoun→noun right
       ↓
loss drops slightly
       ↓
backprop nudges head 1's W_q and W_k further in that direction
       ↓
head 1 gets better at pronoun→noun
       ↓
repeat thousands of times

Meanwhile head 2 accidentally got verb→subject right first, so it gets nudged in that direction instead.

Then when concatenating they get useful information for different patterns.
[  pronoun→noun info  |  verb→subject info  |  adj→noun info  |  proximity info  ]
      8 dims                 8 dims               8 dims              8 dims
                                    ↓
                              (B, T, 32)


# FeedForward
After multi-head attention, each token has gathered information from other tokens. FeedForward then processes **each token independently**:

(B, T, 32)  →  Linear(32, 32)  →  ReLU  →  (B, T, 32)

**Linear** — projects the 32-dim vector through a weight matrix, mixing all 32 dimensions together:

token vector (32,)  @  W (32, 32)  →  new vector (32,)


**ReLU** — sets all negative values to zero:

[-0.5, 1.2, -0.3, 0.8]  →  [0, 1.2, 0, 0.8]

This adds **non-linearity** — without it the whole network collapses into one big linear operation and loses expressive power.

Attention handles **where to look** — communication between tokens. FeedForward handles **what to do with that information** — computation within each token.

They're complementary:

attention     → tokens talk to each other
feedforward   → each token thinks about what it heard

# nn.Sequential for the transformer blocks
        self.blocks = nn.Sequential(
            Block(n_embed, n_head=4),
            Block(n_embed, n_head=4),
            Block(n_embed, n_head=4),
        )

x = Block1(x)   # attention + feedforward
x = Block2(x)   # attention + feedforward
x = Block3(x)   # attention + feedforward

Each block takes `x`, enriches it, and passes it to the next. Instead of writing that manually in `forward`, `nn.Sequential` does it automatically when you call `self.blocks(x)`.

Why stack multiple blocks:
One block = one round of "look at context, think about it". That's limited.
Block 1  →  learns simple patterns  (which chars follow which)
Block 2  →  learns combinations     (short word patterns)
Block 3  →  learns higher structure (phrase-level patterns)

What we are doing here is repeat the whole process several times by linking the blocks, not loosing information and improving the reasoning. 

nn.Sequential does this from passing the output of Block 1 as input for Block 2 and etc...

# Residual block connection
You go from inputs to the targets via addition, this is good because addition distributes gradient equally, so the gradients from the loss hop through the addition node to the imput. "Gradient superhighway"

Without residual connections, gradients have to travel through every block to reach the early layers:
loss → Block3 → Block2 → Block1 → embeddings
Each block slightly weakens the gradient as it passes through (multiplications shrink it). By the time it reaches Block1, the gradient is tiny — Block1 barely learns. This is called the vanishing gradient problem.

Instead of replacing x, each block adds its output back to the original input:

# without residual
x = block(x)

# with residual
x = x + block(x)

# LayerNorm1d
LayerNorm normalises each token's vector independently so all 100 numbers have mean=0 and variance=1.

Why it's needed
After attention and feedforward, some dimensions of x might be very large and others tiny:
token vector: [0.001, 952.3, 0.04, 0.8, 411.2, ...]
This makes training unstable — large values dominate, gradients explode or vanish. LayerNorm fixes that by rescaling every token back to a standard range.

Step 1 — raw token vector (messy, unbalanced)
[0.001, 952.3, 0.04, 0.8, 411.2, ...]

Step 2 — compute mean and variance
mean = (0.5 + 950.2 + 0.03 + 412.7 + 0.8 + 100.1) / 6
mean = 244.1

variance = avg of (x - mean)²
var = 144820.3

Step 3 — subtract mean, divide by std (xhat = (x − mean) / sqrt(var + eps))
[−0.64, 1.86, −0.64, 0.45, −0.64, −0.39]

Step 4 — scale and shift (out = gamma × xhat + beta)
gamma starts as all 1s, beta as all 0s → output = xhat at first, then learned
[−0.64, 1.86, −0.64, 0.45, −0.64, −0.39]

950.2 and 0.03 were wildly different, now they're 1.86 and −0.64
same relative difference, but rescaled to a stable range for training

Full flow:
token indices (B, T)
       ↓
embeddings (B, T, 32)     ← each token is a 32-dim vector, knows nothing about others
       ↓
Block 1 — attention + ffwd
Block 2 — attention + ffwd
Block 3 — attention + ffwd  ← now each token's 32 dims contain context from ALL other tokens
       ↓
LayerNorm (B, T, 32)      ← same shape, just stabilised
       ↓
lm_head (B, T, 65)        ← project to vocab size

# Dropout
Every forward pass dropout shuts off a number of neurons, what it does, is at the end, end up training sub networks and at eval time all the subnetworks are merged into a single network

# N heads, N layers
6 heads — each attention layer looks at the sequence through 6 parallel lenses simultaneously:
head_size = n_embed // n_head = 384 // 6 = 64 dims each
So instead of one 384-dim attention, you get 6 independent 64-dim attentions, each potentially specialising in a different relationship type (grammar, proximity, pronoun resolution, etc.), then concatenated back to 384.

6 layers — you stack 6 full blocks of (attention + feedforward):
embeddings (B, T, 384)
    ↓
Block 1 — 6-head attention + ffwd
Block 2 — 6-head attention + ffwd
Block 3 — 6-head attention + ffwd
Block 4 — 6-head attention + ffwd
Block 5 — 6-head attention + ffwd
Block 6 — 6-head attention + ffwd
    ↓
LayerNorm → lm_head → logits

If it were for example 3 layers:
embeddings (B, T, 384)
    ↓
Block 1 — 6-head attention + ffwd   ← 6 perspectives, 1st round of reasoning
    ↓
Block 2 — 6-head attention + ffwd   ← 6 perspectives, 2nd round of reasoning
    ↓
Block 3 — 6-head attention + ffwd   ← 6 perspectives, 3rd round of reasoning
    ↓
lm_head → logits

# N params used
embedding = 65*384 = 24960
position embedding = 256 * 348 = 98304

PER BLOCK (6 blocks)
MultiHeadAttention
key:    n_embed × n_embed  =  384 × 384  =  147,456
query:  n_embed × n_embed  =  384 × 384  =  147,456
value:  n_embed × n_embed  =  384 × 384  =  147,456
proj:   n_embed × n_embed  =  384 × 384  =  147,456

FeedForward
linear1:  n_embed × 4*n_embed  =  384 × 1536  =  589,824
linear2:  4*n_embed × n_embed  =  1536 × 384  =  589,824

LayerNorm ×2 (gamma + beta)
ln1:  384 + 384  =  768
ln2:  384 + 384  =  768

LayerNorm:  768
lm_head:    n_embed × vocab_size  =  384 × 65  =  24,960

Total ≈ 10.7 million parameters
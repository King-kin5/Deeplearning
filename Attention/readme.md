# Seq2Seq Encoder + Bahdanau Attention Decoder Study Note

## What is Attention?

Attention helps the decoder focus on the most relevant parts of the encoder outputs while generating each token.

Instead of relying only on the final encoder hidden state, the decoder can look back at all encoder outputs and decide which positions matter most.

---

# Tensor Shape Symbols

These symbols are commonly used when describing tensor shapes.

| Symbol | Meaning                                                |
| ------ | ------------------------------------------------------ |
| B      | Batch size (number of examples processed at once)      |
| M      | Encoder sequence length / number of encoder time steps |
| T      | Decoder sequence length                                |
| H      | Hidden size / feature dimension                        |
| V      | Vocabulary size                                        |
| 1      | Single decoding time step                              |
| t      | Current decoding step                                  |

---

---

# Important Shape Summary

| Tensor          | Shape     |
| --------------- | --------- |
| encoder_outputs | [B, M, H] |
| encoder_hidden  | [1, B, H] |
| decoder_input   | [B, 1]    |
| query           | [B, 1, H] |
| scores          | [B, 1, M] |
| alphas          | [B, 1, M] |
| context         | [B, 1, H] |
| decoder_output  | [B, 1, V] |

---





# Example Shapes

If:

```python
B = 32
M = 10
H = 256
```

Then:

```python
encoder_outputs.shape = [32, 10, 256]
query.shape = [32, 1, 256]
alphas.shape = [32, 1, 10]
```

Meaning:

* 32 sentences in the batch
* each sentence has 10 encoder positions
* each position contains a 256-dimensional hidden vector

---

# Bahdanau Attention

```python
class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.W1 = nn.Linear(hidden_size, hidden_size)
        self.W2 = nn.Linear(hidden_size, hidden_size)
        self.V = nn.Linear(hidden_size, 1)

    def forward(self, query, values, mask):
        """
        query:  [B, 1, H]
        values: [B, M, H]
        mask:   [B, M]
        """

        query_expanded = query.expand(-1, values.size(1), -1)

        scores = self.V(
            torch.tanh(
                self.W1(query_expanded) + self.W2(values)
            )
        )

        scores = scores.transpose(1, 2)

        scores = scores.masked_fill(
            mask.unsqueeze(1) == 0,
            -1e9
        )

        alphas = F.softmax(scores, dim=-1)

        context = torch.bmm(alphas, values)

        return context, alphas
```

---

# Step-by-Step Explanation

## 1. Expand Query

```python
query_expanded = query.expand(-1, values.size(1), -1)
```

Original query shape:

```python
[B, 1, H]
```

Expanded shape:

```python
[B, M, H]
```

Why?

The decoder query must be compared with every encoder output position.

---

## 2. Compute Attention Scores

```python
scores = self.V(
    torch.tanh(
        self.W1(query_expanded) + self.W2(values)
    )
)
```

This computes similarity between:

* decoder hidden state
* encoder outputs

### Internal Steps

```python
self.W1(query_expanded)
```

Transforms the decoder query.

```python
self.W2(values)
```

Transforms encoder outputs.

Then both are added together:

```python
self.W1(query_expanded) + self.W2(values)
```

Then passed through tanh:

```python
torch.tanh(...)
```

Finally converted into a single score:

```python
self.V(...)
```

Output shape:

```python
[B, M, 1]
```

Meaning:

* one attention score per encoder position

---

## 3. Transpose Scores

```python
scores = scores.transpose(1, 2)
```

Before:

```python
[B, M, 1]
```

After:

```python
[B, 1, M]
```

This prepares scores for softmax.

---

## 4. Apply Mask

```python
scores = scores.masked_fill(
    mask.unsqueeze(1) == 0,
    -1e9
)
```

Purpose:

* ignore padding tokens

Padding positions receive:

```python
-1e9
```

After softmax, these become almost zero.

---

## 5. Compute Attention Weights

```python
alphas = F.softmax(scores, dim=-1)
```

Output shape:

```python
[B, 1, M]
```

These are probabilities over encoder positions.

All attention weights sum to 1.

---

## 6. Compute Context Vector

```python
context = torch.bmm(alphas, values)
```

Batch matrix multiplication.

Shapes:

```python
alphas = [B, 1, M]
values = [B, M, H]
```

Output:

```python
context = [B, 1, H]
```

This is the weighted sum of encoder outputs.

---

# Encoder Explanation

The encoder reads the input sequence and converts it into hidden representations that the decoder can later use.

The encoder processes tokens one step at a time using a recurrent neural network (GRU).

---

## Encoder Responsibilities

The encoder:

1. receives token IDs
2. converts them into embeddings
3. processes the sequence with a GRU
4. produces hidden states

---

## Encoder Tensor Shapes

| Tensor    | Shape     |
| --------- | --------- |
| input_seq | [B, T]    |
| embedded  | [B, T, H] |
| output    | [B, T, H] |
| hidden    | [1, B, H] |

---

## Encoder Step-by-Step

### Embedding Layer

```python
self.embedding = nn.Embedding(input_size, hidden_size)
```

Purpose:

* converts token IDs into dense vectors

Example:

```text
word ID 42 → vector of length H
```

Input:

```python
[B, T]
```

Output:

```python
[B, T, H]
```

---

### GRU Layer

```python
self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
```

Purpose:

* processes the sequence over time
* updates hidden states

Input shape:

```python
[B, T, H]
```

Outputs:

```python
output = [B, T, H]
hidden = [1, B, H]
```

---

### Encoder Outputs

```python
output
```

Contains hidden states for every encoder position.

Example:

```text
Sentence:
I love deep learning

Encoder states:
h1, h2, h3, h4
```

Attention uses these hidden states.

---

### Final Hidden State

```python
hidden
```

Represents the final memory/state after the entire sequence has been processed.

The decoder often uses this to initialize its hidden state.

---

# Full Encoder

```python
class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)

        self.gru = nn.GRU(
            hidden_size,
            hidden_size,
            batch_first=True
        )

    def forward(self, input_seq, hidden):
        """
        input_seq: [B, T]
        hidden:    [1, B, H]
        """

        embedded = self.embedding(input_seq)

        output, hidden = self.gru(embedded, hidden)

        return output, hidden

    def init_hidden(self, batch_size, device):
        return torch.zeros(
            1,
            batch_size,
            self.hidden_size,
            device=device
        )
```

---

# Decoder Explanation

The decoder generates output tokens one step at a time.

At every decoding step, the decoder:

1. receives the previous token
2. updates its hidden state
3. computes attention over encoder outputs
4. creates a context vector
5. predicts the next token

---

## Decoder Tensor Shapes

| Tensor         | Shape     |
| -------------- | --------- |
| decoder_input  | [B, 1]    |
| embedded       | [B, 1, H] |
| query          | [B, 1, H] |
| context        | [B, 1, H] |
| decoder_output | [B, 1, V] |

---

## Decoder Step-by-Step

### Input Token

At each step, the decoder receives one token.

Shape:

```python
[B, 1]
```

Initially this is usually the SOS token.

---

### Embedding

```python
embedded = self.embedding(input_token)
```

Converts token IDs into dense vectors.

Output:

```python
[B, 1, H]
```

---

### Decoder GRU

```python
gru_output, hidden = self.gru(embedded, hidden)
```

Updates the decoder hidden state.

Outputs:

```python
gru_output = [B, 1, H]
hidden = [1, B, H]
```

---

### Query

```python
query = gru_output
```

The decoder hidden state becomes the attention query.

This query asks:

```text
Which encoder positions are most important right now?
```

---

### Attention

```python
context, attn = self.attention(
    query,
    encoder_outputs,
    input_mask
)
```

Attention compares the decoder query against all encoder outputs.

Outputs:

```python
context = [B, 1, H]
attn = [B, 1, M]
```

---

### Combine Query + Context

```python
combined = torch.cat((gru_output, context), dim=2)
```

Shapes:

```python
gru_output = [B, 1, H]
context = [B, 1, H]
```

After concatenation:

```python
[B, 1, 2H]
```

---

### Reduce Dimensions

```python
combined = torch.tanh(self.concat(combined))
```

Transforms:

```python
[B, 1, 2H] → [B, 1, H]
```

---

### Vocabulary Prediction

```python
output = self.out(combined)
```

Final prediction scores over the vocabulary.

Output:

```python
[B, 1, V]
```

Where:

* V = vocabulary size

---

### Teacher Forcing

If training targets are provided:

```python
decoder_input = target_tensor[:, t].unsqueeze(1)
```

The decoder uses the real target token as the next input.

Otherwise:

```python
_, topi = decoder_output.topk(1, dim=-1)
```

The decoder uses its own prediction.

---

# Full Bahdanau Attention Decoder

```python
class AttentionDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super().__init__()

        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)

        self.gru = nn.GRU(
            hidden_size,
            hidden_size,
            batch_first=True
        )

        self.attention = BahdanauAttention(hidden_size)

        self.concat = nn.Linear(hidden_size * 2, hidden_size)

        self.out = nn.Linear(hidden_size, output_size)

        self.bridge = nn.Linear(hidden_size, hidden_size)

    def forward(
        self,
        encoder_outputs,
        encoder_hidden,
        input_mask,
        target_tensor=None,
        SOS_token=0,
        max_len=10,
        device="cpu"
    ):

        batch_size = encoder_outputs.size(0)

        decoder_input = torch.full(
            (batch_size, 1),
            SOS_token,
            dtype=torch.long,
            device=device
        )

        decoder_hidden = torch.tanh(
            self.bridge(encoder_hidden)
        )

        decoder_outputs = []
        attentions = []

        steps = (
            target_tensor.size(1)
            if target_tensor is not None
            else max_len
        )

        for t in range(steps):

            decoder_output, decoder_hidden, attn = self.forward_step(
                decoder_input,
                decoder_hidden,
                encoder_outputs,
                input_mask
            )

            decoder_outputs.append(decoder_output)
            attentions.append(attn)

            if target_tensor is not None:
                decoder_input = target_tensor[:, t].unsqueeze(1)
            else:
                _, topi = decoder_output.topk(1, dim=-1)
                decoder_input = topi.squeeze(-1).detach()

        decoder_outputs = torch.cat(decoder_outputs, dim=1)

        decoder_outputs = F.log_softmax(
            decoder_outputs,
            dim=-1
        )

        attentions = torch.cat(attentions, dim=1)

        return decoder_outputs, decoder_hidden, attentions

    def forward_step(
        self,
        input_token,
        hidden,
        encoder_outputs,
        input_mask
    ):

        embedded = self.embedding(input_token)

        embedded = F.relu(embedded)

        gru_output, hidden = self.gru(embedded, hidden)

        query = gru_output

        context, attn = self.attention(
            query,
            encoder_outputs,
            input_mask
        )

        combined = torch.cat(
            (gru_output, context),
            dim=2
        )

        combined = torch.tanh(
            self.concat(combined)
        )

        output = self.out(combined)

        return output, hidden, attn
```

---

# Decoder Flow

At each decoder step:

```text
Input Token
     ↓
Embedding
     ↓
GRU
     ↓
Query
     ↓
Attention over Encoder Outputs
     ↓
Context Vector
     ↓
Concatenate(Query + Context)
     ↓
Linear Layer
     ↓
Vocabulary Prediction
```



# Core Idea of Bahdanau Attention

The decoder:

1. generates a query from its current hidden state
2. compares that query with all encoder outputs
3. produces attention weights
4. creates a weighted context vector
5. uses the context vector to predict the next token

This allows the decoder to dynamically focus on different parts of the input sequence during generation.

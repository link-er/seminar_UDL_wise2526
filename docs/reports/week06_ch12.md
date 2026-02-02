# Report – Week 06: Transformers

**Presenter:** Fadi Dalbah  
**Date:** 01.12.2025  

---

## Summary

### 1. Motivation

- **Text as input**
  - Texts can be **very long**.
  - **Variable length**: each text has a different number of inputs.
  - **Ambiguity**: the meaning of a word depends strongly on its context.
- **Goal:** A mechanism that can flexibly relate **any input to any other** in the sequence and share parameters between similar inputs.

### 2. Self-Attention

- Think of self-attention as **routing information**:
  - For each position, the model decides **how much to take** from each other position.
- $n^{th}$ output at position is the **weighted sum** of $N$ value vectors:
  - Different outputs can use **different weight distributions**, i.e. focus on different parts of the sequence.

#### 3.1 Values, queries and keys

- Each input token embedding $x_m$ is linearly mapped to a **value**:
  $$
  v_m = \beta_v + \Omega_v x_m
  $$
- The same input is also mapped to a **query** and a **key**:
  $$
  q_n = \beta_q + \Omega_q x_n
  $$
  $$
  k_m = \beta_k + \Omega_k x_m
  $$

#### 3.2 Attention scores and weights

- **Similarity score** between token $m$ and position $n$ via dot product:
  $$
  e_{mn} = k_m^T q_n
  $$
- Convert scores into **attention weights** with a softmax over $m$:
  $$
  a[x_n, x_m] = \text{softmax}_m(e_{mn})
  $$
  $$
  = \frac{\exp(e_{mn})}{\sum_{m'=1}^N \exp(e_{m'n})}
  $$


#### 3.3 Output of self-attention

- **Per position** $n$:
  $$
  \text{sa}_n([x_1,\dots,x_N]) = \sum_{m=1}^N a[x_m, x_n] \, v_m
  $$
- **Matrix notation**:
  - Compute:
    $$
    Q = \Beta_q1^T + \Omega_q X
    $$
    $$
    K = \Beta_q1^T + \Omega_k X
    $$
    $$
    V = \Beta_q1^T + \Omega_v X
    $$
  - Basic self-attention:
    $$
    \text{Sa}(X) = V \cdot \text{Softmax}(K^T Q)
    $$

### 4. Important Extensions of Self-Attention

#### 4.1 Positional encoding

- Self-attention alone is **order-invariant**.
- Add a **positional encoding** to each token:
- $p_n$ can be:
  - **Absolute** (depends on position index) 
    - **Chosen** or **learned**,
  - **Relative** (depends on distances between positions),

#### 4.2 Scaled dot-product attention

- In high dimensions, dot products can become large.
- This can cause:
  - Largest value dominates softmax,
  - Small gradient changes → harder training.
- Solution: **scale** the scores by query dimension $\sqrt{D_q}$:
  $$
  \text{SA}(X) = V \cdot \text{Softmax}\!\left(\frac{K^T Q}{\sqrt{D_q}}\right)
  $$

#### 4.3 Multi-head attention

- Run **multiple self-attentions in parallel**:
  - Each head $h$ has its own computation.
- Compute:
    $$
    Q_h = \Beta_{vh}1^T + \Omega_{kh} X
    $$
    $$
    K_h = \Beta_{qh}1^T + \Omega_{kh} X
    $$
    $$
    V_h = \Beta_{kh}1^T + \Omega_{kh} X
    $$
- Concatenate and linearly transform:
  $$
  \text{Sa}_h(X) = V_h \cdot \text{Softmax}\!\left(\frac{K^T_h Q_h}{\sqrt{D_q}}\right)
  $$
- Can make network more robust to bad initializations

### 5. Transformers for NLP

#### 5.1 Tokenization and embeddings

- **Tokenization**:
  - Split text into subword **tokens** from a fixed vocabulary.
- **Embedding**:
  - Each token index is mapped to a dense vector (word embedding).

#### 5.2 Encoder-only model: BERT (example configuration)

- Input: full sentence → **bidirectional** self-attention.
- Typical configuration presented:
  - Vocabulary: ~30 000 tokens.
  - Embedding dimension: 1 024.
  - 24 transformer layers, each with 16 heads.
  - Query/key/value projections: $64 \times 1\,024$ per head.
  - Feed-forward hidden dimension: 4 096.
  - ≈ 340M parameters.
- For pretaining
  - Inputs are converted to embeddings
  - Passed through transformer layers
  - Small fraction of tokens replaced with `<mask>` token
  - Goal is to predict the right token
- For classification
  - `<cls>` token placed at the start of string
  - Token mapped to a number

### 6. Decoder-only Models and Masked Attention (GPT-3 style)

#### 6.1 Autoregressive objective

- Model predicts the **next token**:
  $$
  Pr(t_1, t_2, \dots, t_N)
  $$
- This defines a probability for the whole sequence:
  $$
  Pr(t_1, t_2, \dots, t_N) = Pr(t_1)\prod_{n=2}^N Pr(t_n \mid t_1, \dots, t_{n-1})
  $$

#### 6.2 Masked self-attention

- During training, token at position $n$ must **not see future tokens**.
- Implemented by adding a **mask matrix**
- Entries with $-\infty$ become **0** after softmax.

#### 6.3 Generating text

- Start with a special `<start>` token.
- Repeatedly:
  1. Compute distribution over next token via masked self-attention and output layer.
  2. Sample or choose the most probable token.
  3. Append it to the sequence and feed back into the decoder.
- Stop when `<end>` token is produced.

- Large decoder models support **few-shot learning**:
  - A few examples in the prompt are enough to perform a new task without changing the model parameters.


### 7. Encoder–Decoder Transformers and Cross-Attention

- **Encoder**:
  - Processes source sentence, producing context representations.
- **Decoder**:
  - Uses:
    - **Masked self-attention** over previous target tokens.
    - **Cross-attention** over encoder states:
      - Queries $Q$ from decoder embeddings,
      - Keys $K$ and values $V$ from encoder outputs $E$.
  - Same scaled dot-product formula, just with **different sources** for $Q$ vs. $K, V$.

- Main application: **machine translation**.


### 8. Variants

- Long-sequence transformers (efficient attention for long texts).
- Image transformer and ImageGPT (apply attention to image patches).
- Vision Transformer (ViT) and multi-scale ViT (hierarchical image representations).
- Many other adaptations to different data types and tasks.

---

## Discussion Notes

- Can the first dot product computations in inference be cached instead of recomputing them every time?
- Comparison of Vision Transformers and CNNs for image processing.
- Questions about nonlinearity in attention:
  - Why there is no activation in the attention score computation itself.
  - How scaled dot-product self-attention introduces nonlinearity.
  - The softmax acts as an activation function.

---

## References

- Understanding Deep Learning (Simon J.D. Prince, 2023)

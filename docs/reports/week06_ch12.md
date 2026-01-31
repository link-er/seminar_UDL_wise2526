## Discussion Notes

### Sampling Methods
When generating text from a Large Language Model we always try to predict the most likely token given the preceding 
text sequence. However, just outputting the token with the highest probability is not always the best option. For 
example, in a scenario, where the cumulative probability for a group of words that points in a similar direction is 
bigger than the cumulative probability for a group of words that points in a different direction, but the word with 
the biggest associated probability is from the second group and thus would be sampled and lead the generation in a
wrong direction. This problem is also already known in other fields, like in elections, as 
[Spoiler effect](https://en.wikipedia.org/wiki/Spoiler_effect).

A common solution to tackle this problem is **Top-k sampling**. Here, instead of just always outputting the token with 
the highest probability, we sample from the k most likely tokens according to their probability. Another similar 
approach would be **Top-p sampling**, where $p$ is a probability, and we sample from the tokens with the biggest 
probability that cumulate to least $p$ of the total probability from all options. With for example 
**Beam search** there are also different approaches that keep track of sequences with the highest probabilities, and 
thus try to predict the optimal sequence of subsequent tokens.

### Attention for long sequences
In attention, each token interacts with every other token from the sequence. This leads to a quadratic complexity of the 
attention mechanism, and thus the attention computation for very long sequences takes up a lot of resources. However, 
there are some methods developed to tackle this problem. Most approaches sparsify the attention interaction matrix for 
example through a convolutional structure (Fig. 1c-f). The tradeoff here, however, is that tokens can only interact 
with some of the other tokens through the course of several subsequent layers. This problem can be partially tackled by 
introducing some tokens that attend to all other tokens (Fig. 1g).
![Fig.1: Types of sparse attention](../images/sparse-attention.png)

### Why are positional encodings added to the tokens and not concatenated with them?
Adding positional encodings keeps the dimensionality of the model smaller and makes it easier to alter already existing 
models, since the dimensionality does not have to change, while the model can still reasonably distinguish between 
positional and token information.

### How big are modern Large Language Models (LLMs)?
Modern LLMs can have in the hundreds of billion parameters. Some of them even exceed the mark of a trillion parameters. For 
example [Metas Llama 4 Maverick](https://ai.meta.com/blog/llama-4-multimodal-intelligence/) has a total of 400 Billion 
parameters. In order to run this model with 16-bit floating point precision, you would need around 800 GB of GPU RAM.

### Challenges of transformers in computer vision tasks
Images have a lot of pixels, which poses a problem, because the attention matrix grows quadratically with the number of 
inputs. Also, convolutional networks are particularly well suited for the two-dimensional structure of images. However, 
because of the massive number of training datapoints and the increase in compute resources transformer models have now 
eclipsed the performance of convolutional networks in many computer vision tasks.

---
## References
- [Understand Deep Learning](https://udlbook.github.io/udlbook/)
- [Spoiler effect](https://en.wikipedia.org/wiki/Spoiler_effect)
- [Metas Llama 4 Maverick](https://ai.meta.com/blog/llama-4-multimodal-intelligence/)
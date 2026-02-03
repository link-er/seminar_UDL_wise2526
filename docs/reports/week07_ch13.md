# Report - Week 7, Chapter 13: Graph Neural Networks

**Presentation Date:** 08.12.2025\
**Presenter:** Nhat Vu Minh\
**Discussion Lead:** Aisaiah Pellecer\
**Report by:** Aisaiah Pellecer

## Graph Representation

A graph consists of a set of nodes connected by edges. Both nodes and edges may carry information in the form of node embeddings and edge embeddings. 

Graphs are a natural way to represent structured data where relationships between entities are important, such as social networks, citation graphs, and molecular structures.

In some cases, edges themselves can be treated as nodes by constructing an edge graph, which allows graph neural networks to update edge embeddings using the same message-passing mechanisms applied to nodes.

## Graph Neural Networks (GNNs)

A graph neural network is a model that takes the node embeddings *X* and the adjacency matrix *A* as inputs and processes them through a sequence of *K* layers. At each layer, node embeddings are updated, producing intermediate hidden representations $H_{k}$, until final output embeddings $H_{K}$ are obtained.

Initially, each node embedding contains information only about the node itself. As embeddings pass through successive layers, information from neighboring nodes is incorporated through message passing. By the final layer, each node embedding captures both the node’s own features and its context within the graph.

This process is analogous to word embeddings in transformer models: while input embeddings represent individual words in isolation, the output embeddings represent word meanings conditioned on their surrounding context.

Since node ordering in graphs is arbitrary, GNN layers must be permutation equivariant with respect to node indices. Parameter sharing across nodes enables generalization across graphs of different sizes and structures.

## Tasks in Graph Neural Networks

Graph neural networks can be applied to three main types of tasks:

- **Edge prediction tasks:**  
  The model predicts whether an edge should exist between two nodes. This is commonly used for link prediction and recommendation problems.

- **Node-level tasks:**  
  The model assigns a label (classification) or continuous values (regression) to each node. Predictions depend on both node embeddings and the graph structure, allowing nodes to be interpreted in context.

- **Graph-level tasks:**  
  The model assigns a label or predicts one or more values for the entire graph, exploiting global structural information and node embeddings.

Node- and edge-level tasks require permutation-equivariant outputs, while graph-level tasks require permutation-invariant functions.


## Spatial-Based Graph Convolutional Networks (GCNs)

Spatial-based graph convolutional networks update node representations by aggregating information from neighboring nodes in the original graph. They are referred to as convolutional because the same local aggregation rule is applied at every node.

These models induce a relational inductive bias, favoring information from nearby nodes, and are considered spatial-based because they operate directly on the given graph structure rather than transforming the graph into another domain.


## Inductive vs. Transductive Models

Graph-level tasks occur exclusively in the inductive setting, where the model is trained on a set of graphs and evaluated on unseen graphs.

Node-level and edge prediction tasks can occur in both settings:

- In the transductive setting, learning takes place on a single large graph with partial labels. The loss function is computed only where ground truth is known, while predictions for unlabeled nodes or edges are obtained by running a forward pass and reading out the corresponding outputs. Unlabeled nodes still influence learning through message passing.

- In the inductive setting, the model is trained on multiple graphs or subgraphs and can generalize to unseen nodes or graphs. Partitioning large graphs into subgraphs can effectively convert a transductive problem into an inductive one.

---

## Layers for Graph Convolutional Neural Networks (GCNNs)

A typical GCNN layer consists of:
1. Aggregation of neighbor information (mean, sum, or max)
2. Combination with the node’s current embedding
3. Non-linear transformation using shared parameters

Stacking multiple layers increases the receptive field (region of the graph that contributes to a given node), allowing nodes to incorporate information from multi-hop neighborhoods of batch nodes-- think of this as neighborhood sampling or graph partitioning. 

---

## Discussion Notes

### Key Questions and Responses

**Why are graphs in a batch treated as disjoint components of a single large graph?**  
This allows efficient batching and parallel computation. Since message passing only occurs along edges, disconnected graphs do not exchange information. However, unbalanced graphs may dominate gradient updates and affect training stability.

**How is unlabeled data used in transductive learning?**  
Unlabeled nodes participate in message passing and influence node embeddings, acting as a form of semi-supervised learning. Class imbalance within a single large graph can introduce bias.

**What is the receptive field in GNNs?**  
The receptive field corresponds to the *k*-hop neighborhood of a node, where *k* is the number of GNN layers.

**Do unbalanced partitions affect training?**  
Yes. Large or dense subgraphs can dominate gradients. Further partitioning improves efficiency but may increase bias by removing long-range dependencies.

**What is diagonal enhancement?**  
Diagonal enhancement adds self-loops to preserve node identity and stabilize message passing.

**Why use mean aggregation?**  
Mean aggregation normalizes by neighborhood size, leading to more stable training and better inductive generalization across graphs with varying node degrees.

## References

- Deep Learning using Rectified Linear Units (ReLU) [[Link]](https://arxiv.org/pdf/1803.08375)

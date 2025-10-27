# Report – Week 01: Shallow & Deep Neural Networks



**Presenter:** [Name]  

**Date:** 27.10.2025  



## Summary

(1–2 page summary of key concepts, equations, and methods)


## Discussion Notes

- Why is ReLU the most used activation function in practice ?

  **ReLU** is the most widely used activation function in practice because it offers
  a simple yet powerful balance between computational efficiency and effective gradient propagation.
  
  Other activation functions like **sigmoid** or **tanh**, for example, squash inputs into narrow ranges
  which makes their gradients very small for large input magnitudes ==> **Vanishing gradients** problem
  [[Link]](https://www.geeksforgeeks.org/deep-learning/vanishing-and-exploding-gradients-problems-in-deep-learning/)
  
  **ReLU** on the other hand allows gradients to flow efficiently through the different layers of the network.
  Its derivative of the output with respect to the input is always a constant **1**  for positive inputs.
  This helps to avoid the saturation problem (derivative becomes close to zero) for large input data that
  the derivatives of the **sigmoid** activation function have.
  
  Another advantage of using **ReLU** is that it is extremely simple to compute (we just compare with zero and clip negative inputs)
  and has no exponentials or divisions like **sigmoid** or **tanh**.

  

- What does folding the input space actually mean ? (TODO)

- Why using Deep Neural Networks is mostly better than using Shallow ones ? (TODO)

- **Bias-Variance tradeoff** : (TODO)



## References

- Deep Learning using Rectified Linear Units (ReLU) [[Link]](https://arxiv.org/pdf/1803.08375)

# Report – Week 04: Performance & Regularization



**Presenters:** Ben Halima Ibrahim, Zoghlami Fadi 

**Date:** 10.11.2025 



## Summary

### Chapter 8: Measuring performance

**<u>I - Training a simple model:</u>**

A neural network with **sufficient capacity** will mostly perform well on the training dataset. However, this does not mean that it will generalize well on the testing dataset (which is normally new and unseen data for the model). This causes a big problem especially for real-world scenarios, where the model's performance has to be as good as possible. <br>
=> Our goal is to train a model that **generalizes well on new data**

The test errors have three distinct causes:

- the inherent uncertainty in the task
- the amount of training data
- the choice of model

In the first section of this chapter, a simple model is trained on the [**MNIST-1D**](https://arxiv.org/pdf/2011.14439) dataset, which is 1D analogue of the [**MNIST**](https://en.wikipedia.org/wiki/MNIST_database) dataset: each data example is created by **randomly** transforming one of the templates and adding noise.

![MNIST-1D dataset](../images/MNIST-1D-Dataset.jpg)

Our simple model/neural network consists of **D_i = 40** inputs and **D_o = 10** outputs representing the number of classes the dataset has (*numbers form 0 to 9*). The neural network has **2** hidden layers each with **D = 100** hidden units. **Multiclass cross-entropy** is used as a loss function with the **Softmax** function to produce class probabilities.<br>
The model is then trained for **6000 steps (150 epochs)** using **SGD** (Stochastic Gradient Descent) as a learning algorithm with a learning rate of **0.1** and a batch-size of **100**. After the training process, we tested our trained model on **1000** extra examples from the dataset.

 ![Train-Test-Error-Loss](../images/PerfMNIST1DResults.svg)

In figure (a), we can see that the training error decreases as the training proceeds (the training data is classified perfectly after around **4000
training steps**). The testing error, however, decreases as well but to about **40%** and does not drop below it.<br>
In figure (b), the training loss also decreases continuously towards zero as the training proceeds. The testing loss, on the other hand, decreases at first but suddenly starts going up after around **1500 training steps** reaching higher values than before.<br>
=> Our model is making, in this case, the same mistakes but with increasing confidence and this will decrease the probability of correct answers, and therefore increase the negative log-likelihood<br>
=> Our model has then **memorized** the training data but **does not generalize well** on the testing data

**<u>II - Sources of error:</u>**

When a neural network fails to generalize well, there are mainly three sources of error:

- **Noise:** the data generation process itself includes the addition of noise to the input data. Therefore, there are **multiple possible valid** outputs for each input (figure (a) below). This may be caused due to a **stochastic element** in the data generation process (mislabeled data as an example). In some rare cases, the noise can be **absent**: for example, a network might approximate a function that is deterministic but requires significant computation to evaluate.<br>
=> However, noise is <u>usually</u> a fundamental limitaion on the test performance
- **Bias:** this happens when the model is **not flexible enough** to fit the data perfectly. In figure (b) below for example, the three-region model (*cyan line*) cannot exactly fit the true function (*black line*), even with the best possible parameters (*gray regions represent signed error*). 
- **Variance:** this occurs when there are **limited** training examples, and therefore there is no way to distinguish noise in the underlying data from systematic changes in the underlying function.This means that, for different training datasets, the result will be slightly different each time (figure (c) below). In practice, however, there can be an **additional variance** due to the stochastic learning algorithm, which does not necesseraliy converge to the same solution each time.

 ![Noise-Bias-Varinace](../images/PerfNoiseBiasVariance.svg)

- **Mathematical formulation of test error:** (TODO !!)

 ![Noise-Bias-Variance-Equation](../images/Noise-Bias-Variance-Equation.png)

**<u>III - Reducing error:</u>**

The **Noise** component is **insurmountable**, which means there is nothing we can do to avoid it. It represents a <u>fundamental limit</u> on expected model performance. **However**, we can reduce the Variance and Bias terms.

- **Reducing Variance:** variance results from limited noisy training data. This actually means that we can reduce it by **increasing the quantity** of our training data. This approach averages out the inherent noise and ensures that the input space is well sampled.<br>
The figure below shows the effect of training with three different samples (*6, 10 and 100 samples*). The best-fitting model for each dataset is then plotted: as we can see, with only **6 samples**, the fitted function is <u>different</u> each time and the variance term is therefore significant. When we **increase** the number of samples, the fitted models become very <u>similar</u> and the variance term reduces as a result.

![Reducing-Variance](../images/PerfVariance.svg)

=> In general, adding more training data <u>almost always</u> improves test performance.

- **Reducing Bias:** in order to reduce the bias term, we can **increase the capacity** of our model/neural network (number of hidden units and/or layers) which makes it **more flexible** and able to describe the true underlying function.<br>
The figure below shows the effect of increasing the number of linear regions (*3, 5 and 10 regions*): by increasing the number of linear
regions, the model becomes flexible enough to fit the true function closely. As a result, the bias term decreases (*gray region in a-c*). **Unfortuantely**, this causes the variance term to go up (*gray region d-f*).<br>
=> Increasing the model capacity <u>does not necesseraliy</u> reduce the test error ==> **Bias-Variance trade-off**

![Reducing-Bias](../images/PerfBias.svg)

- **Bias-Variance trade-off:** when a model is too simple (*low capacity*), it **ignores** useful information, and the error is composed mostly of that from bias ==> **Underfitting**<br>
When a model is too complex, it **memorizes** non-general patterns, and the error is composed mostly of that from variance ==> **Overfitting**<br>
In both cases the model does not generalize well on new unseen data. The ideal model aims to <u>minimize</u> both bias and variance. It lays in the sweet spot (not too simple, nor too complex) in between => Achieving such a balance will yield the **minimum error**

![Bias-Variance-trade-off](../images/PerfBiasVarianceTradeoff.svg)

|  | Bias | Variance | Result | 
|--|------|----------|--------|
| **Underfitting** | High | Low | Poor training and test performances |
| **Optimal** | Moderate | Moderate | Best generalization |
| **Overfitting** | Low | High | Poor test performance |

- **Double-Descent:** (TODO !!)

**<u>IV - Choosing hyperparameters:</u>**

Hyperparameters consist of not only the number of hidden layers and the number of hidden units per layer, but also of the learning rate, the choice of the learning algorithm itself, the batch size and much more. The process of finding the best hyperparameters is called **hyperparameter search** or **neural architecture search** (when focusing on the network structure).

Hyperparameters are typically chosen <u>empirically</u>: we train many models with different hyperparameters on the **same training dataset**, then measure their performance and retain the best model. However, measuring the performance **does not** happen on the test set, as this may produce good results on only that specific set but the model does not generalize well on other new unseen data. Instead, we use a third dataset called the **validation set**. That means, for each choice of hyperparameters we train our model on the training set and evaluate its performance on the validation set and finally test the model using the testing set.

### Chapter 9: Regularization (TODO !!)

## Discussion Notes

### Measuring performance

**How do we determine if we have enough training samples?**

- There is no fixed limit or specific number of samples required, it depends on the task and the dimensionality of the data.
- As the number of dimensions increases, the volume of the input space grows exponentially so the amount of data required also increases exponentially.

**What is the Generalization Gap?**

- It is the difference between the model's performance on the training data and its performance on new test data.

**Why does Test Loss increase even while accuracy stabilizes or improves?**

- Overconfidence, doesn't allow us to know if it's making a mistake because it's so confident.
- Softmax Effect, activations are pushed to extreme values to make the probability of the correct class higher.

**What is the Bias-Variance Trade-off?**
Balance between model complexity and error:

- High bias, low variance - underfitting.
- Low bias, high variance - overfitting.

---

- What exactly is bias and variance?
    - Bias: fitting is limited by the expressiveness of the model => underfitting
    - Variance: prediction strongly depends on specific training set instead of averaging => overfitting

- Low training error doesn't guarantee low test error, but it is the hope: generalization

- Training increases confidence even in wrong predictions => cross-entropy can grow even while accuracy stays constant (sign of overfitting)

- Generalization is measured by test error or gap between training and test error

- Should you stop training when the testing error stays the same?
    - Potentially reduces overfitting

- When do you have enough training/test data?
    - More data generally helps, also depends on problem
    - Architecture of neural network should be able to "accommodate" the data
    - Cost factor must be considered

- High-dimensional problems require exponentially many samples, underlying lower-dimensional manifolds could help

---

### Regularization

**What is Explicit Regularization?**

- Explicit regularization involves adding a term to the loss function that penalizes certain parameter values to discourage overfitting.
- Maximizing the product of Likelihood and Prior is equivalent to minimizing the sum of Loss and Regularization.

**How does L2 Regularization work?**

- It adds a penalty proportional to the sum of the squared weights.
- This encourages weights to be small.
- It reduces variance but increases bias. In over-parameterized models.

---

- Gaussian prior: belief that small parameters are more probable; applies to hypothesis space not directly to data

- How does regularization come into the bias-variance-tradeoff?
    - Increases bias while reducing variance and overfitting risk (depends on type)

- Data augmentation increases complexity despite producing samples that lie in the same manifold

- Why use L2-regularization, especially in comparison to L1-regularization?
    - L1 promotes zero weights while L2 penalizes large weights smoothly

## References

- IBM - What is overfitting ? [[Link]](https://www.ibm.com/think/topics/overfitting)
- Three Sources of Model Error: Bias, Variance, and Noise [[Link]](https://python.plainenglish.io/three-sources-of-model-error-bias-variance-and-noise-9d740c0ba5d6)
- A Modern Take on the Bias-Variance Tradeoff in Neural Networks [[Link]](https://arxiv.org/pdf/1810.08591)
- The Bias-Variance trade-off [[Link]](https://mlu-explain.github.io/bias-variance/)
- The Importance of Data Splitting [[Link]](https://mlu-explain.github.io/train-test-validation/)

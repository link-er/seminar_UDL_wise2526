# Report – Week 04: Performance & Regularization



**Presenters:** Ben Halima Ibrahim, Zoghlami Fadi 

**Date:** 10.11.2025 



## Summary

### Chapter 8: Measuring performance

**<u>I - Training a simple model:</u>**

A neural network with **sufficient capacity** will mostly perform well on the training dataset. However, this does not mean that it will generalize well on the testing dataset (which is normally new and unseen data for the model). This causes a big problem especially for real-world scenarios, where the model'performance has to be as good as possible. <br>
=> Our goal is to train a model that **generalizes well on new data**.

The test errors have three distinct causes:
- the inherent uncertainty in the task
- the amount of training data
- the choice of model

In the first section of this chapter, a simple model is trained on the [**MNIST-1D**](https://arxiv.org/pdf/2011.14439) dataset, which is 1D analogue of the **MNIST** dataset: each data example is created by randomly transforming one of the templates and adding noise.

![MNIST-1D dataset](../images/MNIST-1D-Dataset.jpg)

Our simple model/neural network consists of **D_i = 40** inputs and **D_o = 10** outputs representing the number of classes the dataset has (numbers form 0 to 9). The neural network has **2** hidden layers each with **D = 100** hidden units. **Multiclass cross-entropy** is used as a loss function with the **Softmax** function to produce class probabilities.<br>
The model is then trained for **6000 steps (150 epochs)** using **SGD** (Stochastic Gradient Descent) as a learning algorithm with a learning rate of **0.1** and a batch-size of **100**. After the training process, we tested our trained model on **1000** extra examples from the dataset.

 ![Train-Test-Error-Loss](../images/PerfMNIST1DResults.svg)

In figure (a), we can see that the training error decreases as the training proceeds (the training data is classified *perfectly* after around **4000
training steps**). The testing error, however, decreases as well but to about **40%** and does not drop below it.<br>
In figure (b), the training loss also decreases continuously towards zero as the training proceeds. The testing loss, on the other hand, decreases at first but suddenly starts going up after around **1500 training steps** reaching higher values than before.<br>
=> Our model is making, in this case, the same mistakes but with increasing confidence and this will decrease the probability of correct answers, and
therefore increase the negative log-likelihood<br>
=> Our model has then **memorized** the training data but **does not generalize well** on the testing data.

**<u>II - Sources of error:</u>**

When a neural network fails to generalize well, there are mainly three sources of error:

- **Noise:** the data generation process itself includes the addition of noise to the input data. Therefore, there are **multiple possible valid** outputs for each input (figure (a) below). This may be caused due to a **stochastic element** in the data generation process (mislabeled data as an example). In some rare cases, the noise can be **absent**: for example, a network might approximate a function that is deterministic but requires significant computation to evaluate.<br>
=> However, noise is <u>usually</u> a fundamental limitaion on the test performance.
- **Bias:** this happens when the model is **not flexible enough** to fit the data perfectly. In figure (b) below for example, the three-region model (*cyan line*) cannot exactly fit the true function (*black line*), even with the best possible parameters (*gray regions represent signed error*). 
- **Variance:** this occurs when there are **limited** training examples, and therefore there is no way to distinguish noise in the underlying data from systematic changes in the underlying function.This means that, for different training datasets, the result will be slightly different each time (figure (c) below). In practice, however, there can be an **additional variance** due to the stochastic learning algorithm, which does not necesseraliy converge to the same solution each time.

 ![Noise-Bias-Varinace](../images/PerfNoiseBiasVariance.svg)

- **Mathematical formulation of test error:**

 ![Noise-Bias-Variance-Equation](../images/Noise-Bias-Variance-Equation.png)

**<u>III - Reducing error:</u>**

The **Noise** component is **insurmountable**, which means there is nothing we can do to avoid it. It represents a <u>fundamental limit</u> on expected model performance. **However**, we can reduce the Variance and Bias terms.

- **Reducing Variance:** variance results from limited noisy training data. This actually means that we can reduce it by **increasing the quantity** of our training data. This approach averages out the inherent noise and ensures that the input space is well sampled.<br>
The figure below shows the effect of training with three different samples (*6, 10 and 100 samples*). The best-fitting model for each dataset is then plotted: as we can see, with only **6 samples**, the fitted function is <u>different</u> each time and the variance term is therefore significant. When we **increase** the number of samples, the fitted models become very <u>similar</u> and the variance term reduces as a result.

![Reducing-Variance](../images/PerfVariance.svg)

=> In general, adding more training data <u>almost always</u> improves test performance.
- **Reducing Bias:**

**<u>IV - Hyperparameters:</u>**

### Chapter 9: Regularization (TO DO !!)

## Discussion Notes

\- Key questions raised during seminar

\- Open problems or unclear points



## References

- IBM - What is overfitting ? [[Link]](https://www.ibm.com/think/topics/overfitting)
- Three Sources of Model Error: Bias, Variance, and Noise [[Link]](https://python.plainenglish.io/three-sources-of-model-error-bias-variance-and-noise-9d740c0ba5d6)

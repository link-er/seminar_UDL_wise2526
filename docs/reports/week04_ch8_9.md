# Report – Week 04: Performance & Regularization



**Presenters:** Ben Halima Ibrahim, Zoghlami Fadi 

**Date:** 10.11.2025 



## Summary

### Chapter 8: Measuring performance

**I - Training a simple model:**

A neural network with **sufficient capacity** will mostly perform well on the training dataset. However, this does not mean that it will generalize well on the testing dataset (which is normally new and unseen data for the model). This causes a big problem especially for real-world scenarios, where the model'performance has to be as good as possible. <br>
==> Our goal is to train a model that **generalizes well on new data**

The test errors have three distinct causes:
- the inherent uncertainty in the task
- the amount of training data
- the choice of model

In the first section of this chapter, a simple model is trained on the [**MNIST-1D**](https://arxiv.org/pdf/2011.14439) dataset, which is 1D analogue of the **MNIST** dataset: each data example is created by randomly transforming one of the templates and adding noise.

![MNIST-1D dataset](../images/MNIST-1D-Dataset.jpg)

Our simple model/neural network consists of **D_i = 40** inputs and **D_o = 10** outputs representing the number of classes the dataset has (numbers form 0 to 9). The neural network has **2** hidden layers each with **D = 100** hidden units. **Multiclass cross-entropy** is used as a loss function with the **Softmax** function to produce class probabilities.<br>
The model is then trained for **6000 steps (150 epochs)** using **SGD** (**S**tochastic **G**radient **D**escent) as a learning algorithm with a learning rate of **0.1** and a batch-size of **100**. After the training process, we tested our trained model on **1000** extra examples from the dataset.

![Train-Test-Error-Loss](../images/Train-Test-Error-Loss.jpg)

**II - Sources of error:**

**III - Reducing error:**

**IV - Hyperparameters:**

### Chapter 9: Regularization (TO DO !!)

## Discussion Notes

\- Key questions raised during seminar

\- Open problems or unclear points



## References

- IBM - What is overfitting ? [[Link]](https://www.ibm.com/think/topics/overfitting)

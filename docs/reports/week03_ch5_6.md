# Report – Week 03: Loss Functions & Fitting Models

**Presenter:** Ivanna Romanovych and Lucas Filipiak

**Date:** 03.11.2025

## Summary

### Loss Functions (Ivanna Romanovych)

#### The General Recipe

The process of adapting a model $f[x, \phi]$ to compute a probability distribution involves three steps:

1.  **Choose a Distribution:** Select a parametric probability distribution $Pr(y|\theta)$ defined over the output domain $y$.
2.  **Predict Parameters:** Configure the machine learning model to predict one or more of the distribution's parameters ($\theta$), such that $\theta = f[x, \phi]$.
3.  **Minimize Negative Log-Likelihood:** Train the model by finding parameters $\phi$ that minimize the negative log-likelihood over the training dataset.

#### The Maximum Likelihood Criterion

The objective is to maximize the likelihood of the parameters. This approach implicitly assumes the data are independent and identically distributed (I.I.D). Because the product of many probabilities can become incredibly small and difficult to represent with finite precision arithmetic, we equivalently maximize the logarithm of the likelihood.

Therefore, the final loss function $L[\phi]$ becomes the minimization of the negative log-likelihood:

$$\hat{\phi} = \underset{\phi}{\mathrm{argmin}} \left[ - \sum_{i=1}^{I} \log[Pr(y_i | f[x_i, \phi])] \right]$$

#### Applications of the Recipe

##### Univariate Regression

**Goal:** Predict a single scalar output $y \in \mathbb{R}$.

- **Distribution:** Univariate Normal distribution defined by mean $\mu$ and variance $\sigma^2$.
- **Model Prediction:** The model usually predicts the mean, $\mu = f[x, \phi]$.
- **Loss Derivation:** Substituting the Normal distribution into the negative log-likelihood formula and performing algebraic manipulations yields the **Least Squares Loss** function.
  $$L[\phi] = \sum_{i=1}^{I} (y_i - f[x_i, \phi])^2$$
  This demonstrates that least squares loss naturally follows the assumption that predictions are drawn from a normal distribution.

**Heteroscedastic Regression:**
If the uncertainty of the model varies as a function of the input data, the model is referred to as **heteroscedastic**. In this case, the model predicts both the mean $\mu$ and the variance $\sigma^2$ (modeled as $f_2[x, \phi]^2$).

#### Binary Classification

**Goal:** Assign data to one of two discrete classes, where $y \in \{0, 1\}$.

- **Distribution:** Bernoulli distribution, which has a single parameter $\lambda$ representing the probability that $y=1$.
- **Model Prediction:** The model predicts $\lambda$. To ensure the prediction is a valid probability $\lambda \in [0, 1]$, the network output is passed through a **sigmoid** function:
  $$\text{sig}[z] = \frac{1}{1 + \exp[-z]}$$
- **Loss Function:** The resulting loss is the **Binary Cross-Entropy** loss:
  $$L[\phi] = \sum_{i=1}^{I} -(1-y_i)\log[1-\text{sig}[f[x_i, \phi]]] - y_i\log[\text{sig}[f[x_i, \phi]]]$$

#### Multiclass Classification

**Goal:** Assign data to one of $K > 2$ classes.

- **Distribution:** Categorical distribution with $K$ parameters ($\lambda_1, ..., \lambda_K$) representing the probability of each category.
- **Model Prediction:** To ensure parameters are positive and sum to one, the network outputs are passed through the **softmax** function:
  $$\text{softmax}_k[z] = \frac{\exp[z_k]}{\sum_{k'=1}^{K} \exp[z_{k'}]}$$
- **Loss Function:** This results in the **Multiclass Cross-Entropy** loss:
  $$L[\phi] = -\sum_{i=1}^{I} \log[\text{softmax}_{y_i}[f[x_i, \phi]]]$$

#### Multiple Outputs

When a model makes multiple predictions simultaneously, the predictions are usually treated as independent. This implies the probability is a product of univariate terms, meaning the loss function becomes a sum of the negative log probabilities for each output.

#### Connection to KL Divergence

The cross-entropy loss can be understood through the lens of **Kullback-Leibler (KL) divergence**. This metric evaluates the distance between the empirical distribution of the observed data, $q(y)$, and the model distribution, $Pr(y|\theta)$.

Minimizing the KL divergence between the empirical distribution (represented as weighted sum of point masses at the data points) and the model distribution is mathematically equivalent to minimizing the negative log-likelihood.

$$\hat{\phi} = \underset{\phi}{\mathrm{argmin}} \left[ -\sum_{i=1}^{I} \log[Pr(y_i | f[x_i, \phi])] \right]$$

---

### Fitting Models (Lucas Filipiak)
> Figures and equations can be found on the slides

#### Model Fitting
- Objective: minimize loss on the training set by adjusting the parameters
- Process: known as *model fitting*

#### Gradient Descent
- Basic idea: change parameters in the direction opposite to the gradient of the loss function, going "downhill" in loss space
- Initialization: parameters start with heuristic values.

##### Example: 1D Linear Regression
- Model: linear function, 2 parameters
- Loss function: least squares 
- Properties:
    - Convex loss surface  
    - Local = global minimum
    - No saddle points

#### Example: Gabor Model
- Nonlinear function involving sinusoidal and exponential terms, 2 parameters
- Loss: least squares
- Properties:
    - Non-convex loss surface
    - Local minima may not be global  
    - Saddle points possible
    - Gradient descent may get "stuck" at local minima and saddle points

#### Stochastic Gradient Descent (SGD)
- Runs gradient descent iteratively on subsets of the training set
- Pass of all training examples: epoch

- Advantages:
    - Adds noise => helps escape local minima  
    - Uses mini-batches => computationally cheaper per iteration  
    - Improves generalization

#### Momentum-based methods

##### SGD with momentum
- Incorporates previous updates into next step

- Advantages:
    - Smoother trajectory
    - Reduced oscillation at valleys

##### Nesterov accelerated momentum
- Computes gradient at position predicted by momentum
- Modification of SGD with momentum, same advantages

#### Adaptive movement estimation (Adam)
- Combines all previous approaches.

##### Modifications:
1. Disconnect step distance from gradient magnitude by normalizing the gradient
2. Apply momentum
3. Accelerate beginning movement
4. Can operate in batches for efficiency
5. Optionally, apply in batches

- Advantages:
    - Smooth trajectory
    - Can converge at minima

## Discussion Notes

- **Why does the presence of noise contributes to better finding the solution?**
    
    Noise in gradient-based optimization, e.g. through stochastic gradient descent, helps the optimizer escape local minima.
    Noise promotes exploration of the parameter space and often leads to solutions with better generalization.

- **What is a full-batch gradient descent?**

    Full-batch gradient descent computes the gradient of the loss function using the entire training dataset for each update.
    While this yields stable and deterministic updates, it is computationally expensive and lack stochasticity that can 
    improve optimization in non-convex problems. 

- **What is the role of the learning rate, and how does it influence the behavior of gradient descent?**
  
    The learning rate controls the step size of parameter updates in the direction of the negative gradient.
    Small learning rates lead to slow but stable convergence, whereas large learning rates accelerate training but may
    cause instability or divergence.   

- **What is the advantage of computing both parameters $\mu$ and $\sigma^2$?**

    Predicting both $\mu$ and $\sigma^2$ allows the model to quantify uncertainty in its predictions. 
    This is particularly important in probabilistic and Bayesian settings.

- **When should Cross Entropy Loss be used, and when is Least Squares Loss more appropriate?**

    Least Squares Loss is best suited for regression tasks under a Gaussian noise assumption.
    Cross Entropy Loss is more appropriate for classification tasks, as it models probability distributions and directly
    optimizes class likelihoods. 
    
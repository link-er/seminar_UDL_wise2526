# Report – Week 09: Diffusion Models

**Presenter:** Alexander Semler

**Date:** 05.01.2026

---

# Presentation Summary

## Overview

### Diffusion Models: Introduction

**Core Idea:** Generate new data by learning to reverse a gradual noise-addition process.

**Encoder (Forward Process)**
- Maps input **x** through latent variables **z₁, …, z_T**
- Gradually mixes data with noise
- Pre-specified (no learning needed)

**Decoder (Reverse Process)**
- Learned neural network
- Removes noise at each stage
- Passes data back through latent variables

**Sampling:** To generate an image, sample noise **z_T** ~ q(**z_T**) and pass through decoder.

### Notation and Key Variables

**Data and Latent Variables**
- **x** — Original data example (e.g., an image)
- **z_t** — Latent variable at diffusion step t
  - t = 1, 2, …, T where T is the total number of steps
  - All **z_t** have the same dimensionality as **x**

**Noise Parameters**
- **ε_t** — Random noise drawn from standard normal N(**0**, **I**)
- **β_t** — Noise schedule parameter at step t
  - Controls how much noise is added at each step
  - β_t ∈ [0, 1] (typically small values, e.g., 0.0001 to 0.02)

## Encoder

### Forward Diffusion Process

**Diffusion Step:**
```
z₁ = √(1-β₁) · x + √β₁ · ε₁
z_t = √(1-β_t) · z_{t-1} + √β_t · ε_t    ∀ t ∈ 2,…,T
```
(Equation 18.1)

**Diffusion Step (probabilistic):**
```
q(z₁|x) = Norm[√(1-β₁) · x, β₁I]
q(z_t|z_{t-1}) = Norm[√(1-β_t) · z_{t-1}, β_tI]
```
(Equation 18.2)

**Joint distribution of all latent variables:**
```
q(z_{1…T}|x) = q(z₁|x) ∏_{t=2}^T q(z_t|z_{t-1})
```
(Equation 18.3)

### Diffusion Kernel q(z_t|x)

**Goal:** Since computing **z_t** iteratively can be highly time consuming, we want to find a closed form expression.

**Approach:** By substituting the equation for **z_{t-1}** into **z_t** repeatedly, we can derive a closed form expression that skips all intermediate variables.

**Closed Form Expression (Diffusion Kernel):**
```
z_t = √α_t · x + √(1-α_t) · ε,    α_t = ∏_{s=1}^t (1-β_s)
```
(Equation 18.7)
```
q(z_t|x) = Norm[√α_t · x, (1-α_t)I]
```
(Equation 18.8)

### Marginal Distribution q(z_t)
```
q(z_t) = ∫∫ q(z_{1…t}|x)Pr(x) dz_{1…t-1} dx
```
(Equation 18.9)
```
q(z_t) = ∫ q(z_t|x)Pr(x) dx
```
(Equation 18.10)

**Note:** The marginal distribution q(**z_t**) cannot be written in closed form because we don't know the original data distribution Pr(**x**).

### Conditional Diffusion Distribution q(z_{t-1}|z_t, x)
```
q(z_{t-1}|z_t, x) = q(z_t|z_{t-1}, x)q(z_{t-1}|x) / q(z_t|x)
```
(Equation 18.12)
```
q(z_{t-1}|z_t, x) = Norm[(√α_{t-1}β_t)/(1-α_t) · x + (√(1-β_t)(1-α_{t-1}))/(1-α_t) · z_t, (β_t(1-α_{t-1}))/(1-α_t)I]
```
(Equation 18.15)

**Why is this useful?** This distribution is used to train the decoder - it tells us the true distribution over **z_{t-1}** when we know both **z_t** and the training example **x**, which we do during training.

## Decoder

**The Challenge:** The true reverse distributions q(**z_{t-1}**|**z_t**) are complex, multi-modal distributions that depend on the unknown data distribution Pr(**x**).

**Approach:** We approximate the reverse process with learned normal distributions:
```
Pr(z_T) = Norm[0, I]
Pr(z_{t-1}|z_t, φ_t) = Norm[f_t[z_t, φ_t], σ_t²I]
Pr(x|z₁, φ₁) = Norm[f₁[z₁, φ₁], σ₁²I]
```
(Equation 18.16)

where f_t[**z_t**, **φ_t**] is a neural network predicting the mean.

## Training

### Training Setup

**Joint Distribution:**
```
Pr(x, z_{1…T}|φ_{1…T}) = Pr(x|z₁, φ₁) ∏_{t=2}^T Pr(z_{t-1}|z_t, φ_t) · Pr(z_T)
```
(Equation 18.17)

**Likelihood:**
```
Pr(x|φ_{1…T}) = ∫ Pr(x, z_{1…T}|φ_{1…T}) dz_{1…T}
```
(Equation 18.18)

**Training Objective:** Maximize the log-likelihood of the training data {**x_i**}:
```
φ̂_{1…T} = argmax_{φ_{1…T}} [∑_{i=1}^I log Pr(x_i|φ_{1…T})]
```
(Equation 18.19)

### Training: ELBO

**The Problem:** The marginalization in equation 18.18 is intractable! We cannot directly maximize the log-likelihood.

**Evidence Lower Bound (ELBO):**
```
ELBO[φ_{1…T}] = ∫ q(z_{1…T}|x) log[Pr(x, z_{1…T}|φ_{1…T}) / q(z_{1…T}|x)] dz_{1…T}
```
(Equation 18.21)
```
ELBO[φ_{1…T}] = E_{q(z₁|x)}[log Pr(x|z₁, φ₁)]
                 - ∑_{t=2}^T E_{q(z_t|x)}[D_KL(q(z_{t-1}|z_t, x) || Pr(z_{t-1}|z_t, φ_t))]
```
(Equation 18.25)

## Reparameterization

### Reparameterization of Loss Function

**Motivation:** Predicting noise **ε** instead of **z_{t-1}** yields better empirical performance.

**Approach:**
1. **Reparameterize target:** Express **x** in terms of noise: **x** = (1/√α_t)**z_t** - (√(1-α_t)/√α_t)**ε**
2. **Reparameterize network:** Replace f_t[**z_t**, **φ_t**] (predicts **z_{t-1}**) with g_t[**z_t**, **φ_t**] (predicts **ε**)

**Final Simplified Loss:**
```
L[φ_{1…T}] = ∑_{i=1}^I ∑_{t=1}^T ||g_t[√α_t · x_i + √(1-α_t) · ε_{it}, φ_t] - ε_{it}||²
```
(Equation 18.40)

Train network to predict the noise that was added to create **z_t** from **x**.






## Implementation

### Application to Images

**Network Architecture: U-Net**
- Image-to-image mapping: noisy image → predicted noise
- Single U-Net shared across all time steps t = 1, …, T
- Time step t encoded as sinusoidal embedding (like positional encoding)

### Improving Generation Speed

**Key Insight:** The loss function (Eq. 18.40) is valid for any forward process with diffusion kernel q(**z_t**|**x**) = Norm[√α_t · **x**, √(1-α_t)**I**]

**Denoising Diffusion Implicit Models**
- No longer stochastic after first step
- i.e. from step z₁ to z_t

**Accelerated Sampling Models**
- Forward process only on sub-sequence of steps
- Skip steps during reverse process

**Result:** Much more efficient sampling

### Conditional Generation

**Goal:** Control generation using labels c (class, text, etc.)

**Classifier Guidance:** Modify denoising update to make class c more likely:
```
z_{t-1} = ẑ_{t-1} + σ_t² · ∂log Pr(c|z_t)/∂z_t + σ_t · ε
```
(Equation 18.41)

- Requires separate classifier Pr(c|**z_t**) trained on latent variables
- Classifier shared across time steps (takes t as input)

**Classifier-Free Guidance**
- Incorporate class directly into main model: g_t[**z_t**, **φ_t**, c]
- Add embedding of c to U-Net layers (similar to time embedding)
- Train on both conditional and unconditional objectives (by randomly dropping class information)

### Improving Generation Quality

**Key Techniques:**

**1. Estimate Variances:**
- Learn σ_t² in addition to mean
- Particularly helps when sampling with fewer steps

**2. Adaptive Noise Schedule:**
- Vary β_t at each step for better results

**3. Cascaded Diffusion Models:**
- First model: generate low-resolution image
- Subsequent models: progressively increase resolution
- Condition on lower-resolution image + class/text info

---
# Discussion Notes


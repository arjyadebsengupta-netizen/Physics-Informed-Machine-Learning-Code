# Physics-Informed-Machine-Learning-Code
This repository contains the Masters Thesis Project Code and some additional codes. 
# Code 1( the thesis code)
This framework runs a 30-fold Monte Carlo experiment to benchmark six different machine learning and physics-informed regression models. The objective is to fit qubit energy relaxation decay curves under severe data constraints and reconstruct the hidden two-state switching distribution caused by Two-Level System (TLS) noise.

## 1. Data Generation Process
The synthetic dataset simulates 2,000 experimental cycles of a qubit undergoing energy relaxation:
* **Time Domain ($t$):** 50 uniform sampling points spanning $0$ to $200\,\mu\text{s}$.
* **Telegraphic Noise (TLS):** A hidden state switches between two discrete baselines with a 4% transition probability per step:
  * State 0: $T_1 = 58.0\,\mu\text{s}$
  * State 1: $T_1 = 42.0\,\mu\text{s}$
* **Fluctuations:** Gaussian noise ($\sigma = 2.0$) is added directly to the $T_1$ values, and independent readout noise ($\sigma = 0.015$) is added to the population probability ($P_e$).
* **Governing Equation:** The underlying physical profile follows single-exponential decay:
  $$\frac{dP_e}{dt} = -\lambda P_e \implies P_e(t) = \exp\left(-\frac{t}{T_1}\right)$$

---

## 2. Model Architectures Evaluated
The framework compares standard data-driven methods, pure physics-informed neural networks, and hybrid architectures:

* **NN (SirenNet):** A deep neural network utilizing Sinusoidal Representation (SIREN) layers with a frequency scaling parameter ($w_0 = 10.0$) to fit the coordinate-to-value mapping.
* **PINN (Physics-Informed Neural Network):** Parallel SIREN networks that jointly predict the population curve $y(t)$ and the continuous decay rate $\lambda(t)$. It optimizes a combined loss function:
  $$\mathcal{L} = e^{-s_d} \mathcal{L}_{\text{data}} + s_d + e^{-s_p} \mathcal{L}_{\text{physics}} + s_p$$
  where $s_d$ and $s_p$ are learned adaptive log-variance parameters used to dynamically balance data fidelity and the physical derivative constraint ($\frac{dy}{dt} + \lambda y = 0$).
* **HYB (Hybrid PINN + Neural Residual):** Features a frozen, pre-trained PINN architecture supplemented by an independent SIREN residual network tasked with fitting the remaining unmodeled data variance.
* **GB (Gradient Boosting):** A standard non-parametric tree ensemble (300 estimators) trained purely on the empirical coordinate data points.
* **PILR (Physics-Informed Linear Regression):** A linear framework that projects inputs onto a combined baseline matrix composed of a constant offset and an exact physical exponential decay basis ($\exp(-t \cdot \lambda_{\text{ref}})$).
* **PILR_GB (Hybrid PILR + Tree Residual):** Pairs the rigid exponential basis of the PILR model with a downstream Gradient Boosting Regressor that fits the residual errors.

---

## 3. Training & Execution Pipeline
For each of the 30 Monte Carlo iterations, the framework executes the following steps:
1. **Data Split:** Shuffles and segments the single target decay curve into an 80/20 train/test split.
2. **Two-Stage Optimization:** Neural networks are trained for 400 epochs using the Adam optimizer ($\text{lr} = 1\times10^{-3}$) to discover coarse patterns, followed by a localized L-BFGS optimization phase with a strong Wolfe line search to enforce strict convergence.
3. **Residual Tree Fitting:** Residuals from both the PINN and the linear physics baselines are isolated and targeted by their respective gradient-boosted secondary models.

---

## 4. Evaluation Metrics
Models are strictly benchmarked across five metrics averaged over all 30 validation runs:
* **Train / Test MSE:** Mean Squared Error calculated on the training and testing data splits to determine curve-fitting precision.
* **$R^2$ Score:** Coefficient of determination on the test set to evaluate total variance explained.
* **Generalization Gap ($\text{Gap}$):** Quantifies over-fitting tendencies using the normalized difference:
  $$\text{Gap} = \frac{|\text{MSE}_{\text{train}} - \text{MSE}_{\text{test}}|}{\text{MSE}_{\text{test}} + 10^{-8}}$$
* **Physics Residual:** Evaluates strict compliance with the underlying physical ordinary differential equation:
  $$\text{Residual} = \frac{1}{N}\sum_{i=1}^N \left( \frac{dy_i}{dt} + \lambda_{\text{ref}} y_i \right)^2$$

---

## 5. TLS Identifiability & Bimodality Analysis
To test if a model preserves the physical integrity of the true underlying noise distribution, the script takes the predicted decay curves, extracts their numerical gradients, and maps them back into a continuous profile of effective lifetimes ($T_1 = 1/\lambda$). 

A Gaussian Mixture Model (GMM) fits both a 1-component (unimodal) and a 2-component (bimodal) distribution over these extracted lifetimes. The model's performance footprint is evaluated via the Bayesian Information Criterion (BIC) drop:
$$\Delta\text{BIC} = \text{BIC}_1 - \text{BIC}_2$$

### Interpretation
* **Positive $\Delta\text{BIC}$:** The model correctly retains the underlying physical traits, confirming that a two-state bimodal switching structure (the hidden $42\,\mu\text{s}$ and $58\,\mu\text{s}$ TLS targets) is preferred.
* **Negative $\Delta\text{BIC}$:** The model fails to resolve the physical parameters, smoothing the predictions into an incorrect, single-component average curve.



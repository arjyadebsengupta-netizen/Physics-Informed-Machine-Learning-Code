# Physics-Informed-Machine-Learning-Code
This repository contains the Masters Thesis Project Code and some additional codes. 
# Code 1( the thesis code)
This framework runs a 30-fold Monte Carlo experiment to benchmark six different machine learning and physics-informed regression models. The objective is to fit qubit energy relaxation decay curves under severe data constraints and reconstruct the hidden two-state switching distribution caused by Two-Level System (TLS) noise.

## 1. Data Generation Process

The synthetic dataset simulates 2,000 experimental cycles of a qubit undergoing energy relaxation:

* **Time Domain:** 50 uniform sampling points spanning 0 to 200 microseconds stored as a vector of shape `(50, 1)`.
* **Telegraphic Noise:** A hidden state switches between two discrete baselines with a 4% transition probability per step: State 0 has a base baseline of 58.0 microseconds, and State 1 has a base baseline of 42.0 microseconds.
* **Fluctuations:** Gaussian noise with a standard deviation of 2.0 is added directly to the lifetime values, and independent readout noise with a standard deviation of 0.015 is added to each element of the population probability vector.

The underlying physical profile follows single-exponential decay:

$$\frac{dP_e}{dt} = -\frac{1}{T_1}P_e \implies P_e(t) = \exp\left(-\frac{t}{T_1}\right)$$

The script targets the very first generated decay curve `y_history[0]` reshaped to `(50, 1)` as the static target array for all training validation steps.

---

## 2. Model Architectures Evaluated

The framework compares standard data-driven methods, pure physics-informed neural networks, and hybrid architectures:

* **NN (SirenNet):** A deep neural network utilizing Sinusoidal Representation (SIREN) layers with a frequency scaling parameter of 10.0 to fit the coordinate-to-value mapping.
* **PINN (Physics-Informed Neural Network):** Parallel SIREN networks that jointly predict the population curve and the continuous decay rate parameter.
* **HYB (Hybrid PINN + Neural Residual):** Features a frozen, pre-trained PINN architecture supplemented by an independent SIREN residual network trained to fit the remaining unmodeled data variance.
* **GB (Gradient Boosting):** A standard non-parametric tree ensemble with 300 estimators trained purely on the empirical coordinate data points.
* **PILR (Physics-Informed Linear Regression):** A linear framework that projects inputs onto a combined baseline matrix composed of a constant offset and the physical exponential decay basis.
* **PILR_GB (Hybrid PILR + Tree Residual):** Pairs the rigid exponential basis of the PILR model with a downstream Gradient Boosting Regressor that fits the residual errors.

The PINN optimizes a combined loss function where the hyper-parameters are clamped between -5 and 5 as learned adaptive log-variance parameters used to dynamically balance data fidelity and the physical derivative constraint:

$$\mathcal{L} = e^{-s_d} \mathcal{L}_{\text{data}} + s_d + e^{-s_p} \mathcal{L}_{\text{physics}} + s_p$$

---

## 3. Training & Execution Pipeline

For each of the 30 Monte Carlo iterations, the framework executes the following steps:

1. **Data Split:** Shuffles and segments the 50 rows of the target decay curve into an 80/20 train/test split (40 points for training, 10 points for testing) using a rolling seed configuration.
2. **Two-Stage Optimization:** Neural networks are trained for 400 epochs using the Adam optimizer with a learning rate of 0.001 to discover coarse patterns, followed by a localized L-BFGS optimization phase with a strong Wolfe line search to enforce strict convergence.
3. **Residual Tree Fitting:** Residuals from both the PINN and the linear physics baselines are isolated and targeted by their respective gradient-boosted secondary models.

---

## 4. Evaluation Metrics

Models are strictly benchmarked across five metrics averaged over all 30 validation runs:

* **Train / Test MSE:** Mean Squared Error calculated on the training and testing data splits to determine curve-fitting precision.
* **R2 Score:** Coefficient of determination on the test set to evaluate total variance explained.
* **Generalization Gap:** Quantifies over-fitting tendencies using the normalized difference calculation:

$$\text{Gap} = \frac{|\text{MSE}_{\text{train}} - \text{MSE}_{\text{test}}|}{\text{MSE}_{\text{test}} + 10^{-8}}$$

* **Physics Residual:** Evaluates strict compliance with the underlying physical ordinary differential equation:

$$\text{Residual} = \frac{1}{N}\sum_{i=1}^N \left( \frac{dy_i}{dt} + \lambda_{\text{ref}} y_i \right)^2$$

---

## 5. TLS Identifiability & Bimodality Analysis

To test if a model preserves the physical integrity of the true underlying noise distribution, the script takes the predicted decay curves, extracts their numerical gradients, and maps them back into a continuous profile of effective lifetimes:

$$T_1 = \frac{1}{\lambda} \quad \text{where} \quad \lambda = -\frac{d}{dt}\left(\ln y_{\text{pred}}\right)$$

A Gaussian Mixture Model (GMM) fits both a 1-component (unimodal) and a 2-component (bimodal) distribution over these extracted lifetimes. The model's performance footprint is evaluated via the Bayesian Information Criterion (BIC) drop:

$$\Delta\text{BIC} = \text{BIC}_1 - \text{BIC}_2$$

### Interpretation

* **Positive Delta BIC:** The model correctly retains the underlying physical traits, confirming that a two-state bimodal switching structure (the hidden 42 microseconds and 58 microseconds TLS targets) is preferred.
* **Negative Delta BIC:** The model fails to resolve the physical parameters, smoothing the predictions into an incorrect, single-component average curve.



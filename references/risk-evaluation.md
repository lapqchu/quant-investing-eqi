# Risk Evaluation Metrics and Tests

Based on Chapter 5 of Paleologo's "Elements of Quantitative Investing", this reference covers comprehensive methods for evaluating factor model risk properties, including covariance matrix accuracy, precision matrix quality, and ancillary tests.

## Overview

Factor model evaluation requires testing across three dimensions:
1. **Covariance Matrix Accuracy** - Volatility prediction quality
2. **Precision Matrix Accuracy** - Mean-variance portfolio optimization quality
3. **Ancillary Metrics** - Model turnover, beta prediction accuracy

## Volatility Forecast Evaluation

### QLIKE Loss Function

The QLIKE (Quasi-Likelihood) loss function is defined as:

```
QLIKE(σ_hat, r) = (1/T) Σ_t [r_t²/σ_hat_t² - log(r_t²/σ_hat_t²) - 1]
```

**Properties:**
- Rank-robust: maintains ranking when comparing forecast accuracy
- Asymmetric: penalizes underestimation more than overestimation
- Related to log-likelihood of normal distribution
- Increasingly preferred over bias statistics

### Mean Squared Error (MSE) Loss Function

```
MSE(σ_hat, r) = (1/T) Σ_t [σ_hat_t² (r_t²/σ_hat_t² - 1)²]
```

**Properties:**
- Rank-robust: comparable to QLIKE for ranking forecasts
- Symmetric loss around true value
- More stable for small variance values
- Computationally efficient

### Rank Robustness

A loss function L is rank-robust if:
```
E[L(σ_hat^(1), σ)] ≤ E[L(σ_hat^(2), σ)] 
⟺ E[L(σ_hat^(1), σ_tilde)] ≤ E[L(σ_hat^(2), σ_tilde)]
```

This ensures that rankings based on observed volatility proxies correspond to true volatility rankings.

## Multivariate Covariance Matrix Evaluation

### 1. Production Strategies

Test covariance matrices through actual portfolio performance:
- Simulate strategies using different factor models
- Evaluate realized Sharpe ratios and PnL
- Ensure models are used consistently (don't mix models between construction and testing)

### 2. Average-Case Analysis

Estimate expected loss over distributions of portfolios and returns:

**Procedure 5.1: Random Portfolios Average Variance Testing**

1. Draw random portfolio weights: **w** ~ N(0, **I**_n)
2. Normalize: **w** ← **w**/|**w**|
3. Select random time period s uniformly from {1,...,T}
4. Compute loss: L_tot ← L_tot + L((**r**_s)^T **w**)², **w**^T Ω_hat_s **w**)
5. Repeat until convergence (L_current/L_tot < tolerance)
6. Output: L̄ = L_tot/n_iter

**Drawbacks:**
- Arbitrary choice of portfolio distribution
- Computationally expensive (high-dimensional simulation)
- Result depends heavily on chosen basis (even eigenportfolios show sensitivity)

### 3. Worst-Case Analysis

Maximize loss over portfolio choices with unit norm constraint:

```
max_w E_r[L(r^T w)², w^T Ω_hat w] 
s.t. |w| ≤ 1
```

**Procedure 5.2: Worst-Case Variance Testing**

1. Initialize random weight vector **w**
2. Gradient step: **w** ← **w** - n_iter^(-1) ∇_w L(...)
3. Normalize weights
4. Repeat until convergence

**Limitation:** Non-convex objective - computationally intractable for large portfolios.

### 4. Leading-Alpha MVO Portfolios

Construct portfolios using realized future returns as proxies for alpha:

**Procedure 5.3: Realized Alpha Variance Testing**

For each time t = 0,...,T-τ:
1. Compute ex-ante alpha: α̂_t = (1/τ) Σ_{s=t+1}^{t+τ} r_s
2. Solve MVO: **w** = Ω_hat_{r,t}^(-1) α̂_t
3. Add loss: L_tot ← L_tot + L((**r**_t)^T **w**)², **w**^T Ω_hat_{r,t} **w**)

Output: L̄ = L_tot/(T - τ + 1)

**Advantages:**
- Tests on economically relevant portfolios
- Can augment with noise: **w** = Ω_hat^(-1)(α̂_t + **η**_t) where **η**_t ~ N(0, σ²**I**)

### 5. Distribution Likelihood (QDIST)

Portfolio-independent test using multivariate normal log-likelihood:

```
QDIST = Σ_t [r_t^T Ω_hat_{r,t}^(-1) r_t + log|Ω_hat_{r,t}| + n log(2π)]
```

Lower values indicate better covariance matrix predictions.

## Precision Matrix Evaluation

### Minimum-Variance Portfolios (Theorem 5.1)

**Key Insight:** A better precision matrix produces lower realized variance portfolios.

Given true covariance **Ω** and estimated covariance Ω_hat, solve:

```
min_w w^T Ω_hat w
s.t. μ^T w = 1
```

Let **w**(Ω_hat) be the optimal portfolio. Then:

**The realized variance var(**w**(Ω_hat), **Ω**) is greater than var(**w**(**Ω**), **Ω**), 
with equality if and only if **Ω** ∝ Ω_hat.**

**Application:** Set μ = α (alpha vector). Correct covariance matrix produces optimal Sharpe ratio.

### Mahalanobis Distance (MALV)

The Mahalanobis distance for a return vector **r** and covariance **Ω** is:

```
d(**r**, **Ω**) = √(r^T Ω^(-1) r)
```

For Gaussian returns under true covariance, d² follows χ²_n distribution.

**MALV Test:**

```
ν_t = (1/n_t) r_t^T Ω_hat_{r,t}^(-1) r_t

MALV = var(ν_1,...,ν_T)
```

**Interpretation:**
- Low MALV (≈ 2/n) indicates precise inverse covariance estimation
- Tests precision matrix independent of portfolio choice
- Can be rewritten as testing closeness of:
  ```
  Ω_r,t^(1/2) Ω_hat_{r,t}^(-1) Ω_r,t^(1/2) ≈ ν_bar I_n
  ```

## Ancillary Tests

### Model Turnover

Measure strategic stability and transaction cost implications.

**Factor-Mimicking Portfolio (FMP) Turnover:**

```
P_t = Ω_ε,t^(-1) B_t (B_t^T Ω_ε,t^(-1) B_t)^(-1)
w_t = P_t b  (portfolio weights for factor exposures)

Turnover_2 = (1/T) Σ_t ||P_t - P_{t-1}||_F
```

where ||·||_F is the Frobenius norm.

**Trading-Cost-Adjusted Turnover:**

```
Turnover_TC = (1/T) Σ_t TC[(P_t - P_{t-1}) b]
```

where TC is a cost function (transaction costs, market impact).

### Beta Accuracy (BETAERR)

Compare predicted vs. realized betas to reference portfolios.

**Predicted Beta:**
```
β_t(**w**) = Ω_{r,t} **w** / (w^T Ω_{r,t} **w**)
```

**Realized Beta (exponentially weighted):**
```
Ω_hat_{r,t} = [(1-e^(-T/τ))/(1-e^(1/τ))] Σ_{s=1}^t e^(-s/τ) r_{t-s} r_{t-s}^T

β_hat_t(**w**) = Ω_hat_{r,t} **w** / (w^T Ω_hat_{r,t} **w**)
```

**Beta Error Metric:**
```
BETAERR(**w**) = Σ_t |β_t - β_hat_t|²
```

Lower values indicate more accurate beta predictions.

## Why R² is Invalid for Factor Model Selection

The coefficient of determination is problematic because:

1. **Data Mining Amenability:** Easy to increase R² through successive adjustments without improving actual performance

2. **Rotational Invariance Violation:** R² is invariant under orthogonal transformations of factor returns, but the estimated factor covariance matrix is not

   Example: Apply random diagonal rotation C_t to factors at each t:
   - R²(B·C_t) = R²(B) [unchanged]
   - But estimated factor covariance becomes decorrelated [different]

3. **No Out-of-Sample Test:** Cannot use holdout samples since factors must be estimated every period

4. **Misleading Improvements:** Adding random factors always increases R², though actual model quality degrades

**Recommendation:** Use QLIKE/MSE for volatility, minimum-variance portfolios for precision, and Sharpe ratios of realized factor returns for factor quality assessment.

## Key Takeaways

1. **QLIKE and MSE** are theoretically justified, rank-robust loss functions for volatility prediction
2. **Multiple evaluation approaches** should be used - no single metric dominates all scenarios
3. **Covariance evaluation** methods range from production strategies to portfolio-independent likelihood tests
4. **Precision matrix accuracy** directly impacts portfolio optimization performance (Theorem 5.1)
5. **Mahalanobis distance (MALV)** provides portfolio-independent precision matrix testing
6. **Ancillary metrics** (turnover, beta accuracy) ensure production readiness
7. **Avoid R²** for factor model comparison - focus on realized performance metrics instead

## References

- Hansen and Lunde (2006): Rank robustness characterization
- Patton and Sheppard (2009), Patton (2011): Loss function theory
- Engle and Colacito (2006): Theorem 5.1 on minimum-variance portfolios

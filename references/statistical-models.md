# Statistical Factor Models (Chapter 7 Reference)

## Overview

Statistical factor models estimate both factor returns and exposures using **Principal Component Analysis (PCA)** on return data alone, without relying on firm characteristics or macroeconomic factors. This contrasts with fundamental models that require additional data sources.

### Key Advantages
- **Complementarity**: Compare solutions across multiple models to identify shortcomings
- **Data availability**: Works when firm characteristics or macroeconomic factors are unavailable
- **Short timescales**: Effective at intraday frequencies (1-5 minutes) where fundamental factors are less relevant
- **Performance**: May outperform other models empirically

### Key Disadvantages
- Loadings are less interpretable than fundamental models (though interpretation is possible)
- Requires careful handling of eigenvector stability and sign ambiguity
- Subject to estimation error in finite samples

---

## Best Low-Rank Approximation and PCA

### Fundamental Problem (Equation 7.1-7.2)

Find factor loadings **B** ∈ ℝ^(n×m) and factor returns **F** ∈ ℝ^(m×T) that minimize the Frobenius norm:

```
min ||R - BF||²_F
```

where **R** ∈ ℝ^(n×T) is the returns matrix, n is number of assets, T is number of periods, m is number of factors.

### SVD Solution (Equation 7.3-7.4)

The Singular Value Decomposition provides the optimal solution. With R = U S V^T:

```
B = U_m
F = S_m V_m^T
```

where U_m and V_m contain the first m columns of U and V, and S_m contains the first m singular values.

### Equivalent PCA Formulation (Equation 7.7-7.11)

Maximize variance subject to unit norm constraint:

```
max w^T Σ̂ w
s.t. ||w|| ≤ 1
```

Connection to eigenvalues: The solution is the eigenvector with the highest eigenvalue λ. Iteratively find orthogonal principal components.

---

## Probabilistic PCA (PPCA)

### Model Specification (Equation 7.12-7.22)

Assume:
- r = B f + ε
- f ~ N(0, I_m)
- ε ~ N(0, σ² I_n)  (isotropic idiosyncratic noise)

Under maximum likelihood estimation:

```
B̂ = U_m (S_m² - σ̂² I_m)^(1/2)
σ̂² = λ̄ (average of smallest n-m eigenvalues)
```

**Key insight**: Sample factor eigenvalues are **upwardly biased**. The bias is higher when the true eigenvalue is closer to the noise floor.

---

## Factor Count Determination

Multiple criteria exist to determine the number of factors. No single criterion is universally correct, but erring toward more factors is safer than too few (risk of underestimating portfolio risk).

### Threshold-Based Method (Equation 7.37)

For spiked covariance models with unit ground eigenvalues:

```
m = max{k | λ̂_k ≥ 1 + √γ}
```

where γ = n/T (ratio of assets to observations).

**Practical guidance**: For 1000-10000 assets and 250-1000 observations, the threshold ranges from 2 to 7.

### Scree Plot

Plot eigenvalues vs rank. Select the last eigenvalue before the "elbow" where eigenvalues stabilize. Logarithmic variant plots log(eigenvalues).

### Maximum Change Points (Equation 7.38)

```
m = arg max_k (λ̂_{k-1} - λ̂_k)
m = arg max_k (log λ̂_{k-1} - log λ̂_k)
```

Select the largest gap between consecutive eigenvalues.

### Penalty-Based Methods (Equation 7.39-7.40)

Add penalty term to residual sum of squares:

```
min_{rank(R̂) ≤ k} ||R - R̂||²_F + k · f(n,T)
f(n,T) = (n + T)/(nT) · log((nT)/(n+T))
```

This is the information criterion approach (similar to AIC/BIC).

---

## Spiked Covariance Model: Spectral Theory

### Model Setup (Equation 7.27-7.28)

Assume m factors with large variances separated from noise:

```
λ_i = { > C·n, for i ≤ m (spiked eigenvalues)
       { ≈ 1, for i > m (noise/bulk eigenvalues)
```

### Spectrum Bounds (Equation 7.30-7.32)

For large n and T with γ = n/T:

**For λ̂_i > 1 + √γ:**
- Sample eigenvalue: λ̂_i → λ_i(1 + γ/λ_i) a.s.
- Eigenvector collinearity: |⟨u_i, û_i⟩| → 1 + √(1/γ_i - c_i) a.s.

**For λ̂_i ≤ 1 + √γ:**
- Sample eigenvalue converges to (1 + √γ)²
- Eigenvector shows no collinearity with population eigenvector

**Critical insight**: Below threshold λ = 1 + √γ, eigenvalues cannot be separated from noise, and eigenvectors contain no information.

---

## Eigenvalue Shrinkage

### Optimal Shrinkage (Equation 7.33)

For operator norm loss:

```
L(λ) = λ - γ, for λ ≥ 1 + √γ
```

For large λ: L(λ) ≈ λ + 1 - γ (constant offset from each eigenvalue).

### Linear Shrinkage

Combine constant offset with proportional scaling:

```
L(λ) = κ₁λ - κ₂
where κ₂ ≥ λ_min, κ₁ ∈ (0,1)
```

Useful for non-Gaussian returns (e.g., heavy-tailed distributions).

### Empirical Finding

For heavy-tailed returns (t-distributed), simple constant-offset shrinkage underperforms. Instead, use proportional scaling of eigenvalues.

---

## Real-World PCA Behavior

### Eigenvector Turnover (Equation 7.41-7.42)

Turnover between consecutive periods for portfolio **v**:

```
turnover₁(v) = ||v_t - v_{t-1}||₁ (L1 norm)
turnover₂(v) = ||v_t - v_{t-1}||² (L2 norm, preferred)
               = 2(1 - |s_c(v_t, v_{t-1})|)
```

where s_c is cosine similarity.

**Key observation**: First eigenvector (market factor) is stable. Higher-order eigenvectors show occasional large turnover spikes when eigenvalues are nearly degenerate.

### Sign Ambiguity (Equation 7.43)

Rotate consecutive loadings to minimize turnover:

```
B̃_{t+1} = arg min ||B_t - B_{t+1}X||²_F
s.t. X^T X = I_m, X ∈ ℝ^(m×m)
```

Solution via SVD: X* = V U^T, where A = U S V^T is SVD of B_t^T B_{t+1}.

This is a zero-cost operation that preserves all model predictions (volatility, optimization).

---

## Statistical Model Estimation in Practice

### Two-Stage PCA with Reweighting (Procedure 7.1)

**Stage 1: Time-Series Reweighting**
- Apply exponential weights with fast half-life τ_f
- Captures rapidly changing volatilities
- Compute first-stage SVD on weighted returns

**Stage 2: Idiosyncratic Reweighting**
- Extract idiosyncratic variance proxy from first m_p components
- Weight assets by inverse idiosyncratic volatility: w_σ = diag(σ₁⁻¹, ..., σₙ⁻¹)
- Apply slower exponential weights with half-life τ_s
- Compute second-stage SVD on doubly reweighted returns

**Rationale**: Volatilities change rapidly; correlations are stable. This separation improves model stability.

### Dynamic Implementation

For time t with estimation window [t - T_max + 1, t]:

**Factor returns** (Equation 7.55-7.56):
```
f̂_t = (B_{t-1}^T w_σ,t-1² B_{t-1})^{-1} B_{t-1}^T w_σ,t-1² r_t
    = Û_m^T w_σ,t-1 r_t
```

**Idiosyncratic returns**:
```
ε̂_t = r_t - B_t f̂_t
```

### Handling Non-Estimation Universe Assets

**For assets with complete returns**: Regress returns on factor returns from estimation universe:

```
β_i = (F^T F)^{-1} F^T r_i
```

where β_i becomes the loading for asset i.

**For new/illiquid assets**: Regress observed loadings on industry/country characteristics:

```
b_i = G γ + η_i
```

and predict missing loadings. Shrink toward zero (useful for hedging applications).

---

## Interpreting Principal Components

### Clustering Interpretation (Equation 7.44-7.51)

PCA can be viewed as **relaxed k-means clustering**. Loadings represent approximate cluster membership:

```
min ∑_{k=1}^K ∑_{i∈C_k} ||r^i - m_k||²  →  max trace(H^T R R^T H)
s.t. H^T H = I_k
```

This optimization (relaxed version) is equivalent to PCA.

**Application**: Inspect loadings distribution to identify natural clusters of assets.

### Regression Interpretation (Section 7.4.2)

Regress PCA loadings on observable stock characteristics (G):

```
b_i = G β^(i) + η_i
```

Regression coefficients reveal which characteristics drive each principal component.

**Example from US equities (Table 7.2-7.3)**:
- **PC1**: Market factor (intercept dominates), negative volatility exposure
- **PC2**: Value/size factor (Dividend Yield, Size positive; Momentum negative)

---

## Key Formulas Summary

| Concept | Formula | Key Use |
|---------|---------|---------|
| Optimal low-rank approximation | min ‖R - R̂‖²_F, rank ≤ m | Define factor model |
| SVD solution | R̂ = U_m S_m V_m^T | Compute loadings/factors |
| PCA optimization | max w^T Σ̂ w, ‖w‖ = 1 | First principal component |
| Eigenvalue bias | λ̂_i → λ_i(1 + γ/λ_i) | Understand sample error |
| Threshold criterion | m = max{k: λ̂_k ≥ 1 + √γ} | Determine factor count |
| Eigenvalue shrinkage | L(λ) = λ - γ | Correct bias |
| Eigenvector turnover | turnover = 2(1 - \|s_c\|) | Measure stability |
| Rotation for turnover | X* = VU^T from A = USV^T | Minimize loadings drift |

---

## Practical Recommendations

1. **Data preparation**: Normalize returns (by idiosyncratic or total volatility) before PCA to improve eigenvalue separation
2. **Factor selection**: Use threshold method (1 + √γ) as starting point; err on side of more factors
3. **Shrinkage**: Always shrink eigenvalues; use proportional scaling for heavy-tailed returns
4. **Stability**: Apply two-stage reweighting with different time scales for volatilities vs correlations
5. **Interpretation**: Combine clustering and regression views of loadings for practical insight
6. **Implementation**: Rotate eigenvectors in each period to minimize turnover for performance attribution
7. **Validation**: Check consistency of first principal component (should be market-like) and compare to fundamental models


# Linear Models of Returns: Factor Models

Reference guide for Chapter 4 of *The Elements of Quantitative Investing*.

## Central Equation

$$\mathbf{r}_t = \boldsymbol{\alpha} + \mathbf{B} \mathbf{f}_t + \boldsymbol{\epsilon}_t$$

### Notation
- $\mathbf{r}_t \in \mathbb{R}^n$: $n$ asset excess returns at time $t$
- $\boldsymbol{\alpha} \in \mathbb{R}^n$: alpha vector (intercepts/expected returns)
- $\mathbf{B} \in \mathbb{R}^{n \times m}$: loadings matrix ($n$ assets, $m$ factors with $m \ll n$)
- $\mathbf{f}_t \in \mathbb{R}^m$: $m$ factor returns at time $t$
- $\boldsymbol{\epsilon}_t \in \mathbb{R}^n$: idiosyncratic returns (asset-specific, uncorrelated with factors)

### Core Assumptions
- $E[\boldsymbol{\epsilon}_t] = 0$
- $\text{cov}(\boldsymbol{\epsilon}_t, \mathbf{f}_s) = 0$ for all $s, t$
- Covariance matrix $\boldsymbol{\Omega}_\epsilon$ is diagonal (or sparse in approximate models)
- Pair $(\mathbf{f}_t, \boldsymbol{\epsilon}_t)$ identically distributed or slowly time-varying

## Covariance Decomposition

$$\boldsymbol{\Omega}_r = \mathbf{B} \boldsymbol{\Omega}_f \mathbf{B}^T + \boldsymbol{\Omega}_\epsilon$$

where:
- $\boldsymbol{\Omega}_r = \text{cov}(\mathbf{r}_t)$: asset return covariance
- $\boldsymbol{\Omega}_f = \text{cov}(\mathbf{f}_t)$: factor return covariance
- $\boldsymbol{\Omega}_\epsilon = \text{cov}(\boldsymbol{\epsilon}_t)$: idiosyncratic covariance (diagonal)

**Proof**: 
$$\text{cov}(\mathbf{r}_t) = \text{cov}(\mathbf{B}\mathbf{f}_t + \boldsymbol{\epsilon}_t) = \mathbf{B} \text{cov}(\mathbf{f}_t) \mathbf{B}^T + \text{cov}(\boldsymbol{\epsilon}_t)$$

since factor and idiosyncratic terms are independent.

## Three Interpretations

### 1. Graphical Model
Each asset return depends on factors through loadings:

$$E[r_i - \alpha_i | \mathbf{f}] = \sum_j B_{ij} f_j$$

Visualize as bipartite graph: factors (circles) connect to assets (squares) via loadings (arrows). When $\mathbf{B}$ is sparse, the graph is sparse.

### 2. Superposition of Effects
Expected excess returns are weighted sum of factor loading vectors:

$$E[\mathbf{r} - \boldsymbol{\alpha} | \mathbf{f}] = \sum_j [\mathbf{B}]_{:,j} f_j$$

The systematic component lives in the column subspace of $\mathbf{B}$, a low-dimensional space.

### 3. Single-Asset Product
For individual asset, expected return is inner product of loadings and factor returns:

$$E[r_i - \alpha_i | \mathbf{f}] = \langle [\mathbf{B}]_{i,:}, \mathbf{f} \rangle$$

Extended to portfolio $\mathbf{w}$:

$$E[\mathbf{w}^T \mathbf{r} | \mathbf{f}] = \sum_i \alpha_i w_i + \sum_i w_i \langle [\mathbf{B}]_{i,:}, \mathbf{f} \rangle$$

This factorizes into per-stock loadings and factor returns—foundation for performance attribution.

## Alpha Spanned vs Alpha Orthogonal

### Decomposition
Decompose $\boldsymbol{\alpha} = \mathbf{B} \boldsymbol{\lambda} + \boldsymbol{\alpha}^\perp$ where:
- $\mathbf{B}^T \boldsymbol{\alpha}^\perp = 0$ (orthogonality condition)
- $\boldsymbol{\alpha}^{\parallel} = \mathbf{B} \boldsymbol{\lambda}$: alpha spanned by factors
- $\boldsymbol{\alpha}^\perp$: alpha orthogonal to factor space

### Rewrite with Expected Factor Returns
Setting $\boldsymbol{\lambda} = 0$ and $\boldsymbol{\mu}_f = E[\mathbf{f}_t]$:

$$\mathbf{r}_t = \boldsymbol{\alpha}^\perp + \mathbf{B}(E[\mathbf{f}_t] + (\mathbf{f}_t - E[\mathbf{f}_t])) + \boldsymbol{\epsilon}_t$$

where $\mathbf{B}^T \boldsymbol{\alpha}^\perp = 0$.

### Sharpe Ratio of Alpha Orthogonal Portfolio

Portfolio proportional to orthogonal alpha: $\mathbf{w} = \boldsymbol{\alpha}^\perp / \|\boldsymbol{\alpha}^\perp\|_1$

Expected return:
$$E[\mathbf{w}^T \mathbf{r}_t] = \|\boldsymbol{\alpha}^\perp\|_2$$

Variance:
$$\text{var}(\mathbf{w}^T \mathbf{r}_t) = \frac{\boldsymbol{\alpha}^{\perp T} \boldsymbol{\Omega}_\epsilon \boldsymbol{\alpha}^\perp}{\|\boldsymbol{\alpha}^\perp\|_1^2}$$

Upper bound on variance via operator norm:
$$\frac{\boldsymbol{\alpha}^{\perp T} \boldsymbol{\Omega}_\epsilon \boldsymbol{\alpha}^\perp}{\|\boldsymbol{\alpha}^\perp\|_2^2} \leq \|\boldsymbol{\Omega}_\epsilon\|_{\text{op}}$$

Thus:
$$\text{SR} \geq \frac{\|\boldsymbol{\alpha}^\perp\|_2}{\sqrt{\|\boldsymbol{\Omega}_\epsilon\|_{\text{op}}}}$$

### Key Theorem (Diversification Benefit)
If average absolute alpha per asset is $\mu > 0$:

$$\frac{1}{n}\sum_i |\alpha^\perp_i| = \mu$$

Then by 1-norm to 2-norm inequality $\|\mathbf{x}\|_1 \leq \sqrt{n} \|\mathbf{x}\|_2$:

$$\|\boldsymbol{\alpha}^\perp\|_2 \geq \sqrt{n} \mu$$

**Result (Equation 4.3)**:
$$\text{SR}_{\text{orthogonal}} \geq \frac{\sqrt{n} \mu}{\sqrt{\|\boldsymbol{\Omega}_\epsilon\|_{\text{op}}}}$$

**Implication**: SR grows like $\sqrt{n}$ with number of assets (pure diversification gain). If actual SRs don't grow this fast, at least one assumption fails: factor model incorrect, or idiosyncratic alphas vanish with $n$.

### Practical Interpretation
1. **Alpha orthogonal is precious**: Positive asset-level alpha orthogonal to factors compounds as $\sqrt{n}$
2. **Alpha spanned is risky**: Comes from factor exposures that don't diversify
3. **Empirical reality**: If observed SRs don't grow like $\sqrt{n}$, orthogonal alphas are likely vanishing in $n$

## Transformations

### 1. Rotations
Factor model is not uniquely identified. For invertible $m \times m$ matrix $\mathbf{C}$:

$$\tilde{\mathbf{B}} = \mathbf{B} \mathbf{C}^{-1}, \quad \tilde{\mathbf{f}} = \mathbf{C} \mathbf{f}$$

The model $\mathbf{r} = \boldsymbol{\alpha} + \tilde{\mathbf{B}} \tilde{\mathbf{f}} + \boldsymbol{\epsilon}$ gives identical predictions.

Covariance of transformed factors:
$$\tilde{\boldsymbol{\Omega}}_f = \mathbf{C} \boldsymbol{\Omega}_f \mathbf{C}^T$$

#### Applications

**Identity factor covariance**: Choose $\mathbf{C} = \mathbf{S}^{-1/2} \mathbf{U}^T$ from SVD $\boldsymbol{\Omega}_f = \mathbf{U} \mathbf{S} \mathbf{U}^T$:

$$\tilde{\boldsymbol{\Omega}}_f = \mathbf{I}_m$$

Uncorrelated factors with unit variances; factor exposures directly interpretable as volatilities.

**Orthonormal loadings**: Transform so $\tilde{\mathbf{B}}^T \tilde{\mathbf{B}} = \mathbf{I}_m$. From SVD $\mathbf{B} = \mathbf{U} \mathbf{S} \mathbf{V}^T$:

$$\mathbf{C}^{-1} = \mathbf{V} \mathbf{S}^{-1}, \quad \tilde{\mathbf{B}} = \mathbf{U}$$

**Z-scored loadings**: Center and scale loadings to zero mean and unit variance. Possible when unit vector $\mathbf{e} = (1, \ldots, 1)^T$ is in column space of $\mathbf{B}$ (e.g., when market factor exists).

### 2. Projections
Reduce dimensionality while approximating original model. Desired loadings matrix $\mathbf{A}$ with column space contained in column space of $\mathbf{B}$:

$$\mathbf{r} = \boldsymbol{\alpha} + \mathbf{A} \mathbf{g} + \boldsymbol{\eta}$$

**Distance minimization**: Minimize $\|\mathbf{B} \mathbf{f} - \mathbf{A} \mathbf{g}\|^2$.

Optimal approximate factors:
$$\mathbf{g} = \mathbf{H} \mathbf{f}, \quad \mathbf{H} = (\mathbf{A}^T \mathbf{A})^{-1} \mathbf{A}^T \mathbf{B}$$

Approximate factor covariance:
$$\boldsymbol{\Omega}_g = \mathbf{H} \boldsymbol{\Omega}_f \mathbf{H}^T$$

$\mathbf{H}$ is idempotent (projection operator).

#### Common Use Cases
1. Nested subset: columns of $\mathbf{A}$ are subset of columns of $\mathbf{B}$
2. Qualitative comparison: subspace of $\mathbf{A}$ may not contain subspace of $\mathbf{B}$; useful for model comparison

### 3. Push-Outs (Factor Addition)
Augment model with additional factors to capture structure in idiosyncratic returns:

$$\boldsymbol{\epsilon} = \mathbf{A} \mathbf{g} + \boldsymbol{\eta}$$

New model:
$$\mathbf{r} = \boldsymbol{\alpha}^\perp + \mathbf{B} \mathbf{f} + \mathbf{A} \mathbf{g} + \boldsymbol{\eta}$$

**Orthogonality constraint**: Require $\mathbf{A}^T \mathbf{B} = 0$ (new factors orthogonal to original ones). Otherwise, factor definitions would need revision.

**Use case**: Original model captures historical regime; new factors explain current regime-specific idiosyncratic structure.

## Applications

### 1. Performance Attribution
Portfolio PnL over period $[t-1, t]$:

$$\text{PnL}_t = \mathbf{w}_t^T \mathbf{r}_t = \mathbf{w}_t^T \mathbf{B} \mathbf{f}_t + \mathbf{w}_t^T (\boldsymbol{\alpha}^\perp + \boldsymbol{\epsilon}_t)$$

Define factor exposures:
$$\mathbf{b}_t = \mathbf{B}^T \mathbf{w}_t$$

Then:
$$\text{PnL}_t = \mathbf{b}_t^T \mathbf{f}_t + \text{idiosyncratic PnL}$$

**Factor PnL**: $\sum_j b_{tj} f_{tj}$ decomposes by factor
**Idiosyncratic PnL**: $\mathbf{w}_t^T (\boldsymbol{\alpha}^\perp + \boldsymbol{\epsilon}_t)$ decomposes by asset

Further partition factors/assets by style, sector, region, or other groups.

### 2. Risk Decomposition
Portfolio variance:
$$\text{var}(\mathbf{w}^T \mathbf{r}) = \mathbf{b}^T \boldsymbol{\Omega}_f \mathbf{b} + \mathbf{w}^T \boldsymbol{\Omega}_\epsilon \mathbf{w}$$

**Systematic variance**: First term (factor risk)
**Idiosyncratic variance**: Second term (specific risk)

**Percentage of variance** for group $i$ (beta interpretation):
$$p_i = \frac{\text{cov}(\text{group } i \text{ return}, \text{total return})}{\text{var}(\text{total return})} = \beta_i$$

**Marginal contribution to risk** (MCR) for group $i$:
$$\text{MCR}_i = \frac{1}{v_i} \times v_{\text{TOT}} \times \sum_j \Omega_{ij} = \rho_i \times \frac{v_{\text{TOT}}}{v_i}$$

where $\rho_i$ is correlation of group to total PnL.

**Sharpe ratio sensitivity**: Change in portfolio SR from increasing group volatility by \$1M is related to MCR and group Sharpe ratio.

### 3. Portfolio Construction
Factor models enable:
- Estimation of asset covariance matrix (well-defined even when $T < n$)
- Separation of factor-driven returns (come with systematic risk) from alpha-driven returns (uncorrelated)
- Legible factor exposures for monitoring and hedging
- Inverse covariance matrix for mean-variance optimization

### 4. Alpha Research
Distinguish:
- **Factor alpha** ($\mathbf{B}^T E[\mathbf{f}_t]$): Expected return from factor exposures (comes with factor risk)
- **True alpha** ($\boldsymbol{\alpha}^\perp$): Orthogonal expected return (diversifiable across portfolio)

Signal researchers focus on $\boldsymbol{\alpha}^\perp$ (pure alpha per asset); portfolio managers balance alpha against systematic factor bets.

## Factor Model Types

### 1. Characteristic Models
**Inputs**: Time series of returns $\mathbf{r}_t$ and asset characteristics $\mathbf{B}_t$ (available at start of period).

**Process**: Estimate factor returns $\mathbf{f}_t$ and idiosyncratic returns $\boldsymbol{\epsilon}_t$ using cross-sectional regression.

**Intuition**: Characteristics (momentum, value, size, etc.) drive returns. Covered in Chapter 6.

### 2. Statistical Models
**Input**: Only asset returns $\mathbf{r}_t$.

**Process**: Estimate $\mathbf{B}, \mathbf{f}_t, \boldsymbol{\epsilon}_t$ simultaneously (e.g., via PCA).

**Advantage**: No need for external characteristic data
**Disadvantage**: Factors may lack economic interpretation

Covered in Chapter 7.

### 3. Macroeconomic Models
**Inputs**: Asset returns $\mathbf{r}_t$ and macroeconomic factor series $\mathbf{f}_t$ (GDP, inflation, interest rates, etc.).

**Process**: Estimate loadings $\mathbf{B}$ and idiosyncratic returns $\boldsymbol{\epsilon}_t$.

**Advantage**: Factors have direct economic meaning
**Disadvantage**: Requires macroeconomic data and may miss financial factors

## Key Theorems and Results

### Theorem 4.1: Frisch–Waugh–Lovell (FWL)
For partitioned regression $\mathbf{y} = [\mathbf{X}_1, \tilde{\mathbf{X}}_2][\boldsymbol{\beta}_1, \boldsymbol{\beta}_2]^T + \boldsymbol{\epsilon}$ where $\tilde{\mathbf{X}}_2$ contains components orthogonal to $\mathbf{X}_1$:

Two-stage estimation gives identical results to joint regression:
1. Regress $\mathbf{y}$ on $\mathbf{X}_1$, get residuals $\hat{\boldsymbol{\eta}}_1$
2. Regress $\mathbf{X}_2$ on $\mathbf{X}_1$, get residuals $\tilde{\mathbf{X}}_2$
3. Regress $\hat{\boldsymbol{\eta}}_1$ on $\tilde{\mathbf{X}}_2$, get $\hat{\boldsymbol{\beta}}_2$

**Application**: Separates contributions of factor groups in performance attribution.

### Why Not Use Empirical Covariance?
When $T \ll n$, empirical covariance $\hat{\boldsymbol{\Omega}}_r = T^{-1} \mathbf{R} \mathbf{R}^T$ (where $\mathbf{R}$ is $n \times T$ return matrix) has:
- Rank at most $T$: many portfolios have zero variance
- Singular in optimization: mean-variance optimizer produces unbounded positions

Factor structure regularizes by imposing $\boldsymbol{\Omega}_r = \mathbf{B} \boldsymbol{\Omega}_f \mathbf{B}^T + \boldsymbol{\Omega}_\epsilon$ with $m \ll T$, yielding full-rank, invertible covariance.

## Rotational Invariance of Risk Decomposition

When factor model is rotated ($\tilde{\mathbf{B}} = \mathbf{B} \mathbf{C}^{-1}$, $\tilde{\mathbf{f}} = \mathbf{C} \mathbf{f}$):

**Total factor variance unchanged**:
$$\tilde{\mathbf{b}}^T \tilde{\boldsymbol{\Omega}}_f \tilde{\mathbf{b}} = \mathbf{b}^T \boldsymbol{\Omega}_f \mathbf{b}$$

**Total idiosyncratic variance unchanged**: Independent of rotation

**Single-factor variance changes**: Rotations allow flexible attribution to meaningful factor groups

**Implication**: Same model can be presented in different bases; choose representation for intuitive interpretation.

---

**References**: Chapter 4, *The Elements of Quantitative Investing* by Giuseppe A. Paleologo.

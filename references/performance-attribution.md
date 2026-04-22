# Performance Attribution: Decomposing Portfolio Returns

Based on Chapters 14-15 of Paleologo's *Elements of Quantitative Investing*

## Overview

Performance attribution decomposes realized PnL into interpretable components to answer fundamental questions about portfolio management: Was performance due to skill or luck? What drove PnL—factors or stock-picking? Did we succeed via asset selection or position sizing?

This reference covers the methodology, the critical paradoxes and their resolution, maximal attribution techniques, and the decomposition into selection and sizing skills.

## Basic PnL Decomposition

Portfolio performance is partitioned into three components. Over period $[\tau_{i-1}, \tau_i]$:

$$PnL = \underbrace{\sum_t (PnL_t - \mathbf{r}_t^T \mathbf{w}_t)}_{\text{Trading PnL}} + \underbrace{\mathbf{b}_t^T \mathbf{f}_t}_{\text{Factor PnL}} + \underbrace{\mathbf{w}_t^T \boldsymbol{\epsilon}_t}_{\text{Idiosyncratic PnL}}$$

where:
- $\mathbf{w}_t$ = portfolio weights
- $\mathbf{r}_t$ = asset returns
- $\mathbf{b}_t = \mathbf{B}^T \mathbf{w}_t$ = factor exposures (loadings matrix $\mathbf{B}$)
- $\mathbf{f}_t$ = factor returns
- $\boldsymbol{\epsilon}_t = \mathbf{r}_t - \mathbf{B}\mathbf{f}_t$ = idiosyncratic returns

**Component Definitions:**
- **Trading PnL**: Intraday trading gains/losses; alpha from market-making or price discovery
- **Position PnL**: Sum of factor and idiosyncratic PnL (what we realize from holding positions)
- **Factor PnL**: Returns attributable to systematic factor exposures
- **Idiosyncratic PnL**: Returns from stock-specific bets

## Performance Attribution with Estimation Error

### The Critical Problem

Estimated factor returns $\hat{\mathbf{f}}_t$ are noisy:

$$\hat{\mathbf{f}}_t = \mathbf{f}_t + \boldsymbol{\eta}_t, \quad \boldsymbol{\eta}_t \sim N(0, (\mathbf{B}^T \boldsymbol{\Omega}_\epsilon^{-1} \mathbf{B})^{-1})$$

This creates **two paradoxes**:

**Paradox 1: Factor-Mimicking Portfolios (FMPs)**
- Each FMP has non-zero idiosyncratic variance: $\sigma^2_{\mathbf{v}_i} = \mathbf{v}_i^T \boldsymbol{\Omega}_\epsilon \mathbf{v}_i > 0$
- Yet FMPs have **zero idiosyncratic PnL** by construction: $\mathbf{v}_i^T \boldsymbol{\epsilon} = 0$
- This holds for all periods, even FMPs with >50% idiosyncratic risk

**Paradox 2: Factor-Neutral Portfolios**
- Portfolio $\mathbf{w}$ with $\mathbf{B}^T\mathbf{w} = 0$ (zero factor exposure)
- Adding any FMP multiple $\lambda\mathbf{v}$ doesn't change idiosyncratic PnL
- But idiosyncratic volatility $\sigma^2_{\mathbf{w}+\lambda\mathbf{v}}$ changes dramatically with $\lambda$
- Same PnL, vastly different predicted volatility

### Resolution: Accounting for Estimation Error

Decompose observed PnL into true and estimated components:

$$\text{(True Factor PnL)}_t = \text{(Estimated Factor PnL)}_t - \mathbf{w}_t^T \mathbf{B} \boldsymbol{\eta}_t$$
$$\text{(True Idio PnL)}_t = \text{(Estimated Idio PnL)}_t + \mathbf{w}_t^T \mathbf{B} \boldsymbol{\eta}_t$$

Over $T$ periods, both factor and idiosyncratic PnL become random variables:

$$\text{(True Factor PnL)} \sim N\left(\sum_t \mathbf{b}_t^T \hat{\mathbf{f}}_t, \sum_t \mathbf{b}_t^T (\mathbf{B}^T\boldsymbol{\Omega}_\epsilon^{-1}\mathbf{B})^{-1} \mathbf{b}_t\right)$$

$$\text{(True Idio PnL)} \sim N\left(\sum_t \mathbf{w}_t^T \hat{\boldsymbol{\epsilon}}_t, \sum_t \mathbf{b}_t^T (\mathbf{B}^T\boldsymbol{\Omega}_\epsilon^{-1}\mathbf{B})^{-1} \mathbf{b}_t\right)$$

**Insight 14.1: Always Report Standard Errors**

When presenting factor-based attribution, include confidence intervals around attributed PnL using the above variances. This acknowledges estimation uncertainty and helps portfolio managers understand the precision of their attributions.

### Paradox Resolution Details

**For Factor Portfolios** (exposure vector $\mathbf{b}_i$ with unit in position $i$):
$$\text{(True Idio PnL)} \sim N(0, T \cdot [(\mathbf{B}^T\boldsymbol{\Omega}_\epsilon^{-1}\mathbf{B})^{-1}]_{ii})$$

Idiosyncratic PnL is zero-mean with variance growing in $T$.

**For Factor-Neutral Portfolios** (augmented with FMP):
$$\text{(True Factor PnL)} \sim N(\lambda \sum_t \hat{\mathbf{f}}_{t,i}, \lambda^2 T [(\mathbf{B}^T\boldsymbol{\Omega}_\epsilon^{-1}\mathbf{B})^{-1}]_{ii})$$
$$\text{(True Idio PnL)} \sim N(\sum_t \mathbf{w}_t^T \hat{\boldsymbol{\epsilon}}_t, \lambda^2 T [(\mathbf{B}^T\boldsymbol{\Omega}_\epsilon^{-1}\mathbf{B})^{-1}]_{ii})$$

Uncertainty grows linearly with hedge magnitude $\lambda$.

## Maximal Performance Attribution

Standard attribution assigns PnL to factors based on current loadings. But factors are correlated—this assignment is ambiguous. Maximal attribution removes this ambiguity by assigning maximum possible PnL to a chosen factor set.

### Four Equivalent Approaches

**1. Cross-Sectional Return Explanation**

Solve:
$$\min_{\boldsymbol{\beta}} E[||\boldsymbol{\eta}||^2] \text{ subject to } \mathbf{r} = \mathbf{b}\mathbf{f} + \boldsymbol{\epsilon}, \quad \mathbf{r} = \boldsymbol{\beta}\mathbf{f}_S + \boldsymbol{\eta}$$

Solution:
$$\boldsymbol{\beta} = \mathbf{B}\boldsymbol{\Omega}_{U,S}\boldsymbol{\Omega}_{S,S}^{-1}$$

Maximal PnL for factors in set $S$:
$$PnL_S^{\max} = \mathbf{b}^T \boldsymbol{\Omega}_{U,S}\boldsymbol{\Omega}_{S,S}^{-1} \mathbf{f}_S$$

**2. Conditional Expectation**

Given factor returns $\mathbf{f}_S$, the expected returns of omitted factors are:
$$E[\mathbf{f}_{\bar{S}} | \mathbf{f}_S] = \boldsymbol{\Omega}_{\bar{S},S}\boldsymbol{\Omega}_{S,S}^{-1}\mathbf{f}_S$$

Attribution becomes:
$$PnL = \mathbf{b}_S^T \mathbf{f}_S + \mathbf{b}_{\bar{S}}^T \boldsymbol{\Omega}_{\bar{S},S}\boldsymbol{\Omega}_{S,S}^{-1}\mathbf{f}_S + \text{(residual)}$$

**3. Portfolio PnL Optimization**

Explain portfolio PnL by solving:
$$\min_{\tilde{\mathbf{b}}} E[|\mathbf{b}^T\mathbf{f} - \tilde{\mathbf{b}}^T\mathbf{f}_S|^2]$$

Yields adjusted dollar betas:
$$\tilde{\mathbf{b}} = \boldsymbol{\Omega}_{S,S}^{-1}\boldsymbol{\Omega}_{S,U}\mathbf{b}$$

**4. Factor Model Rotation**

Rotate the factor model via transformation $\mathbf{C}$:
$$\mathbf{B}_{\text{new}} = \mathbf{B}\mathbf{C}, \quad \mathbf{f}_{\text{new}} = \mathbf{C}^{-1}\mathbf{f}$$

Choose $\mathbf{C}$ to make first $p$ factors orthogonal to the rest:
$$\mathbf{C} = \begin{pmatrix} I_{S,S} & 0 \\ \mathbf{A} & I_{\bar{S},\bar{S}} \end{pmatrix}$$

where $\mathbf{A} = \boldsymbol{\Omega}_{U,S}\boldsymbol{\Omega}_{S,S}^{-1}$

In rotated model, factors in $S$ have:
$$[\mathbf{B}^T\mathbf{C}]_S = \mathbf{b}_S^T + \mathbf{b}_{\bar{S}}^T \boldsymbol{\Omega}_{U,S}\boldsymbol{\Omega}_{S,S}^{-1} = \mathbf{b}^T\boldsymbol{\Omega}_{U,S}\boldsymbol{\Omega}_{S,S}^{-1}$$

New covariance matrix has block structure:
$$\mathbf{C}^{-1}\boldsymbol{\Omega}(\mathbf{C}^{-1})^T = \begin{pmatrix} \boldsymbol{\Omega}_{S,S} & 0 \\ 0 & \boldsymbol{\Omega}_{\bar{S},\bar{S}} - \boldsymbol{\Omega}_{\bar{S},S}\boldsymbol{\Omega}_{S,S}^{-1}\boldsymbol{\Omega}_{S,\bar{S}} \end{pmatrix}$$

### Nested Maximal Attribution

Extend to partition of all factors into groups $\mathcal{S}_1, \mathcal{S}_2, \ldots, \mathcal{S}_p$:
1. Apply maximal attribution to $\mathcal{S}_1$ first
2. Apply maximal attribution to $\mathcal{S}_2$ on remaining PnL
3. Continue sequentially

Effective for thematic groupings (market factors, value factors, sentiment, etc.).

### Procedures for Implementation

**Procedure 14.1: Maximal Attribution**
1. Input: $\boldsymbol{\Omega}, \mathbf{B}, \mathcal{S}, \bar{\mathcal{S}} = \mathcal{U} \setminus \mathcal{S}, \mathbf{w}$
2. Compute:
   - $\mathbf{b} = \mathbf{B}^T \mathbf{w}$
   - $\mathbf{A} = \boldsymbol{\Omega}_{\mathcal{U},\mathcal{S}}\boldsymbol{\Omega}_{\mathcal{S},\mathcal{S}}^{-1}$
   - Rotation matrix $\mathbf{C}$
3. Output: Per-factor maximal PnL: $PnL_k = [\mathbf{b}^T\mathbf{A}]_k f_k$ for $k \in \mathcal{S}$

**Procedure 14.2: Nested Maximal Attribution**
1. For each factor group $\mathcal{S}_i$ (in order):
   - Apply Procedure 14.1 on current $\boldsymbol{\Omega}, \mathbf{B}, \mathcal{S}_i$
   - Record PnL for group $i$
   - Update $\boldsymbol{\Omega} \leftarrow$ rotated covariance, $\mathbf{b} \leftarrow$ orthogonal exposures
2. Returns hierarchical PnL decomposition

## Selection vs. Sizing Decomposition

Idiosyncratic PnL can be split into two components reflecting distinct investor skills:

### Definition: Information Ratio Decomposition

$$IR = \frac{1}{T}\sum_{t=1}^T \left[(\text{Selection})_t \times (\text{Diversification})_t + (\text{Sizing})_t\right]$$

where $IR = \frac{\text{Idio PnL}}{\text{Idio Vol}}$ is the Information Ratio.

### Selection Skill

Measures ability to choose the right assets:

$$(\text{Selection})_t := \frac{1}{n}\sum_{i=1}^n \tilde{\epsilon}_{t,i} \cdot \text{sgn}(w_{t,i})$$

where $\tilde{\epsilon}_{t,i} = \epsilon_{t,i} / \sigma_i$ (z-scored idiosyncratic returns).

**Interpretation**: Positive when long positions outperform and short positions underperform (on z-scored basis). Rewards picking right securities, not just right size.

### Diversification

Characterizes portfolio concentration:

$$(\text{Diversification})_t := \frac{||\tilde{\mathbf{w}}_t||_1}{||\tilde{\mathbf{w}}_t||_2}$$

where $\tilde{w}_{t,i} = \sigma_i w_{t,i}$ (dollar volatility of position).

**Properties:**
- For equal dollar volatilities: $= \sqrt{n}$ (number of positions)
- For single position: $= 1$
- Equivalent to $1/\sqrt{H}$ where $H$ is Herfindahl index

### Sizing Skill

Measures ability to size positions appropriately:

$$(\text{Sizing})_t := \sqrt{n} \cdot \widehat{\text{cor}}(\tilde{\epsilon}_{t,i} \cdot \text{sgn}(w_{t,i}), |\tilde{w}_{t,i}|)$$

where $\widehat{\text{cor}}$ is cross-sectional correlation.

**Interpretation**: Positive if:
- Large positions when right ($\tilde{\epsilon}_i > 0, w_i > 0$)
- Large positions when right ($\tilde{\epsilon}_i < 0, w_i < 0$)

**Negative sizing** means: manager is larger when wrong. Solution: equalize position sizes.

### Theorem 14.1: Selection-Sizing Identity

For portfolio sequence $\{\mathbf{w}_t\}$ and iid idiosyncratic returns $\{\boldsymbol{\epsilon}_t\}$ with covariance $\boldsymbol{\Omega}$:

$$\widehat{IR} = \frac{1}{T}\sum_{t=1}^T \left[(\text{Selection})_t \times (\text{Diversification})_t + (\text{Sizing})_t\right]$$

This decomposes IR into three actionable components tied to investor skill.

### Practical Guidance

Three levers to improve IR:

1. **Increase Diversification** (lowest effort, automatic benefit)
   - Make positions more equal in dollar volatility
   - Benefit accrues via selection (marginal benefit of diversification)
   - Trade-off: May lower selection skill per stock if universe expands

2. **Improve Selection Skill** (core competence)
   - Track selection at sub-industry or thematic level
   - Compare performance across earnings vs. non-earnings periods
   - Identify sectors/styles where selection is strongest
   - Requires deep fundamental research

3. **Manage Sizing Skill** (critical if negative)
   - Measure sizing relative to selection
   - If negative: equalize position sizes to eliminate drag
   - If positive: optimize position sizing by conviction
   - Positive sizing often less reliable than selection; be cautious

### Long-Short Attribution

Selection skill naturally decomposes into long and short contributions:

$$(\text{Selection})_t = \frac{n_L}{n} (\text{Selection})_{L,t} + \frac{n_S}{n} (\text{Selection})_{S,t}$$

where:
- $n_L, n_S$ = number of long, short positions
- $(\text{Selection})_{L,t}$ = selection among longs
- $(\text{Selection})_{S,t}$ = selection among shorts

Enables diagnosing whether alpha comes from long or short book.

## Practical Implementation

### Data Requirements
- Daily portfolio weights $\mathbf{w}_t$ (or epoch-based)
- Asset returns $\mathbf{r}_t$
- Factor loadings matrix $\mathbf{B}$ (time-varying or static)
- Factor returns $\mathbf{f}_t$
- Factor covariance matrix $\boldsymbol{\Omega}$
- Asset-level idiosyncratic volatilities $\sigma_i$

### Workflow

1. **Estimate factor model**
   - Cross-sectional regression or PCA
   - Compute covariance $\boldsymbol{\Omega}$, idiosyncratic volatilities

2. **Compute basic attribution**
   - Factor PnL: $\sum_t \mathbf{b}_t^T \hat{\mathbf{f}}_t$
   - Idiosyncratic PnL: $\sum_t \mathbf{w}_t^T \hat{\boldsymbol{\epsilon}}_t$

3. **Add uncertainty bands** (Insight 14.1)
   - Standard errors from $(\mathbf{B}^T\boldsymbol{\Omega}_\epsilon^{-1}\mathbf{B})^{-1}$
   - 95% confidence intervals

4. **Apply maximal attribution** (if needed)
   - Identify factor groups of interest
   - Rotate model to orthogonalize
   - Report maximal PnL for each group

5. **Decompose idiosyncratic PnL**
   - Compute selection, diversification, sizing
   - Identify skill drivers
   - Track over time

6. **Report and interpret**
   - Compare to benchmarks
   - Identify improvement opportunities
   - Adjust strategy if needed

## Consistency Check: Beta vs. Exposure

**Common Pitfall**: Confusing beta with exposure.

Portfolio beta to factor $i$:
$$\beta_i = \frac{\text{Cov}(\mathbf{w}^T\mathbf{r}, f_i)}{Var(f_i)}$$

Can be non-zero even if exposure $b_i = 0$, due to correlations with other factors:
$$\beta_i = \sum_{k} b_k \frac{\text{Cov}(f_k, f_i)}{Var(f_i)}$$

Expected PnL from factor $i$ depends on beta, not exposure. This is why standard attribution can be misleading when factors are correlated.

## Summary: Key Formulas

| Concept | Formula |
|---------|---------|
| Basic attribution | $\mathbf{b}_t^T\mathbf{f}_t$ (factor), $\mathbf{w}_t^T\boldsymbol{\epsilon}_t$ (idio) |
| Estimation error | $\boldsymbol{\eta}_t \sim N(0, (\mathbf{B}^T\boldsymbol{\Omega}_\epsilon^{-1}\mathbf{B})^{-1})$ |
| Maximal attribution | $\mathbf{b}^T\boldsymbol{\Omega}_{U,S}\boldsymbol{\Omega}_{S,S}^{-1}\mathbf{f}_S$ |
| Selection | $\frac{1}{n}\sum_i \tilde{\epsilon}_{i} \cdot \text{sgn}(w_i)$ |
| Diversification | $\|\tilde{\mathbf{w}}\|_1 / \|\tilde{\mathbf{w}}\|_2$ |
| Sizing | $\sqrt{n} \cdot \text{cor}(\tilde{\epsilon}_i \cdot \text{sgn}(w_i), \|\tilde{w}_i\|)$ |
| Information Ratio | $IR = (Selection \times Diversification + Sizing)/T$ |

## Key Takeaways

1. **Attribution reveals truth**: Decomposes PnL into factor and idiosyncratic sources; deterministic but must acknowledge estimation uncertainty.

2. **Two critical paradoxes** highlight that factor models are ambiguous representations; maximal attribution resolves this via factor rotation.

3. **Standard errors matter**: Report confidence intervals around attributed PnL to acknowledge model uncertainty.

4. **Maximal attribution** gives unique, unambiguous attribution to selected factor groups through model rotation.

5. **Selection vs. sizing** decomposes idiosyncratic skill into actionable components tied to portfolio construction choices.

6. **Selection = right asset choice; Sizing = right position size**: Both contribute to IR but often have opposite skill signals.

7. **Three improvement levers**: Diversify, improve selection, manage sizing—each with different effort and reliability.

8. **Nested attribution** enables hierarchical analysis of thematic factor groups (market, value, sentiment, etc.).

9. **Long-short split** allows diagnosis of alpha sources by book side.

10. **Practical reality**: Successful managers naturally track these metrics even without formal framework; formalization enables rigor and improvement.

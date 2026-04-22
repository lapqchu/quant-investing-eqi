# Portfolio Optimization Reference

## Overview

This reference covers mean-variance optimization (MVO), factor models in portfolio construction, trading in factor space, information ratios, signal aggregation, and market-impact-aware portfolio management across Chapters 9-11.

---

## 1. Mean-Variance Optimization Fundamentals

### MVO Objective and Justification

The MVO problem stems from maximizing the expected utility of wealth. Using a local polynomial approximation of the utility function:

**Objective Function:**
```
max_w  [μ'w - (ρ/2)(w'Ωw)]
```

where:
- w: portfolio weights
- μ: vector of expected excess returns
- Ω: return covariance matrix
- ρ: coefficient of absolute risk aversion (CARA)

**Why MVO?**
- **Interpretability**: Simple, transparent optimization
- **Data efficiency**: Requires only first two moments
- **Computational tractability**: Solvable in seconds by commercial solvers
- **Single-period setting**: Covers vast majority of practical applications

### Volatility-Constrained Solution

**Formulation:**
```
max_w  α'w
s.t.  √(w'Ωw) ≤ σ
```

**Solution:**
```
w* = σ / √(α'Ω⁻¹α) × Ω⁻¹α

E[r'w*] = σ × √(α'Ω⁻¹α)

SR* = √(α'Ω⁻¹α)
```

**Shadow Price (Lagrange Multiplier):**
```
λ* = √(α'Ω⁻¹α) / (2σ)
```
If variance budget increases by 1 unit, expected return increases by λ*.

### Sharpe Ratio Formulation with Volatility Normalization

From volatilities {σ₁,...,σₙ}, correlation matrix C, and Sharpe ratios {s₁,...,sₙ}:

**Optimal Dollar Volatilities:**
```
v* = (1/(2λ)) × C⁻¹s
```

**Optimal Sharpe Ratio:**
```
SR* = √(s'C⁻¹s)
```

### Key Insights

**Insight 9.1: Miscalibration of alpha size is not catastrophic**
- If you have accurate relative alphas and good volatility model
- Error in absolute size of alphas doesn't affect portfolio composition (homogeneous of degree 0 in alpha)

**Insight 9.2: Asset correlations and diversification limits**
- With uncorrelated assets: SR* = √(Σsᵢ²)
- With pairwise correlation ρ ≠ 0: SR* depends on dispersion of Sharpe ratios
- For many assets with equal Sharpe s: SR* = s/√ρ
- **Upper bound**: Portfolio Sharpe ratio limited by correlation structure

**Insight 9.3: Interpretation of precision matrix**
- Optimal positions: wᵢ ∝ [Ω⁻¹]ᵢⱼ × (αᵢ - Σⱼ≠ᵢ ρᵢⱼ × [Ω⁻¹]ⱼⱼ × αⱼ)
- Diagonal terms of Ω⁻¹ are always positive
- Positive partial correlation between assets reduces portfolio size (collinearity penalty)

---

## 2. Factor Space Decomposition of PnL

### Factor-Mimicking Portfolios (FMPs)

**Definition**: Portfolios with unit exposure to exactly one factor and minimum idiosyncratic risk.

**Construction:** Minimize tracking variance to factor i:
```
min_w  w'Ω_ε w
s.t.   b'w = eᵢ
```

**Solution:**
```
vᵢ = Ω_ε⁻¹ b (b'Ω_ε⁻¹ b)⁻¹ eᵢ

P = [v₁ | v₂ | ... | vₘ] = Ω_ε⁻¹ b (b'Ω_ε⁻¹ b)⁻¹
```

where:
- P: matrix of FMP portfolios
- b: factor loadings matrix
- Ω_ε: idiosyncratic covariance matrix

**Key Properties:**
- FMPs emerge naturally from MVO with low idiosyncratic variance assumption
- Trading in factor space reduces dimensionality: m-factor problem instead of n-asset problem
- FMPs are associated with specific loadings matrix; many equivalent bases exist

### Trading in Factor Space

**Factor Space Optimization:**
```
max_u  λ'u - (γ/2)(u'Ω_f u)

where u: factor exposures, λ: expected factor returns, Ω_f: factor covariance matrix
```

**Optimal Portfolio:**
```
w* = P u*
```

### Adding a New Factor

**Procedure 9.1: Adding factor to model**

1. **Orthogonalization** (to existing factors):
```
b_{m+1} = [I_n - b(b'Ω_ε⁻¹b)⁻¹b'Ω_ε⁻¹] a
```

2. **Estimation** (via Frisch-Waugh-Lovell):
```
f̂_{m+1,t} = (b'_{m+1}Ω_ε⁻¹ b_{m+1})⁻¹ × b'_{m+1}Ω_ε⁻¹ ε_t

λ̂_{m+1} = (1/T) Σ_t f̂_{m+1,t}
```

3. **FMP Construction:**
```
v_{m+1} = (Ω_ε⁻¹ b_{m+1}) / (b'_{m+1}Ω_ε⁻¹ b_{m+1})
```

---

## 3. Trading in Idiosyncratic Space

**Pure Alpha Portfolio** (zero factor exposure):
```
max_w  α_⊥'w
s.t.   b'w = 0
        w'Ω_ε w ≤ σ²
```

**Solution:**
```
ŵ = σ / √(α̂_⊥'Ω_ε⁻¹ α̂_⊥) × Ω_ε⁻¹ α̂_⊥

where α̂_⊥ = [I_n - b(b'Ω_ε⁻¹b)⁻¹b'Ω_ε⁻¹] α_⊥
```

Alpha orthogonal is "golden currency" in investing: Sharpe ratio scales at least like √n.

---

## 4. Information Ratio and Diversification

### IR = IC × √N Fundamental Law of Active Management

**Information Coefficient (IC):**
IC is a correlation between forecasted alphas and realized returns.

**Whitened transformation:**
```
r̃ = Ω_r^{-1/2} r
α̂ = Ω_r^{-1/2} α
```

**IC Definition:**
```
IC = E[r̃'α̂] / √(α̂'α̂ × E[r̃'r̃])

E[r̃'Ω_r⁻¹r̃] = n
```

**Fundamental Law (Grinold & Kahn):**
```
SR = IC × √n
```

where n is the number of cross-sectional assets.

### Practitioners' Version

Using idiosyncratic space (α_⊥, ε):
```
SR = IC × √n

where IC = corr(standardized α_⊥, standardized ε)
```

### Annualized IR with Multiple Forecasts

```
IR_annual = IC × √n × √T

where T = number of independent forecasts per year
```

**From cross-sectional R²:**
```
R² = IC²
IR = √(R² × n)
```

---

## 5. Signal Aggregation: Centralized vs. Decentralized

### Centralized Approach
- Single portfolio manager aggregates all signals
- Produces single portfolio
- Can enforce global risk limits and constraints
- Optimal when signals are highly correlated

### Decentralized Approach
- Multiple managers, each with independent signal
- Each manager optimizes own sub-portfolio
- Sum of sub-portfolios = total portfolio
- Equivalent to centralized under certain conditions

**Equivalence Condition (Theorem 9.1):**
If alpha spanned = 0 and idiosyncratic variance of FMPs is small:
```
decentralized approach ≡ centralized approach (when optimizing in factor space)
```

---

## 6. Shortcomings of Naïve MVO and Regularization

### Key Problems

1. **Estimation Error**: Small errors in α and Σ produce large portfolio changes
2. **Extreme Positions**: Optimal w may be unrealistic (short huge amounts of some assets)
3. **Counterintuitive Shorting**: Assets with positive α may be shorted
4. **Parameter Sensitivity**: Solution very sensitive to input parameters

### Example: Two-Asset Case
```
v₁* = κ/(1-ρ²) × (s₁ - ρs₂)
v₂* = κ/(1-ρ²) × (s₂ - ρs₁)
```
If ρ is high and Sharpe ratios differ, optimal positions can be extreme (long one, short the other).

### Constraints as Regularization

**Weight Bounds:**
```
max_w  α'w - (λ/2)(w'Ωw)
s.t.   -w_max ≤ wᵢ ≤ w_max
```
Acts as L-infinity regularization.

**Sector Limits:**
```
max_w  α'w - (λ/2)(w'Ωw)
s.t.   |sector_k| ≤ exposure_max
```
Controls concentration.

**Gross Exposure Constraint:**
```
1'|w| ≤ gross_exposure
```
Limits total leverage.

### Estimation Error Effects on Sharpe Ratio

When forecast error in expected returns ≈ Gaussian with standard deviation ε_α:
```
E[realized SR] ≈ true SR - (c₀/√n) × (ε_α / σ_α)

where c₀ is a constant depending on problem structure
```

**Key insight:** Misestimation significantly degrades realized Sharpe ratio, especially for large n.

---

## 7. Market Impact and Transaction Costs

### Temporary vs. Permanent Impact

**Total Transaction Cost:**
```
Cost = Spread Cost + Temporary Impact + Permanent Impact
```

**Spread Cost**: Difference between bid and ask, proportional to notional traded

**Temporary Impact**: Price change during execution, decays after

**Permanent Impact**: Long-term price change post-execution

### Temporary Market Impact Models

**General Form (Equation 11.1):**
```
E[P_T - P_0] = κ ∫₀ᵀ f(ẋ_t) × g(T - t) dt
```

where:
- f(ẋ_t): instantaneous market impact function
- g(T-t): propagator (decay function)
- κ: impact coefficient

#### Almgren-Chriss Model

```
f(ẋ) = σ × sgn(ẋ) × |ẋ|^β / v^β
g(t) = δ(t)  (Dirac delta)

β ≈ 0.6
```

**Trading Cost (constant execution):**
```
C = κ × σ × (Q/V)^β × Q

Unit cost: c = κ × σ × (Q/V)^β
```

where Q = quantity traded, V = market volume, participation rate = Q/V.

#### Kyle Model (special case)

```
f(ẋ) = σ × ẋ / v
g(t) = δ(t)
```

Robust to price manipulation; analytically tractable.

#### Obizhaeva-Wang Model

```
f(ẋ) = ẋ / v
g(t) = e^{-t/τ}
```

**Trading Cost:**
```
C = κ × τ × [1 - (τ/T)(1 - e^{-T/τ})] × (Q/V)

τ = market impact half-life
```

Separates fast and slow execution regimes.

#### Gatheral Model

```
f(ẋ) = σ × sgn(ẋ) × |ẋ|^{1/2} / v^{1/2}
g(t) = 1/√t
```

**Trading Cost:**
```
C = (4/3) × κ × σ × √(Q×T/V) × Q

Unit cost independent of execution time
```

### Square-Root Law

From dimensional analysis, market impact scales as:
```
c ∝ σ × √(Q/V)
```
Valid across multiple market impact models.

---

## 8. Almgren-Chriss Framework

### Finite-Horizon Optimization

Trade across m time intervals [t₀, t₁, ..., t_m] with trading rates zᵢ.

**Transaction Cost:**
```
Cost_transaction = -p × Σᵢ Δᵢ|zᵢ|
```

**Impact Cost:**
```
Cost_impact = -κ × Σᵢ Σⱼ aᵢⱼ × zᵢ × f(zⱼ)

where aᵢⱼ = ∫_{tᵢ₋₁}^{tᵢ} [∫_{tⱼ₋₁}^{tⱼ} g(s-u) du] ds
```

**Variance Penalty:**
```
Var(x_t) = ∫₀ᵗ x_s'Ω x_s ds ≈ Σᵢ [Δᵢ × xᵢ₋₁'Ω xᵢ₋₁ + (Δᵢ²/3) × zᵢ'Ω zᵢ]
```

**Full Optimization:**
```
max_{zᵢ}  Σᵢ μᵢ × Δᵢ × xᵢ₋₁ - transaction cost - impact cost - (1/(2ρ)) × variance penalty
s.t.     Hᵢ xᵢ ≤ bᵢ  (linear constraints for each stage)
         x₀, x_m given
```

**Advantages:**
- Flexible: any market impact model, any constraints
- Accounts for impact decay over time

**Disadvantages:**
- Must solve numerically (execution delays)
- Convergence depends on concavity (not always guaranteed)
- Doesn't account for forecast updates

---

## 9. Infinite-Horizon Optimization

### Quadratic Costs Formulation

**Continuous-time objective:**
```
max_{x_t} E ∫₀^∞ [μ_t'x_t - (1/2)ẋ_t'Cẋ_t - (ρ/2)x_t'Ωx_t] dt
```

where:
- C: diagonal positive-definite cost matrix
- ρ: risk aversion parameter
- μ_t: stochastic alpha process

### Optimal Trading Policy

**Define:**
```
Γ = (ρC⁻¹Ω)^{1/2}

b_t = ∫_t^∞ e^{Γ(t-s)} C⁻¹ E_t[μ_s] ds
```

**Optimal Policy (Procedure 11.1):**
```
x_t = e^{-Γt} (x_0 + ∫₀ᵗ e^{Γs} b_s ds)

ẋ_t = -Γx_t + b_t
```

**Interpretation:**
- First term: liquidate position at rate Γ
- Second term: invest in future alphas

**Parameter Effects:**
- Higher ρ (risk aversion) or σ (volatility) → faster liquidation
- Higher costs C → slower liquidation (want to hold longer to justify trading)

### Special Cases

#### No-Market-Impact Limit (C → 0)
```
x_t = ρ⁻¹Ω⁻¹μ_t
```
Instantaneously rebalance to MVO allocation.

#### Optimal Liquidation (μ_t = 0)
```
x_t = e^{-Γt} x_0
```
Exponential decay at rate Γ.

#### Deterministic Alpha
```
b_t = ∫_t^∞ e^{Γ(t-s)} C⁻¹ μ_s ds

x_t = e^{-Γt} [x_0 + ∫₀ᵗ e^{Γs} b_s ds]
```

#### AR(1) Signal Process
```
μ_{t+1} = Φμ_t + η_t

Optimal trades combine liquidation and dynamic alpha response:
Δx_t = [-Γx_t + Kμ_t] × Δt

where K matrix depends on cost structure and signal persistence
```

---

## 10. Capacity Bounds

Capacity reflects:
1. Maximum position sizes in liquid securities
2. Market volume constraints
3. Impact-driven limits on total AUM

**Participation Rate Constraint:**
```
Q/V ≤ max_pov

where Q = quantity trading, V = market volume, max_pov = max participation rate
```

**Impact-Aware Capacity:**
```
Realized SR_annual = True SR - (1/√(AUM_millions)) × degradation_factor
```

Larger AUM → larger market impact → lower realized Sharpe ratios.

---

## References and Key Theorems

**Theorem 9.1**: FMP Optimality
- If alpha spanned = 0 and idiosyncratic variance of FMPs → 0
- Then MVO problem reduces to optimization in factor space
- Solution: w* = P u* where u* solves low-dimensional factor problem

**Theorem 9.2**: Factor Covariance Shrinkage
- Empirical covariance matrix shrinks toward:
  ```
  Ω̃_f ≈ [Ω_f    0   ]
          [0  (b'Ω_ε⁻¹b)⁻¹]
  ```
- New factor variance ≈ (b'Ω_ε⁻¹b)⁻¹

**Key Papers:**
- Markowitz (1952): Portfolio Selection
- Grinold & Kahn (1999): The Fundamental Law of Active Management  
- Almgren & Chriss (2005): Optimal Execution
- Boyd et al. (2016): Multi-period Transaction Cost Management

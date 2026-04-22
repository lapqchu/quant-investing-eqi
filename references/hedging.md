# Hedging Reference

## Overview

Hedging is the process of reducing portfolio risk by augmenting positions with investments negatively correlated to existing holdings. This reference covers hedging methods from Chapter 12, including simple hedging, factor-based hedging, parameter uncertainty handling, and time-series beta hedging.

---

## 1. Hedging Definition and Types

### Core Concept

**Hedging** = augmenting portfolio with additional investments whose returns are negatively correlated to existing portfolio, reducing overall risk.

### Common Forms

1. **Market Hedging**: Using market index futures or ETFs (e.g., SPY, S&P 500 e-mini)
2. **Currency Hedging**: Hedging foreign exchange exposure
3. **Factor-Mimicking Portfolio (FMP) Hedging**: Using FMPs from fundamental factor models
4. **Tradeable Asset Hedging**: Using futures or liquid ETFs capturing specific risks
   - Energy/commodity futures
   - Sector ETFs (e.g., XLK for technology)
   - Style ETFs (e.g., MTUM for momentum)
5. **Thematic Basket Hedging**: Using bank-created baskets (political risk, industry trends)

---

## 2. Simple Single-Asset Hedging (Toy Story)

### Setup

**Two decision dates** (t₀, t₁) with one return realization between them.

**Two assets:**
- Core portfolio: expected return μ_c ≠ 0, volatility σ_c, position x_c
- Hedge asset: expected return μ_h ≈ 0, volatility σ_h, correlation ρ_{c,h}

**Problem:** Maximize Sharpe ratio of combined portfolio.

### Optimal Hedge Ratio

**MVO Optimization:**
```
max_{x_h} (μ_c x_c + μ_h x_h) - (λ/2)[σ_c²x_c² + σ_h²x_h² + 2ρ_{c,h}σ_cσ_h x_c x_h]
```

**Solution:**
```
x_h* = -ρ_{c,h} × σ_c × x_c / σ_h

Hedge Ratio (Equation 12.2):
|x_h*| / x_c = -ρ_{c,h} × σ_c / σ_h = -β(r_c, r_h)
```

The optimal hedge ratio equals the beta of core portfolio returns to hedge returns.

### Model-Based Beta Calculation

**Equation 12.4:**
```
β(r_c, r_h) = w_c'Ωw_h / (w_h'Ωw_h)

where w_c, w_h: portfolios representing core and hedge
      Ω: return covariance matrix (from factor model)
```

### Risk Reduction from Hedging

**Unhedged Variance:**
```
Var(r_c) = σ_c²x_c²
```

**Hedged Variance:**
```
Var(r_c + r_h × x_h*) = (1 - ρ_{c,h}²) × σ_c² × x_c²
```

**Sharpe Ratio Improvement (Equation 12.3):**
```
SR(hedged) / SR(native) = 1 / √(1 - ρ_{c,h}²)
```

Example: ρ = 0.8 → improvement factor = 1.79×

### Procedure 12.1: Simple Single-Asset Hedging

**Inputs:**
- Core portfolio: NMV x_c, returns r_c
- Hedge asset: returns r_h
- Beta estimate: β(r_h, r_c) [from time-series regression or factor model]

**Output:**
```
x_h* = -β(r_h, r_c) × x_c
```

### Implicit Assumptions (Often Violated)

1. Beta can be estimated accurately
2. Single trading instrument available
3. Trading costs negligible
4. Hedging instrument has zero expected return

---

## 3. Factor Hedging

### Motivation

When core portfolio is exogenous (e.g., aggregate of independent discretionary managers), reduce unwanted systematic factor exposures.

### Simple Procedure 12.2: Factor Hedging

1. **Compute Core Exposure:**
```
b_c = B'w_c

where b_c: factor exposures of core portfolio
      B: factor loadings matrix
      w_c: core portfolio weights
```

2. **Create Hedge Portfolio:**
```
w_h = -P b_c

where P: matrix of factor-mimicking portfolios
```

Result: Combined portfolio has zero factor exposure.

### Problem with Simple Procedure

Ignores:
- Non-zero factor expected returns
- Trading/execution costs
- Idiosyncratic variance added by hedge portfolio

### Full Factor Hedging Problem

**Equation 12.5 (General Case):**
```
max  α_⊥'(w_c + w_h) + μ'b - (1/(2ρ))[σ_fac² + σ_idio²] - f(w_h - w_h,0)

s.t. b = B'(w_c + w_h)
     σ_fac² = b'Ω_f b
     σ_idio² = (w_c + w_h)'Ω_ε(w_c + w_h)

where:
- α_⊥: alpha orthogonal (to factors)
- μ: expected factor returns
- f(·): trading cost function
- Ω_f: factor covariance matrix
- Ω_ε: idiosyncratic covariance matrix
```

### Key Insight from Exercise 12.2

**Optimal hedge policy ≠ perfect neutralization**

When:
- Factor portfolios have zero expected returns
- Hedge using only factor portfolios
- No transaction costs

Optimal hedging is:
```
x* = -[I_m + (b'Ω_ε⁻¹b)Ω_f]⁻¹ b_c
```

This is LESS than perfect neutralization (-b_c) because:
- Adding idiosyncratic risk of hedge portfolio
- Risk-aversion parameter ρ prevents over-hedging
- Trade-off between factor and idiosyncratic risk

---

## 4. Hedging with Time-Series Betas: Parameter Uncertainty

### Problem Setup

Want to hedge portfolio exposure to tradeable instrument (e.g., commodity futures) using time-series betas.

**Estimated Betas with Errors:**
```
β̂_i = β_i + η_i

where β̂_i: estimated beta for asset i
      β_i: true beta
      η_i: estimation error
      E[η_i²] = τ_i²
```

### Naive Hedge Failure

Using β̂ directly can increase portfolio risk if estimation errors large:

**Hedged Portfolio Variance:**
```
E[Var(r'w + r_h × x_h*)] = w'Ωw - (β'w)² × σ_h² + w'Ω_η w × σ_h²

Variance increases when: w'Ω_η w > (β'w)²
```

Left side: squared estimation error of portfolio beta
Right side: portfolio beta-related variance

### Optimal Fractional Hedging with Shrinkage

**Equation 12.6 - Shrinkage Factor:**
```
y_h* = 1 - (w'Ω_η w) / (β̂'w)²
```

**Equation 12.7 - Optimal Hedge Ratio:**
```
x_h* = -y_h* × β̂'w = -[β̂'w - (w'Ω_η w)/(β̂'w)]
```

### Sense-Checking Shrinkage Formula

**Property 1: Scale Invariance**
- Shrinkage factor independent of portfolio size
- If hedge 10× portfolio, use 10× hedge position

**Property 2: Error-Free Case (Ω_η = 0)**
```
y_h* = 1  →  use full optimal hedge ratio
```

**Property 3: True Beta = 0 (Edge Case)**
```
β_i = 0 for all i  →  β̂ = η
y_h* ≈ 0  →  don't hedge (correct!)
```

**Property 4: Signal-to-Noise Interpretation**
```
Numerator: w'Ω_η w ≈ Σ_i w_i² τ_i² (aggregated noise)
Denominator: (β̂'w)² (signal strength)
Ratio: noise-to-signal²
```

### Practical Implementation

**Step 1:** Estimate time-series betas with standard errors τᵢ

**Step 2:** Define diagonal error covariance matrix
```
Ω_η = diag(τ₁², τ₂², ..., τₙ²)
```

**Step 3:** Compute shrinkage factor
```
y_h* = 1 - (Σ_i wᵢ² τᵢ²) / (Σ_i β̂ᵢ wᵢ)²
```

**Step 4:** Buy hedge position
```
x_h* = -max(0, y_h* × β̂'w)

(Lower bound at zero avoids opposite direction hedging)
```

**Optional Step:** Account for correlated estimation errors
```
Ω_η = τ₁² ... 0    ×  1    ρ   ...ρ    ×  τ₁² ... 0
      0  ... τₙ²       ρ    1   ...ρ       0  ... τₙ²
                       ⋮    ⋮    ⋱  ⋮
                       ρ    ρ   ... 1
```

Test sensitivity across different ρ values.

### Simplified Formula (Equation 12.8)

For portfolios where position weights uncorrelated with beta standard errors:

```
y_h* = 1 - E[τ²] × |w|² / (β̂'w)²

where E[τ²] = (1/n) Σ_i τᵢ²
      |w|² = Σ_i wᵢ² (concentration measure)
```

**Further Simplification (Identical Betas):**
```
y_h* = 1 - β̂⁻² E[τ²] × H(w)

where H(w) = |w|²² / |w|₁² (Herfindahl concentration index)
```

For maximum diversification: H = 1/n
For single position: H = 1

**Key Insight:** More concentrated portfolios require more hedging shrinkage.

---

## 5. Factor-Mimicking Portfolios for Time Series

### Problem

Want to construct tradeable portfolio that tracks non-tradeable time series.

**Applications:**
1. Hedge portfolio exposure to non-tradeable macroeconomic theme
2. Create investable tracking portfolio for economic indicator
3. Verify whether time-series is economically meaningful

### Setup

**Minimize tracking error:**
```
min_{w} E[(r'w - r_h)²]

where r: n-asset returns
      w: portfolio weights (decision variable)
      r_h: non-tradeable time series
```

### Solution with Beta Estimation Error

**Optimal Portfolio (with derivation via Woodbury Lemma):**

In presence of estimation errors β̂ᵢ = βᵢ + ηᵢ:

```
w* = (μ_h² + σ_h²) × [Ω_r + μ_h² Γ + μ_h² β β']⁻¹ β

where Γ = diag(τ₁², τ₂², ..., τₙ²): estimation error covariance
      μ_h, σ_h: mean and volatility of target time series
```

Via Woodbury matrix inversion lemma:

```
w* = (μ_h² + σ_h²) × {[I - (Ω_r + μ_h² Γ)⁻¹ β β' / 
                              (μ_h⁻² + β'(Ω_r + μ_h² Γ)⁻¹ β)] 
      × (Ω_r + μ_h² Γ)⁻¹ β}
```

### Interpretation

**Role of Estimation Error:**
- Γ acts as regularizer on covariance matrix
- Larger E[μ_h] → more important regularization
- Prevents overfitting to estimation noise

**Limiting Cases:**

1. **No estimation error (Γ = 0), zero target return (μ_h = 0):**
```
w* ∝ σ_h² Ω_r⁻¹ β  ≈  minimum-variance portfolio (scaled)
```

2. **Large target return (|μ_h| → ∞):**
```
w* → Γ⁻¹ β
```
Emphasis on assets with lower estimation error.

### Key Properties

1. **Beta estimation error serves as regularizer** for covariance matrix
2. **Solution balances tracking accuracy with robustness** to parameter uncertainty
3. **Once optimal w* found**, can apply Equation 12.4 to hedge using this portfolio

---

## 6. Hedging Methodology Comparison

### FMP-Based Hedging

**Advantages:**
- Removes specific factor risk
- Clear factor interpretation
- Can hedge multiple factors simultaneously

**Disadvantages:**
- Requires factor model
- FMP may have high idiosyncratic variance
- Hedging costs reflected in alpha, not model

### Tradeable Asset Hedging (Futures/ETFs)

**Advantages:**
- Liquid, low trading costs
- Available for broad market/sector/style risks
- Time-series betas easier to estimate

**Disadvantages:**
- Beta estimation error can worsen hedging
- Shrinkage factor must be applied
- May not hedge all systematic factors

### Complete Examples from Chapter

**Example 12.1: Market FMP vs Benchmark**
- Compare factor variance reduction using FMP vs S&P 500 benchmark
- Benchmark has unwanted exposures to non-market factors
- FMP pure market exposure, but higher idiosyncratic risk

**Exercise 12.2: Factor Hedging Mathematics**
- Shows optimal hedging size < perfect neutralization
- Idiosyncratic variance of FMPs limits hedging aggression

**Exercise 12.3: Single-Factor Optimization**
- With one factor and one hedge portfolio
- Proves it's suboptimal to fully hedge single-factor exposure
- Trade-off between systematic risk reduction and idiosyncratic risk increase

---

## 7. Rehedging and Transaction Costs

### Rehedging Frequency Trade-off

**Issue:** Over-rehedging increases transaction costs

**Solution Approaches:**
1. **Time-Based**: Rehedge daily/weekly/monthly
2. **Threshold-Based**: Rehedge when exposure drift exceeds limit
3. **Cost-Aware Optimization**: Multi-period problem with transaction costs

### Procedure 12.1 with Transaction Costs

**Extended Problem (Equation 12.5 generalized):**
```
max  α_⊥'(w_c + w_h) + μ'b - (1/(2ρ))[σ_fac² + σ_idio²] 
     - f(w_h - w_h,0)  [transaction costs]

where f: quadratic or square-root market impact function
```

Can be reformulated as multi-period problem (Exercise 12.4 hint) and solved with Procedure 11.3 from Chapter 11.

---

## 8. Summary of Hedging Approaches

| Method | Best For | Assumptions | Trade-offs |
|--------|----------|-------------|-----------|
| Simple Beta Hedge | Market/single risk | β accurately estimated | Zero expected return |
| Factor Hedging | Factor exposure reduction | Factor model accurate | Idiosyncratic variance |
| Time-Series Beta | Non-equity risks (commodities) | Limited parameter uncertainty | Shrinkage reduces benefit |
| FMP Hedging | Multi-factor risk | Pure factors available | Implementation cost |
| Fractional/Shrinkage | Parameter uncertainty | Can estimate standard errors | Conservative approach |

---

## Key Formulas Summary

| Concept | Formula | Reference |
|---------|---------|-----------|
| Optimal hedge ratio | \|x_h*\| / x_c = β(r_c, r_h) | Eq. 12.2 |
| SR improvement | 1 / √(1 - ρ²) | Eq. 12.3 |
| Factor exposure | b_c = B'w_c | Ch. 12 |
| Beta from model | β = (w_c'Ωw_h) / (w_h'Ωw_h) | Eq. 12.4 |
| Shrinkage factor | y_h* = 1 - (w'Ω_η w) / (β̂'w)² | Eq. 12.6 |
| Shrink hedge ratio | x_h* = -y_h* × β̂'w | Eq. 12.7 |
| Simplified shrinkage | y_h* = 1 - E[τ²] × \|w\|² / (β̂'w)² | Eq. 12.8 |

---

## References and Further Reading

**Core Papers:**
- Chapter 12: Hedging (Paleologo)
- Optimal execution literature (Chapter 11 appendix)

**Key Takeaways:**
1. Hedging reduces portfolio risk but comes with costs
2. Simple beta hedging works when beta estimated accurately
3. Parameter uncertainty requires hedging shrinkage
4. Factor hedging reduces systematic risk but adds idiosyncratic risk
5. Rehedging frequency must balance risk control vs transaction costs
6. Fractional hedging often optimal given real-world constraints

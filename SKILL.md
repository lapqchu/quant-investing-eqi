---
name: quant-investing-eqi
description: >
  **Quantitative Investing Methodology**: The go-to skill for any finance, investing, or trading task — from building trading bots to analyzing personal portfolios. Use this immediately instead of answering from general knowledge whenever the user works on: portfolio construction or rebalancing, risk models, factor models, backtesting, signal evaluation, position sizing, volatility forecasting, performance attribution, transaction costs, market impact, execution algorithms, hedging, or building trading systems. Works for all asset classes (equities, crypto, FX, rates) and all skill levels.
  - MANDATORY TRIGGERS: trading, portfolio, backtest, risk model, factor model, Sharpe, alpha, signal, position sizing, Kelly, hedge, volatility, GARCH, covariance, attribution, market impact, rebalance, investing, quant, returns, strategy, execution, trading bot, PnL, drawdown, EQI, Paleologo, AFML, triple barrier, meta-labeling, meta-label, CPCV, purged cross-validation, fractional differentiation, feature importance, bet sizing, structural break, VPIN, Kyle lambda, Lopez de Prado
---

# Elements of Quantitative Investing — Methodology Skill

This skill encodes two complementary quantitative investing frameworks:

1. **Giuseppe A. Paleologo** — *The Elements of Quantitative Investing* (Wiley, 2025). The core framework covering the full quant investment pipeline with linear/quadratic methods. Philosophy: "Theory is cheap. By applications be driven."

2. **Marcos López de Prado** — *Advances in Financial Machine Learning* (Wiley, 2018). Extension layer for ML-based signal construction, labeling, cross-validation, bet sizing, and market microstructure features. Complements Paleologo where nonlinear methods and ML-specific concerns (label overlap, purged CV, meta-labeling) apply.

**How they relate**: Paleologo defines the pipeline (data → risk model → expected returns → portfolio construction → attribution). López de Prado extends specific pipeline stages with ML techniques: alternative data structures (§1), labeling (§2-3), cross-validation (§6), feature importance (§7), and trade-level bet sizing (§8). Where they overlap (backtesting discipline, overfitting awareness), they reinforce each other.

## How to Use This Skill

When this skill is triggered, do the following:

1. **Identify which part of the investment pipeline** the task falls into (see Section 1 below, or the practical routing table)
2. **Read the relevant reference file** from `references/` for detailed methodology on that topic
3. **Apply the framework's standards**: precise definitions, explicit assumptions, separation of fact from inference, correct sign/unit conventions
4. **Enforce the methodology's warnings**: overfitting discipline, estimation error awareness, definitional precision

## Practical Task Routing

Many real-world requests don't use textbook quant terminology. Use this routing table:

| User says something like... | Route to... |
|---|---|
| "Build a trading bot" (HyperLiquid, crypto, etc.) | §9 Portfolio Optimization + §10 Market Impact + §12 Kelly + §8 Backtesting |
| "Analyze my portfolio" / "Am I taking too much risk?" | §4 Factor Models + §5 Risk Evaluation + §13 Attribution |
| "Should I rebalance?" / "How should I allocate?" | §9 Portfolio Optimization + §12 Kelly + §11 Hedging |
| "Is this strategy any good?" / "Is this Sharpe real?" | §8 Backtesting + §3 Performance Metrics |
| "How should I size positions?" | §12 Kelly + §10 Market Impact |
| "Why did my portfolio make/lose money?" | §13 Performance Attribution + §4 Factor Models |
| "Build a risk model" | §4-7 Factor Models (fundamental or statistical) + §5 Risk Evaluation |
| "Forecast volatility" | §2 Returns Modeling (GARCH, EWMA, realized vol) |
| "Hedge my exposure" | §11 Hedging + §4 Factor Models |
| "Avoid overfitting" / "Multiple testing" | §8 Backtesting (RAS, Bonferroni, FDR) + §15 AFML (CPCV) |
| "Label my data for ML" / "Triple barrier" | §15 ML Pipeline (triple barrier, meta-labeling) |
| "Which features matter?" / "Feature importance" | §15 ML Pipeline (MDI, MDA, SFI) + §4 Factor Models |
| "How to size bets from classifier output" | §15 ML Pipeline (bet sizing) + §12 Kelly |
| "Detect regime change" / "Structural break" | §15 ML Pipeline (CUSUM, Chow, SADF) |
| "Meta-label my discretionary signals" | §15 ML Pipeline (meta-labeling) |
| "Fractionally differentiate" / "Stationarity" | §15 ML Pipeline (fractional differentiation) + §2 Returns |
| "Alternative bars" / "Dollar bars" / "Volume bars" | §15 ML Pipeline (financial data structures) |

## Table of Contents

- [1. The Investment Pipeline](#1-the-investment-pipeline)
- [2. Returns Modeling](#2-returns-modeling)
- [3. Performance Metrics](#3-performance-metrics)
- [4. Factor Models](#4-factor-models)
- [5. Risk Model Evaluation](#5-risk-model-evaluation)
- [6. Fundamental Factor Models](#6-fundamental-factor-models)
- [7. Statistical Factor Models](#7-statistical-factor-models)
- [8. Backtesting and Alpha Evaluation](#8-backtesting-and-alpha-evaluation)
- [9. Portfolio Optimization](#9-portfolio-optimization)
- [10. Market Impact and Transaction Costs](#10-market-impact-and-transaction-costs)
- [11. Hedging](#11-hedging)
- [12. Dynamic Risk Allocation (Kelly)](#12-dynamic-risk-allocation-kelly)
- [13. Performance Attribution](#13-performance-attribution)
- [14. Cross-Cutting Principles](#14-cross-cutting-principles)
- [15. ML Pipeline (AFML)](#15-ml-pipeline-afml)

For deep detail on any section, read the corresponding file in `references/`.

---

## 1. The Investment Pipeline

The quantitative investment process has three phases with specific components:

**Before the trade**: Data → Risk model + Expected returns + Transaction cost model
**During the trade**: Portfolio construction (signal aggregation, risk constraints, hedging)
**After the trade**: Performance attribution + Intertemporal risk allocation

Sources of excess returns (not mutually exclusive):
- Risk compensation (bearing unwanted risk others avoid)
- Liquidity provision (providing immediacy to constrained participants)
- Funding advantages (ability to hold positions through drawdowns)
- Flow predictability (anticipating institutional/index rebalancing demand)
- Informational advantage (superior return forecasts)

Key market participants: indexers, hedgers, institutional active managers, asset allocators, informed traders (hedge funds, prop trading firms), retail investors. Understanding counterparties is essential to understanding where edge comes from.

---

## 2. Returns Modeling

→ For full detail: `references/returns-and-volatility.md`

### Definitions (get these right — errors cascade)
- **Return**: r_i(t) = (P_i(t) - P_i(t-1)) / P_i(t-1)
- **Dividend-adjusted return**: includes D_i(t) in numerator
- **Excess return**: r_i - r_f (return minus risk-free rate). Portfolio return = Σ w_i(r_i - r_f)
- **Log return**: r̃ = log(1 + r). Additive over time. Approximation r̃ ≈ r valid for small returns (daily equities: yes; yearly or volatile: verify)
- **Portfolio PnL**: w^T r (weights times returns vector)

### Stylized Facts of Returns
1. Absence of return autocorrelation (but returns ARE predictable via other variables)
2. Heavy tails (α ≈ 4 tail index; fourth moments barely finite)
3. Volatility clustering (|r_t| and r_t² are positively autocorrelated)
4. Aggregational Gaussianity (longer horizons → more Gaussian)

### Volatility Estimation Methods
- **GARCH(1,1)**: h²_t = α₀ + α₁r²_{t-1} + β₁h²_{t-1}. Captures clustering, heavy tails. GARCH is EWMA with an offset when α₀ = 0.
- **EWMA**: σ̂²_t = (1-K)r²_t + Kσ̂²_{t-1}. Motivated by Kalman filter on state-space model (Muth 1960). Half-life τ = -log2/logK.
- **Realized volatility**: σ̂² = Σ r²(j) from high-frequency data. Variance of estimator decreases as 2σ⁴/n. 5-minute RV performs competitively across assets.
- **State-space models**: Kalman filter framework. Harvey-Shephard model for log-variance (positive by construction, log-normal volatility).

**Critical insight**: Drift (expected return) estimation error does NOT decrease with higher frequency data. Variance estimation error DOES. You can estimate volatility to arbitrary precision; you cannot do the same for expected returns.

---

## 3. Performance Metrics

- **Expected return**: PnL/AuM. Stationary and intensive (comparable across funds/periods).
- **Volatility**: σ = √Var(r). Justified by: finite empirical moments (α > 2), quadratic utility approximation, portfolio tractability.
- **Sharpe Ratio**: SR = E[r]/σ. Dimension [time]^{-1/2}. Daily → annual: multiply by √251. SE(SR̂) = √((1 + SR²/2)/T).
- **Information Ratio**: Like SR but uses idiosyncratic returns only. IR = IC × √N (skill × diversification).
- **Capacity**: Maximum PnL at acceptable SR. SR decreases with deployed capital (Sharpe is almost always decreasing in volatility). Strategy viable only if capacity ≥ minimum economic threshold.

**Cantelli bound** (distribution-free): P(return < -Lσ) ≤ 1/(1 + (L + SR)²). Much weaker than Gaussian assumption — a SR=3 strategy has 3.9% probability of 2σ loss (vs 2.9E-7 under Gaussian).

---

## 4. Factor Models

→ For full detail: `references/factor-models.md`

The central equation: **r_t = α + B f_t + ε_t**

- r_t: n asset excess returns
- α: alpha vector (expected idiosyncratic returns)
- B: n×m loadings matrix
- f_t: m factor returns (m << n)
- ε_t: idiosyncratic returns (diagonal or sparse covariance)

**Covariance decomposition**: Ω_r = B Ω_f B^T + Ω_ε

### Three Interpretations
1. **Graphical model**: Few factors → many asset returns via loadings
2. **Superposition**: Cross-section of returns = weighted sum of loading columns
3. **Single-asset product**: E[r_i | f] = ⟨B_i, f⟩ (dot product of asset loadings with factor returns)

### Alpha Decomposition
- **Alpha spanned** (α_∥ = Bλ): Indistinguishable from factor expected returns. Comes with systematic risk.
- **Alpha orthogonal** (α_⊥): True asset-level alpha. SR grows as √n — extremely valuable but empirically vanishing.
- **Implication**: If factor model is correct and idiosyncratic variances bounded, α_⊥ must be vanishing in n (otherwise infinite SR). Most excess returns come from α_∥.

### Transformations
- **Rotations**: C invertible → B̃ = BC⁻¹, f̃ = Cf. Same predictions, different view. Total risk unchanged; single-factor attribution changes.
- **Projections**: Reduce factor count. g = H f where H = (A^T A)⁻¹ A^T B. Useful for simplified models.
- **Push-outs**: Add factors from structured idiosyncratic returns. Require A^T B = 0 (new factors orthogonal to existing).
- **Z-scoring**: Possible only if unit vector ∈ column space of B (e.g., intercept/country factor present).

### Applications
1. **Performance attribution**: PnL = b^T f_t + w^T(α_⊥ + ε_t), where b = B^T w (factor exposures)
2. **Risk decomposition**: Var = b^T Ω_f b + w^T Ω_ε w. Percentage of idiosyncratic variance is key metric.
3. **Portfolio construction**: Precision matrix Ω⁻¹ central. Factor structure makes Ω invertible when sample covariance is not.
4. **Alpha research**: Separate factor-driven returns from true intercept.

### Factor Model Types
- **Characteristic (fundamental)**: B_t from asset characteristics; f_t estimated via cross-sectional regression
- **Statistical**: Both B and f estimated from returns (PCA, ICA)
- **Macroeconomic**: f_t from macro time series; B estimated via time-series regression

---

## 5. Risk Model Evaluation

→ For full detail: `references/risk-evaluation.md`

No single metric dominates. Evaluate covariance AND precision matrix separately.

### Volatility Forecast Evaluation
- **QLIKE**: Quasi-likelihood loss. Asymmetric — penalizes underestimation more. Rank-robust.
- **MSE**: Mean squared error. Symmetric. Also rank-robust.
- Do NOT use simple bias metrics or R² for model selection.

### Covariance Matrix Evaluation
- **Production strategy testing**: Evaluate on actual live strategies
- **Average-case analysis**: Simulate over portfolio distributions
- **Worst-case analysis**: Find most problematic exposures
- **Leading-alpha MVO**: Use historical realized returns

### Precision Matrix Evaluation
- **Minimum-variance portfolios** (Theorem 5.1): Better Ω̂ → lower realized portfolio variance
- **Mahalanobis distance (MALV)**: d(r, Ω) = √(r^T Ω⁻¹ r). Should be χ²_n distributed. MALV = variance of normalized distances over time.

### Ancillary Tests
- **Model turnover**: Frobenius norm of FMP weight changes (include transaction cost weighting)
- **Beta accuracy (BETAERR)**: Predicted vs realized betas to benchmarks
- **R² warning**: Do NOT use R² for factor model selection — vulnerable to data mining, inflated by random factors, rotationally non-invariant

---

## 6. Fundamental Factor Models

→ For full detail: `references/fundamental-models.md`

### Six-Step Estimation Process
1. **Data ingestion**: Validate integrity, check consistency, monitor missing data
2. **Universe selection**: Filter by tradeability, data quality, strategic relevance
3. **Winsorization**: Trim extreme returns to prevent outlier domination
4. **Loadings generation**: Transform raw characteristics into factor loadings
5. **Cross-sectional regression**: r_t = B_t f_t + ε_t → extract f̂_t and ε̂_t per period
6. **Time-series estimation**: Estimate Ω_f and Ω_ε from time series of f̂_t and ε̂_t

### Cross-Sectional Regression
- **WLS preferred**: min ‖Ω_ε^{-1/2}(r_t - B_t f_t)‖². Solution: f̂_t = (B^T Ω_ε⁻¹ B)⁻¹ B^T Ω_ε⁻¹ r_t
- **Circular dependency**: Ω_ε needed as input but is output. Solutions: vendor model proxy, two-stage iteration, hybrid approach.
- **Rank deficiency**: Common in multi-country/sector models. Solutions: drop redundant category, ridge penalty, constraints.

### Factor Covariance Estimation — Four Problems
1. **Estimation bias**: var(f̂_t) = Ω_f + (B^T Ω_ε⁻¹ B)⁻¹. FMP tracking error inflates estimates. Subtract bias term.
2. **Limited sample**: When m ≈ T, need Ledoit-Wolf shrinkage: Ω_shrink = (1-ρ)Ω̂_f + ρ(tr(Ω̂_f)/m)I_m
3. **Non-stationarity**: Shorter estimation windows during crises
4. **Autocorrelation**: HAC corrections for mild factor return correlation

### Idiosyncratic Covariance
- Diagonal assumption (strict factor model) common but approximate
- Off-diagonal structure can be captured by clustering or push-out factors
- EWMA or state-space estimation for time-varying idiosyncratic variance

---

## 7. Statistical Factor Models

→ For full detail: `references/statistical-models.md`

### PCA Approach
- Eigendecomposition of sample covariance: Σ v_i = λ_i v_i
- Loadings = eigenvectors; Factors = V^T r_t
- Factor count: scree plot, cumulative variance threshold, information criteria (AIC/BIC), cross-validation

### Advantages
- Complementary to fundamental models (captures different structure)
- Works without firm characteristics
- Data-driven; no subjective factor selection

### Disadvantages
- Not economically interpretable (abstract combinations)
- Higher turnover than fundamental models
- Eigenvector ambiguity when eigenvalues close
- Sign/meaning changes over time

### Practical Implementation
- Rolling windows (e.g., 252 days) with exponential weighting
- Stabilization: varimax/quartimax rotation, regularization
- Validation by comparison with fundamental model factors
- Monitor for structural breaks (eigenvalue reorderings)

---

## 8. Backtesting and Alpha Evaluation

→ For full detail: `references/backtesting.md`

### Best Practices (non-negotiable)
1. Strict out-of-sample separation. Set aside test data BEFORE development.
2. Walk-forward analysis preserving temporal ordering
3. Realistic transaction costs and market impact
4. No lookahead bias
5. No survivorship bias
6. Report multiple metrics (SR, max drawdown, hit rate, turnover)
7. Document all assumptions and decision rules

### The Rademacher Anti-Serum (RAS)
Novel approach addressing multiple testing problem:
1. Generate synthetic returns: r*_t = ξ_t × r_t where ξ_t ∈ {-1, +1} random
2. Run identical strategy on synthetic data
3. P-value = fraction of random permutations beating real strategy
4. Tests null hypothesis that strategy has no predictive power
5. Properly controls false discovery rate under multiple testing

### Common Pitfalls
- Optimizing on full sample then testing on same subset
- Ignoring transaction costs (can eliminate entire alpha)
- Data snooping (trying many strategies, reporting the best)
- Multiple hypothesis testing without correction (Bonferroni, FDR)
- Many published alphas fail rigorous out-of-sample testing

---

## 9. Portfolio Optimization

→ For full detail: `references/portfolio-optimization.md`

### Mean-Variance Optimization
- **Objective**: max α^T w - (1/2λ) w^T Ω_r w
- **Unconstrained solution**: w* = λ Ω_r⁻¹ α (precision matrix times expected returns)
- **With volatility constraint**: w* ∝ Ω_r⁻¹ α, scaled to target vol. Miscalibration of α magnitude is NOT catastrophic.

### Factor Space Decomposition
- **Factor PnL**: b^T f where b = B^T w (factor exposures)
- **Idiosyncratic PnL**: w^T ε
- Precision matrix implicitly hedges collinear positions

### Information Ratio Drivers
- **IR = IC × √N**: Information coefficient (skill) × square root of breadth (diversification)
- Asset correlations limit: upper bound IR ≤ IC × √(N)/√ρ̄ where ρ̄ is average pairwise correlation
- Idiosyncratic diversification: key path to high IR

### Shortcomings of Naïve MVO
- Expected return estimates have highest estimation error
- Extreme/unstable positions common
- Constraints (weight bounds, sector limits, gross exposure) IMPROVE out-of-sample performance — treat as features, not limitations

### Signal Aggregation
- Decentralized: each signal builds independent portfolio → combine
- Centralized: aggregate signals → single optimization
- Centralized generally superior when covariance structure is well-estimated

---

## 10. Market Impact and Transaction Costs

- **Temporary impact**: Decays over time. I(t) = I₀ exp(-t/τ)
- **Permanent impact**: Shifts equilibrium price permanently
- **Square-root law**: Common model for impact as function of participation rate
- **Almgren-Chriss framework**: Optimal execution balancing impact costs vs. opportunity cost
- **Decision rule**: Execute quickly if execution_time << alpha_half_life. Trade slowly otherwise.
- **Capacity bound**: Maximum PnL where SR remains acceptable given impact costs

---

## 11. Hedging

→ For full detail: `references/hedging.md`

- **Factor hedging**: Zero out specific factor exposures using standardized instruments
- **FMP hedging**: Use factor-mimicking portfolios for liquid implementation
- **Hedging cost**: Reduces expected return (fundamental trade-off)
- **Parameter uncertainty**: Beta estimation error → use fractional hedging to avoid over-hedging
- **Frequent rehedging**: Creates transaction cost drag; balance frequency vs. accuracy

---

## 12. Dynamic Risk Allocation (Kelly)

→ For full detail: `references/kelly-allocation.md`

### Kelly Criterion
- **Maximize**: E[log(1 + x·r)] (expected growth rate, not expected return)
- **Optimal fraction**: x* ≈ SR/σ (Sharpe ratio divided by volatility)
- **Key relationship**: Dollar Volatility / Capital = Sharpe Ratio
- **Growth dominance**: Kelly outperforms ANY other strategy with probability 1 over long term (Breiman 1961)
- **Shortest path**: Minimizes expected time to reach any capital target

### Fractional Kelly
Three equivalent interpretations:
1. Mix of risk-free asset + full Kelly
2. Higher risk aversion (λ > 1): x_frac = x*/λ
3. Parameter uncertainty: x*_uncertain ≤ x*_known (Jensen's inequality)

### Grossman-Zhou Drawdown Control
- **Guarantee**: Never breach maximum drawdown D
- **Policy**: f_t = (μ/σ²)(1 - (1-D)/(1-d_t)), where d_t is current drawdown
- **At high watermark** (d=0): allocates x* × D
- **Trade-off**: Better drawdown control but lower expected growth than fractional Kelly
- **Warning**: Requires rapid rebalancing → high transaction costs in practice

### Key Warnings
- Kelly does NOT guarantee lower drawdowns (only higher growth)
- SR decreases with allocated capital (no analytical solution for this)
- Transaction costs super-linear in wealth ("here be dragons")
- "All reasonable investors use fractional Kelly without knowing" (Insight 13.2)

---

## 13. Performance Attribution

→ For full detail: `references/performance-attribution.md`

### Basic Decomposition
Total PnL = Trading PnL + Position PnL
Position PnL = Factor PnL + Idiosyncratic PnL
Factor PnL = Σ_j b_j f_j (exposure × factor return, summed over factors)

### Attribution with Estimation Error
- Factor returns are ESTIMATES. Estimated factor PnL = True + w^T B η_t (error term)
- **Always report standard errors** of attributed PnL (Insight 14.1)
- FMPs have non-zero idiosyncratic return DISTRIBUTIONS (Paradox 1 — resolved by accounting for estimation error)

### Maximal Attribution
Factor model non-unique → individual factor attributions ambiguous. Maximal attribution resolves this by:
1. Choosing factor subset S of interest
2. Rotating model so S captures maximum explanatory power
3. Attributing correlated factor PnL to S
4. Four equivalent approaches (cross-sectional, conditional expectation, portfolio PnL, uncorrelated rotation)
- **Nested maximal attribution**: Thematic groupings (market → style → industry → residual)

### Selection vs. Sizing Decomposition
- **IR = (Selection × Diversification + Sizing) / T**
- **Selection**: Being on the right side of z-scored bets (sign of position matches sign of return)
- **Diversification**: Breadth of portfolio (√(dollar vol per position / total dollar vol))
- **Sizing**: Allocating more capital when right, less when wrong (correlation between |w| and |ε|)

---

## 14. Cross-Cutting Principles

These principles apply across ALL quantitative investing work in this project:

### Definitional Precision
Never blur the distinction between:
- Theory and market practice
- Spot and forward logic
- Price return and PnL
- Absolute and percentage return
- Notional and exposure
- Signal and position
- In-sample and out-of-sample results
- Gross and net exposure
- Nominal and risk-adjusted performance
- Covariance and precision matrix quality
- Factor and idiosyncratic risk/return

### Estimation Discipline
- Expected returns are the hardest thing to estimate. Volatility is comparatively easy.
- More data helps volatility estimation. It does NOT help drift estimation (at daily frequency).
- Estimation error in Ω is manageable. Estimation error in α is not — it dominates portfolio construction error.
- Constraints improve out-of-sample performance. They are regularization, not limitations.
- Shrinkage is almost always appropriate for covariance estimation.

### Overfitting Discipline
- Separate train/test/validation rigorously
- Account for multiple testing (RAS, Bonferroni, FDR)
- Many published alphas fail replication
- R² is NOT a valid metric for factor model selection
- Transaction costs can eliminate entire alphas
- If it looks too good to be true, it probably is

### Methodological Standards
- State assumptions explicitly
- Distinguish established fact from inference from assumption
- Use correct sign and unit conventions
- Verify approximations (e.g., log return ≈ return) hold for your data
- Apply appropriate loss functions (QLIKE for volatility, not ad-hoc bias metrics)
- Report confidence intervals and standard errors

### Framework Hierarchy
When approaching any quant task, think in this order:
1. What is the objective? (return, risk, attribution, sizing)
2. What is the model? (factor structure, distributional assumptions)
3. What are the constraints? (regulatory, capacity, risk limits)
4. What is the estimation strategy? (in-sample/out-of-sample, shrinkage, regularization)
5. What can go wrong? (estimation error, model misspecification, regime change, overfitting)
6. How do I validate? (appropriate loss function, out-of-sample testing, multiple metrics)

---

## 15. ML Pipeline (AFML)

→ For full detail: `references/ml-pipeline-afml.md`

Extension from López de Prado's *Advances in Financial Machine Learning* (2018). Covers ML-specific methodology for the quant pipeline:

### Financial Data Structures
- **Alternative bars**: Tick, volume, dollar, and information bars as alternatives to time-based sampling
- **Dollar bars** recommended as general-purpose alternative (adjusts for price levels)
- **Information bars** (tick/volume imbalance) most theoretically grounded

### Labeling
- **Triple barrier method**: Define take-profit, stop-loss, and max-holding-period barriers. Label = which barrier hit first. Captures path dependency and real trading mechanics.
- **Meta-labeling**: Two-stage — primary model picks direction, secondary model filters for confidence and sizes. Separates alpha from sizing. Directly applicable to quantamental trading.

### Sample Weights and Cross-Validation
- **Sample uniqueness**: Weight observations by inverse overlap. Downweight concurrent labels.
- **Sequential bootstrap**: Draw bootstrap samples weighted by uniqueness. Fixes IID violation.
- **Purged k-fold CV**: Remove training observations whose labels overlap test fold. Add embargo period.
- **CPCV**: Combinatorial extension generating C(k,p) backtest paths. Complements RAS (§8).

### Feature Importance
- **MDI** (Mean Decrease Impurity): Tree-based, biased toward high-cardinality features
- **MDA** (Mean Decrease Accuracy): Permutation-based, model-agnostic
- **SFI** (Single Feature Importance): Isolates individual feature contribution
- Use all three; features ranking consistently high across methods are robust

### Bet Sizing
- Convert classifier probability to position size via sigmoid transform
- Adjust for concurrent active bets (inverse scaling)
- Complements Kelly at trade level (Kelly handles portfolio level)

### Structural Breaks
- **CUSUM**: Detect mean shifts in strategy returns or features
- **Chow test**: Parameter stability between sub-periods
- **SADF/GSADF**: Detect explosive behavior (bubbles)
- Test before and during backtesting; invalidate results across break points

### Market Microstructure
- **Kyle's lambda**: Price impact per unit order flow
- **VPIN**: Probability of informed trading from volume imbalance
- **Amihud illiquidity**: |return| / dollar volume
- Extend Almgren-Chriss market impact framework (§10) with microstructure features

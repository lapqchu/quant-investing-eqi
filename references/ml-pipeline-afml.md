# Machine Learning Pipeline for Finance (AFML Reference)

Based on Marcos López de Prado, *Advances in Financial Machine Learning* (Wiley, 2018). Integrated into the EQI skill as a complementary methodology layer for signal construction, labeling, cross-validation, and bet sizing.

**Relationship to Paleologo**: Paleologo's EQI framework covers the full quant investment pipeline with linear/quadratic methods. López de Prado extends three areas: (1) how to label and structure financial data for ML, (2) how to cross-validate without leakage in time-series settings, (3) how to size bets from ML classifier output. Where the two overlap (backtesting discipline, overfitting awareness), they reinforce each other. Where they diverge (AFML uses nonlinear classifiers; EQI uses linear factor models), treat them as complementary tools.

---

## 1. Financial Data Structures

### Bars: Alternative to Time-Based Sampling

Standard time bars (daily, hourly) sample at fixed intervals regardless of market activity. AFML proposes information-driven alternatives:

**Tick Bars**: Sample every N transactions. Advantages: more uniform information content per bar, better statistical properties (closer to IID), reduces oversampling of quiet periods.

**Volume Bars**: Sample every V units of volume traded. Rationale: volume correlates with information arrival (Kyle 1985). More stable variance than time bars.

**Dollar Bars**: Sample every $D of dollar volume. Adjusts automatically for price level changes — a $100 stock trading 1000 shares and a $10 stock trading 10000 shares generate similar bars.

**Information Bars**: Sample based on information arrival (trade imbalance, volume imbalance, or tick imbalance). Most theoretically grounded but hardest to implement.

- **Tick imbalance bars**: Aggregate trades until cumulative signed tick imbalance exceeds threshold E[T] × E[b_t], where b_t ∈ {-1, +1} is tick direction.
- **Volume imbalance bars**: Same logic weighted by volume.
- **Tick runs bars**: Monitor runs of same-sign ticks; bar completes when run length is unusual.

**Practical guidance**: Dollar bars are the best general-purpose alternative to time bars. Information bars are superior but require careful threshold calibration.

---

## 2. Labeling: The Triple Barrier Method

### Why Labeling Matters

Standard labeling (sign of next-period return) ignores:
- Path dependency (a trade that's +5% then -10% is different from one that's -10% directly)
- Stop-losses and take-profits (real trading has exit rules)
- Holding period variation (not all positions are held the same duration)

### Triple Barrier Setup

Define three barriers around entry price at time t_0:

1. **Upper barrier** (take-profit): First time price exceeds entry + profit_taking × σ
2. **Lower barrier** (stop-loss): First time price falls below entry - stop_loss × σ
3. **Vertical barrier** (max holding period): Time t_0 + max_hold

Where σ is a volatility estimate (e.g., EWMA of returns).

**Label assignment**:
- If upper barrier hit first → label = +1 (profitable trade)
- If lower barrier hit first → label = -1 (stopped out)
- If vertical barrier hit first → label = sign of return at expiry

**Advantages over fixed-horizon labeling**:
- Incorporates path dependency
- Matches real trading mechanics (stops and targets)
- Variable holding periods reflect actual signal decay
- Volatility-adjusted barriers adapt to regime changes

### Practical Implementation

```
For each signal at time t:
  1. Compute daily volatility σ_t (EWMA with span ~100 days)
  2. Set upper barrier = pt × σ_t (e.g., pt = 2)
  3. Set lower barrier = sl × σ_t (e.g., sl = 2)
  4. Set vertical barrier = t + max_hold (e.g., 5-20 days)
  5. Simulate forward path; record which barrier is hit first
  6. Assign label {-1, 0, +1}
```

---

## 3. Meta-Labeling

### Concept

Meta-labeling is a two-stage approach:

1. **Primary model**: Determines trade direction (long/short/no trade). Can be a quantitative signal, discretionary view, or rule-based system.
2. **Secondary (meta) model**: Determines whether to take the trade and how to size it. Trained on features that predict whether the primary model's signal will be profitable.

### Why Meta-Labeling Works

- **Separates alpha from sizing**: The hard problem (direction) is separated from the easier problem (is this a good time to trade?)
- **Reduces false positives**: Meta-model learns when the primary signal is unreliable
- **Improves F1 score**: Better precision-recall trade-off than a single monolithic classifier
- **Works with any primary model**: The primary model can be ML, rules-based, or discretionary

### Implementation

```
Stage 1: Primary model generates signals s_t ∈ {-1, 0, +1}
Stage 2: For each s_t ≠ 0:
  - Features: market regime, volatility, signal strength, liquidity, etc.
  - Label: Did the primary signal's trade hit take-profit before stop-loss?
  - Meta-model output: probability p ∈ [0, 1] that the trade will be profitable
  - Position size: proportional to p (or use bet sizing formula below)
```

### Connection to Quantamental Trading

Meta-labeling is directly applicable to quantamental workflows. The primary model is the discretionary view; the meta-model provides quantitative evidence for or against taking the trade. This is exactly the "quantitative backing for discretionary views" use case.

---

## 4. Sample Weights and Uniqueness

### The Problem

Financial labels overlap in time. A label at time t may depend on returns from t to t+5, while a label at t+1 depends on returns from t+1 to t+6. These overlapping labels are not independent — treating them as IID corrupts training.

### Uniqueness of Labels

Define **uniqueness** of observation t as:

$$\tilde{u}_t = \frac{1}{\sum_{s} \mathbb{1}[t \in \text{span}(s)]}$$

where span(s) is the time range of returns used to compute label s.

Average uniqueness across all concurrent labels:

$$u_t = \frac{1}{|\{s : t \in \text{span}(s)\}|} \sum_{s: t \in \text{span}(s)} \tilde{u}_s$$

**Use as sample weight**: Weight each observation by its average uniqueness in training. Highly overlapping observations get downweighted.

### Sequential Bootstrap

Standard bootstrap assumes IID — invalid for overlapping financial labels. Sequential bootstrap:

1. Draw first sample with uniform probability
2. For each subsequent draw, compute average uniqueness of candidate samples given already-selected samples
3. Draw with probability proportional to uniqueness
4. Repeat until bootstrap sample complete

This produces bootstrap samples with higher average uniqueness, improving statistical validity of bagged estimators.

---

## 5. Fractional Differentiation

### The Stationarity-Memory Trade-off

- **Raw prices**: Non-stationary (unit root) → cannot be used directly in ML models that assume stationarity
- **Returns (first difference)**: Stationary → but discards long-memory information (support/resistance levels, mean-reversion targets)
- **Fractional differentiation**: Compromise — makes series stationary while preserving maximum memory

### The Operator

The fractional difference operator of order d ∈ [0, 1]:

$$(1 - L)^d = \sum_{k=0}^{\infty} \binom{d}{k} (-L)^k$$

where L is the lag operator and the binomial coefficients are:

$$\binom{d}{k} = \frac{d(d-1)(d-2)\cdots(d-k+1)}{k!}$$

For d = 1: standard first difference (returns)
For d = 0: no differencing (prices)
For 0 < d < 1: fractional differencing

### Finding Minimum d for Stationarity

1. Apply fractional difference with varying d ∈ [0, 1]
2. Run ADF (Augmented Dickey-Fuller) test on each
3. Find minimum d where ADF rejects unit root at desired confidence (e.g., 5%)
4. This d preserves maximum memory while achieving stationarity

**Typical results**: Most financial series achieve stationarity with d ≈ 0.3–0.5, preserving substantial long-range dependence that pure returns (d = 1) would discard.

### Fixed-Width Window Approximation

The infinite sum is truncated to a window of width w:

$$(1-L)^d X_t \approx \sum_{k=0}^{w} \omega_k X_{t-k}$$

where weights $\omega_k = (-1)^k \binom{d}{k}$ decay but never reach exactly zero for fractional d. Choose w such that $|\omega_w| < \tau$ for some threshold τ (e.g., 1e-5).

### Connection to EQI Returns Framework

Paleologo's returns modeling (Ch 2) uses standard returns and log returns. Fractional differentiation is an alternative pre-processing step when building ML-based signals: it produces stationary features that retain more predictive information than standard returns. Use fractionally differentiated series as features in signal construction, not as a replacement for the return definitions used in portfolio PnL calculation.

---

## 6. Cross-Validation for Financial Data

### Why Standard k-Fold CV Fails in Finance

Standard k-fold cross-validation:
- Ignores temporal ordering → training on future data to predict past
- Ignores label overlap → leakage between folds through concurrent labels
- Assumes IID → violated by serial dependence in returns

### Purged k-Fold Cross-Validation

Modification of standard k-fold that removes leakage:

1. Split data into k contiguous time folds
2. For each test fold, **purge** all training observations whose labels overlap with any test observation's time span
3. Additionally, apply an **embargo** period after each test fold — remove h observations from training set following the test fold boundary

**Purging**: If test fold spans [t_a, t_b], remove any training observation whose label span overlaps [t_a, t_b].

**Embargo**: Remove training observations in [t_b, t_b + h] where h is typically the maximum label span length. Prevents leakage through serial correlation.

### Combinatorial Purged Cross-Validation (CPCV)

Standard purged k-fold produces k backtest paths. CPCV generates C(k, k-p) paths by testing on every combination of p folds out of k:

1. Divide data into k groups
2. For each combination of p test groups (out of k), use remaining k-p groups for training
3. Apply purging and embargo for each combination
4. Each combination produces a backtest path over the test groups

**Advantages**:
- Generates many more backtest paths than standard k-fold
- Each path is non-overlapping with training data
- Paths can be combined to form a distribution of backtest performance
- Reduces variance of performance estimates

**Number of paths**: C(k, p) × (p/k) unique time-ordered test paths

### Connection to Paleologo's RAS

Both CPCV and RAS address the same fundamental problem: how to validate strategies without data leakage while accounting for multiple testing. They are complementary:
- **RAS** (Paleologo): Controls false discovery rate across many strategies simultaneously. Best for: evaluating a large universe of candidate signals.
- **CPCV** (López de Prado): Generates distribution of backtest performance for a single strategy. Best for: validating a specific strategy's robustness before deployment.

Use both: CPCV to validate individual strategy robustness, RAS to control family-wise error when comparing multiple strategies.

---

## 7. Feature Importance

### Why It Matters

In ML-based signal construction, understanding which features drive predictions is critical for:
- Avoiding overfitting to noise features
- Understanding the economic logic of the signal
- Detecting regime changes (feature importance shifts)
- Model simplification (drop unimportant features)

### Mean Decrease Impurity (MDI)

For tree-based models (random forests, gradient boosted trees):
- At each node split, record the impurity decrease (e.g., Gini or entropy)
- For each feature, sum impurity decreases across all nodes where it was used
- Normalize to sum to 1

**Limitation**: Biased toward high-cardinality features (more split points → more opportunities to appear important). Misleading for correlated features.

### Mean Decrease Accuracy (MDA)

Permutation importance:
1. Train model on original data, record OOS accuracy
2. For each feature j: permute feature j's values across observations
3. Re-evaluate model accuracy with permuted feature
4. Importance of feature j = accuracy drop from permutation

**Advantages**: Model-agnostic, unbiased by cardinality
**Disadvantage**: Correlated features split importance (permuting one doesn't fully remove information if a correlated feature remains)

### Single Feature Importance (SFI)

Train separate model using only feature j:
1. For each feature j, fit model using only that feature
2. Evaluate OOS performance (e.g., accuracy, AUC)
3. Compare across features

**Advantage**: No interaction effects confound importance
**Disadvantage**: Misses feature interactions that create joint predictive power

### Practical Recommendation

Use all three methods together. Features that rank consistently high across MDI, MDA, and SFI are robust. Features that rank high in only one method may be artifacts of the specific importance metric.

### Clustered Feature Importance

When features are highly correlated:
1. Cluster features into groups (e.g., by correlation)
2. Compute feature importance at the cluster level (permute entire cluster)
3. Within-cluster importance: permute individual features while keeping cluster intact

This resolves the "importance dilution" problem where correlated features split credit.

---

## 8. Bet Sizing from ML Classifiers

### From Probability to Position Size

An ML classifier outputs probability p of a profitable trade. Convert to position size:

**Linear sizing**: size = (2p - 1) × max_size. Simple but doesn't account for distribution shape.

**Sigmoid sizing** (recommended):
$$m = (2p - 1)$$
$$\text{size} = m \cdot \frac{2\Phi(z) - 1}{2\Phi(z_{\max}) - 1}$$

where Φ is the normal CDF, z = m/σ_m is the standardized signal, and σ_m is estimated from the standard deviation of meta-labels.

### Average Active Bets and Concurrency

When multiple signals fire concurrently:
1. Track number of active bets at each time step
2. Scale new bet size inversely with number of concurrent active bets
3. This prevents concentration and implicitly controls portfolio-level risk

$$\text{adjusted\_size}_t = \frac{\text{raw\_size}_t}{c_t}$$

where c_t is the number of concurrent active bets at time t.

### Connection to Kelly

López de Prado's bet sizing complements Paleologo's Kelly framework:
- **Kelly**: Sizes based on Sharpe ratio and volatility (continuous, portfolio-level)
- **AFML bet sizing**: Sizes based on classifier confidence (discrete, trade-level)
- **Integration**: Use AFML bet sizing for individual trade entry, Kelly for overall portfolio leverage/allocation

---

## 9. Structural Breaks and Regime Detection

### CUSUM Tests

The Cumulative Sum (CUSUM) test detects changes in the mean of a process:

$$S_t = \max(0, S_{t-1} + y_t - E[y_t] - h)$$

where h is a threshold parameter. Signal when S_t exceeds decision boundary H.

**Application**: Detect when a signal's behavior has shifted (alpha decay, regime change). CUSUM on strategy returns, residuals, or feature values.

### Chow-Type Structural Break Tests

Test whether model parameters differ between two time periods:
1. Fit model on full sample → RSS_full
2. Fit model separately on sub-periods → RSS_1 + RSS_2
3. F-test: F = (RSS_full - RSS_1 - RSS_2) / (RSS_1 + RSS_2) × (T-2k)/k

**Recursive Chow test**: Run at every possible split point → identify timing of breaks.

### Explosiveness Tests (SADF / GSADF)

Sup ADF test for explosive behavior (bubbles):
1. Run ADF regression with expanding/rolling windows
2. Record t-statistic at each window
3. SADF = supremum of all ADF statistics
4. If SADF exceeds critical value → evidence of explosive episode

**Application**: Detect bubbles in asset prices, unsustainable momentum, or data-generating process instability before backtesting.

### Connection to EQI Framework

Paleologo's backtesting chapter (Ch 8) notes non-stationarity as a key pitfall but doesn't prescribe specific structural break tests. AFML provides the testing toolkit: use CUSUM and Chow tests to segment your backtest into regimes, and SADF to detect periods of explosive behavior that violate model assumptions.

---

## 10. Market Microstructure Features

### Kyle's Lambda

From Kyle (1985): measures price impact per unit of order flow:

$$Δp = λ × \text{signed\_volume}$$

where λ is Kyle's lambda. Higher λ → less liquid market, higher execution cost.

**Estimation**: Regress price changes on signed volume over rolling windows.

**Application**: Use as a feature in signal models (illiquidity predicts returns) or as an input to market impact models alongside Almgren-Chriss (see EQI Ch 10).

### VPIN (Volume-Synchronized Probability of Informed Trading)

Estimates probability that trading activity is driven by informed traders:

1. Classify each volume bar as buy-initiated or sell-initiated
2. Compute volume imbalance: |V_buy - V_sell| over rolling window of n bars
3. VPIN = E[|V_buy - V_sell|] / (V_buy + V_sell)

**Interpretation**: High VPIN → informed traders active → higher adverse selection risk → wider effective spread.

**Application**: Use VPIN as a feature for execution timing (delay trades when VPIN is high) or as an alpha feature (VPIN spikes predict short-term reversals).

### Amihud Illiquidity

$$\text{ILLIQ} = \frac{1}{T}\sum_{t=1}^{T}\frac{|r_t|}{\text{DollarVolume}_t}$$

Simple, robust illiquidity measure. Higher values → less liquid. Can be used as a factor in risk models (illiquidity premium) or as a trading cost proxy.

---

## Cross-Reference to EQI Pipeline

| AFML Concept | Maps to EQI Section | Relationship |
|---|---|---|
| Alternative bars | §2 Returns Modeling | Pre-processing before return calculation |
| Triple barrier labeling | §8 Backtesting | Labeling method for ML-based strategies |
| Meta-labeling | §12 Kelly / §8 Backtesting | Trade-level sizing complement to portfolio-level Kelly |
| Sample weights | §8 Backtesting | Training data quality for ML models |
| Fractional differentiation | §2 Returns Modeling | Alternative stationarity transform preserving memory |
| Purged CV / CPCV | §8 Backtesting | Complements RAS for single-strategy validation |
| Feature importance | §4 Factor Models | Feature selection analog to factor selection |
| Bet sizing | §12 Kelly Allocation | Trade-level complement to portfolio-level allocation |
| Structural breaks | §8 Backtesting | Regime detection toolkit for backtest segmentation |
| Kyle's lambda / VPIN | §10 Market Impact | Microstructure features extending Almgren-Chriss |

---

## Key AFML Principles (Cross-Cutting)

1. **Financial data is not IID**: Every method must account for serial dependence, label overlap, and non-stationarity
2. **Labeling determines ceiling**: A badly labeled dataset cannot be rescued by a better model
3. **Meta-labeling separates direction from sizing**: Don't ask one model to do both
4. **Feature importance is not optional**: If you can't explain which features drive your model, you're overfitting
5. **Cross-validation must be purged and embargoed**: Standard k-fold is invalid for financial time series
6. **Structural breaks invalidate stationarity assumptions**: Test for them before and during backtesting
7. **Bet sizing should reflect confidence, not just direction**: A trade with 90% probability deserves more capital than one with 55%

---

## References

- López de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley.
- Kyle, A. S. (1985). Continuous Auctions and Insider Trading. *Econometrica*.
- Easley, D., López de Prado, M., & O'Hara, M. (2012). Flow Toxicity and Liquidity in a High-Frequency World. *Review of Financial Studies*.

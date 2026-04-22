# Backtesting and Evaluating Excess Returns (Chapter 8 Reference)

## Overview

Backtesting is the systematic evaluation of trading strategies using historical data. The challenge: we cannot design experiments; studies are observational and repeated. This creates data leakage and multiple-testing problems requiring rigorous protocols.

---

## Backtesting Best Practices (Non-Negotiable)

### Data Sourcing Quality

Before any analysis, establish data quality:

- **Definition and interpretation**: Clearly define what data mean. Verify units, currency references, and time intervals. Eliminate unit conversion errors.
  
- **Provenance**: Understand data origin. Is the vendor collecting directly or intermediating? What is the population sampling methodology?

- **Completeness**: Check for obvious gaps (missing prices) and non-obvious missing data (redacted values). Missingness itself can be informative of future returns.

- **Quality assurance**: Verify vendor's quality control procedures. Detect change points in data characteristics.

- **Point-in-time vs restated data**: Ensure data reflects only information available "as of" a given date. This is a critical leakage check.

- **Transformations**: Understand all vendor transformations (imputation, winsorization, price calculations). Document and verify them.

### Data Leakage Prevention

Critical sources of look-ahead bias:

- **Survivorship bias**: Backtest only on surviving stocks → biases toward outperformers. **Remedy**: Use methodology applicable at each point in time; specify delisting rules in advance (e.g., write off entire position).

- **Financial statement timing**: Include data on release date, not period-end date. Use point-in-time data.

- **Price adjustments**: For returns, use split-adjusted prices. For feature generation, use as-of-date unadjusted prices.

- **Stock splits**: Splits adjust future prices, leaking information. Use adjusted prices only for returns.

- **Incomplete return histories**: Do not use characteristics that require future data (e.g., momentum that includes next-day return by mistake).

### Research Process Standards

- **Document everything**: Reproducible code, tracked decisions, archival of datasets used.

- **Define backtesting protocol beforehand**: Write down sequence of actions before any testing. Change rarely, with good reason, and rerun all tests.

- **Define datasets beforehand**: Specify which data sources and versions before optimization. Document new data decisions with integrity.

- **Have a theory**: Pre-register predictions based on theoretical motivation. Reduces multiple-testing pressure.

- **Use same settings in backtesting and production**: Same data point-in-time, optimization formulation, market impact model, codebase.

- **Calibrate market impact**: Don't accept vendor models at face value. Calibrate against live performance of current strategy.

- **Include borrow costs**: Model short-borrow costs explicitly; they have material impact on short-side performance.

---

## The Backtesting Protocol

Two main approaches, each with limitations:

### 1. Cross-Validation

Split data into training and holdout sets. Further split training into k folds:

- Estimate parameters on k-1 folds
- Evaluate on remaining fold
- Repeat k times, average performance
- Final validation on holdout set

**Problems in finance**:
- Serial dependence: Returns have short-memory but volatility has long-memory
- Data leakage: If validation fold precedes training, predictors may contain future information (e.g., momentum uses past returns)
- Inefficiency: Can optimize holdout sample instead of using as true validation
- Many predictors: Cycling through refinements on holdout defeats the purpose

**Empirical example**: With 1250 periods and 500 random predictors (no true signal), 19% of samples pass 1% significance level—massive false positive rate.

### 2. Walk-Forward Backtesting

Use data up to period t to predict returns in period t+1:

- Closest to production process
- Eliminates serial dependence and most data leakage
- Naturally adaptive as parameters update

**Problem**: Uses less training data than cross-validation, limiting applicability with huge parameter spaces.

---

## The Rademacher Anti-Serum (RAS)

A novel protocol addressing limitations of both cross-validation and walk-forward:
- Non-anticipative (no data leakage)
- Accounts for serial dependency  
- Uses all data
- Allows massive multiple testing
- Provides rigorous decision rule

### Setup (Equation 8.1-8.1)

Organize results as T×N matrix **X**:
- Rows: observations (time periods)
- Columns: strategies (N strategies or signals)
- x_{t,n}: Sharpe ratio or Information Coefficient for strategy n at time t

**Key assumption**: Rows are iid from common distribution P. 
- Justification: Daily returns show weak serial dependence
- Caveat: If autocorrelation exists at lag s, average into non-overlapping blocks of size s

Empirical mean vector:
```
θ̂(X) = (1/T) ∑_t x_t
```
This is the empirical Sharpe Ratio or IC for each strategy.

### Rademacher Complexity (Equation 8.1-8.1)

Define Rademacher random vector **ε** where each element is +1 or -1 with probability 1/2.

**Rademacher Complexity**:
```
R̂ = E_ε[sup_n ε^T x^n / T]
```

**Three interpretations**:

1. **Covariance to random noise**: Maximum expected covariance of any strategy to random +/- indicators. High R̂ means for every random sequence, some strategy matches it well.

2. **Two-way cross-validation**: For large T, can rewrite as:
   ```
   ε^T x^n / T = (1/2)(θ̂_n^+ - θ̂_n^-)
   ```
   where θ̂_n^+ and θ̂_n^- are strategy n's performance on random halves. High R̂ means worst-case discrepancy between subsets is large → unreliable performance.

3. **Span over performance space**: Geometric measure of how well N strategy vectors span ℝ^T. Independent of correlation structure: identical strategies → R̂ = 0; uncorrelated strategies → R̂ high.

**Upper bound (Massart's lemma)**:
```
R̂ ≤ √(2 log N / T)
```

### Main Result: Procedure 8.1 (Signals)

For Information Coefficient (IC) where |x_{t,n}| ≤ 1:

**With probability > 1 - δ, for ALL signals simultaneously**:

```
θ_n ≥ θ̂_n - 2R̂ - 2√(log(2/δ) / T)
```

where:
- **2R̂**: Data-snooping term (scales with number and correlation of strategies)
- **2√(log(2/δ) / T)**: Estimation error term (confidence interval on mean)

---

### Main Result: Procedure 8.2 (Sharpe Ratios)

For z-scored strategy returns with unbounded support:

**With probability > 1 - δ, for ALL strategies simultaneously**:

```
θ_n ≥ θ̂_n - 2R̂ - 3√(2 log(2/δ) / T) - √(2 log(2N/δ) / T)
```

Components:
- **2R̂**: Data-snooping penalty (depends on strategy correlation)
- **3√(2 log(2/δ) / T)**: Base estimation error
- **√(2 log(2N/δ) / T)**: Additional term for maximum over N strategies

---

## Practical Application of RAS

### Implementation Steps

1. **Backtest all strategies** using walk-forward procedure
2. **Compute matrix X** of Information Coefficients or Sharpe Ratios
3. **Estimate Rademacher complexity R̂** via simulation:
   - For each of many random Rademacher vectors ε
   - Compute sup_n |ε^T x^n| / T
   - Average over all realizations
4. **Apply bound** from Procedure 8.1 or 8.2
5. **Accept strategy** if right-hand side of bound > 0

### Interpretation and Tuning

**Empirical findings from simulations**:
- Bound is conservative; constant factors could be improved
- Data-snooping term (2R̂) is dominant when testing many strategies
- Estimation term decreases as √T
- False Discovery Rate (FDR) is zero (no false positives)
- Detection rate may be low (many true positives missed)

**Practical adjustment**: Relax bound by tuning coefficients:
```
θ_n ≥ θ̂_n - a·R̂ - b·√(2 log(2/δ) / T)
```

where a, b > 0 are calibrated via simulation on your data.

---

## Multiple Testing Solutions

### Bonferroni Correction

Divide significance level by number of tests: α/N.

**Problem**: Too conservative when testing thousands of strategies.

### False Discovery Rate (FDR)

Control expected proportion of false positives among rejected hypotheses.

Better for large test families than Bonferroni.

### Rademacher Anti-Serum (RAS)

Superior approach:
- Uniform bounds that hold for entire strategy set simultaneously
- Accounts for strategy correlation (dependent tests)
- Finite-sample guarantees (not asymptotic)
- Computationally efficient for millions of strategies

---

## Simulated Results

### Normally Distributed Returns

With 500-5000 strategies, T=2500-5000 periods, varying correlation ρ:

**Scenario 1: All strategies have zero true Sharpe**
- False positive rate: 0% (perfect specificity)
- No strategies pass the bound

**Scenario 2: 20% of strategies have true Sharpe = 0.2**
- Detection rate: 12-20% (varies by parameters)
- False Discovery Rate: 0%
- All detected strategies are true positives

**Key patterns**:
- Maximum empirical Sharpe increases with N (number of strategies)
- Rademacher complexity decreases with correlation (fewer independent strategies)
- Increasing T or decreasing N reduces estimation error
- Bounds are conservative but reliable

### Heavy-Tailed Returns (t-distribution, df=5)

Results qualitatively similar to normal case:
- False positive rate still near zero
- Slightly lower detection rates
- Demonstrates robustness to distributional assumptions

---

## Empirical Analysis: Published Anomalies

### Jensen et al. (2023) Factor Database

Tested 153 published stock characteristics per country.

**Results** (using Rademacher positive threshold only):
- **Most countries**: 0% strategies pass
- **USA** (13,155 observations): 18.3% pass  
- **Hong Kong, Malaysia**: ~0.7% pass

**Interpretation**: Most published anomalies do not survive Rademacher adjustment globally. US dominance likely due to longer history and deeper research.

### Chen & Zimmerman (2023) Database

192 anomalies, 5911 observations:
- Rademacher complexity: 0.033
- 3.1% of strategies pass

---

## ML-Specific Backtesting (AFML Extension)

→ For full detail: `references/ml-pipeline-afml.md` §6

When backtesting ML-based strategies, standard cross-validation is invalid due to temporal dependence and label overlap. Two complementary tools:

### Combinatorial Purged Cross-Validation (CPCV)

Standard k-fold leaks information through overlapping labels and temporal ordering. CPCV fixes this:
1. Divide data into k contiguous groups
2. For each combination of p test groups, train on remaining k-p groups
3. **Purge**: Remove training observations whose label spans overlap any test observation
4. **Embargo**: Remove h additional training observations after each test fold boundary (h = max label span)
5. Generates C(k,p) backtest paths — far more than standard k-fold

**Use CPCV for**: Validating a specific strategy's robustness before deployment.
**Use RAS for**: Controlling false discovery when comparing many strategies simultaneously.

### Triple Barrier Labeling

Standard labeling (sign of next-period return) ignores path dependency and real trading mechanics. Triple barrier defines three exit conditions:
- Upper barrier (take-profit): entry + pt × σ
- Lower barrier (stop-loss): entry - sl × σ  
- Vertical barrier (max holding period): t + max_hold

Label = which barrier is hit first. Produces variable holding periods and volatility-adjusted targets.

### Meta-Labeling

Two-stage approach: primary model determines direction, secondary model determines whether to take the trade and how to size it. Separates the hard problem (direction) from the easier problem (confidence filtering). Directly applicable to quantamental workflows where the primary model is a discretionary view.

---

## Common Pitfalls in Detail

### 1. Overfitting via Parameter Tuning
Using same data for model selection and evaluation inflates performance. **Mitigation**: Walk-forward with pre-specified tuning procedure.

### 2. Survivorship Bias
Testing only on surviving stocks upbiases returns. **Mitigation**: Use historically accurate investment universe; model realistic delisting impact.

### 3. Look-Ahead Bias
Using information not available at decision time. **Mitigation**: Separate data layers by timing; use point-in-time data; check for financial statement leakage.

### 4. Underestimating Transaction Costs
Historical market impact models often understate real costs. **Mitigation**: Calibrate models against live trading; include borrow costs.

### 5. Non-Stationarity
Markets change; past relationships may not persist. **Mitigation**: Use walk-forward (adaptive); validate out-of-sample; understand structural breaks.

### 6. Multiple Testing
Testing many strategies increases false positives. **Mitigation**: Use RAS or other multiple-testing correction; pre-specify hypotheses; have theoretical motivation.

---

## Signal Evaluation Criteria

### Information Coefficient (IC)

Correlation between predicted alpha and actual idiosyncratic returns:
```
IC = α^T ε / (|α| |ε|)
```

Typical IC: 0.01-0.05 for good signals.

**In RAS context**: IC is bounded [0,1], so estimation error is tighter.

### Sharpe Ratio

Strategy's return divided by predicted volatility (z-scored):
```
SR = w^T r / √(w^T Ω w)
```

Typical thresholds:
- SR > 1.0: Good
- SR > 2.0: Excellent
- SR > 3.0: Suspicious (may indicate overfitting)

**In RAS context**: Unbounded, so estimation error includes √log(N) term.

### Information Ratio

Alpha return divided by alpha volatility:
```
IR = E[α] / σ(α)
```

Similar interpretation to Sharpe but applied to alpha specifically.

---

## Practical Implementation Checklist

- [ ] Data sourced from reputable vendor with known quality standards
- [ ] Point-in-time data used (no look-ahead bias)
- [ ] Survivorship bias eliminated (time-dependent universe)
- [ ] Delisting rules specified in advance
- [ ] Transaction costs and market impact modeled conservatively
- [ ] Borrow costs included for shorts
- [ ] Backtesting protocol written before optimization
- [ ] Walk-forward procedure used (not cross-validation)
- [ ] Rademacher Anti-Serum applied for multiple-testing control
- [ ] Results documented with full reproducibility
- [ ] Out-of-sample validation on held-aside period
- [ ] Theoretical motivation articulated for each signal
- [ ] Parameter tuning limited to reasonable space
- [ ] Results validated against published databases when possible

---

## Key Formulas Summary

| Concept | Formula | Interpretation |
|---------|---------|-----------------|
| Empirical performance | θ̂ = (1/T)∑ x_t | Sample mean performance |
| Rademacher complexity | R̂ = E_ε[sup_n ε^T x^n / T] | Max correlation to noise |
| Massart bound | R̂ ≤ √(2 log N / T) | Upper bound on complexity |
| IC lower bound | θ_n ≥ θ̂_n - 2R̂ - 2√(log(2/δ)/T) | Guaranteed true IC |
| SR lower bound | θ_n ≥ θ̂_n - 2R̂ - 3√(2log(2/δ)/T) - √(2log(2N/δ)/T) | Guaranteed true Sharpe |
| False positive bound | P(max θ̂_n > 0 \| true θ = 0) ≤ δ | Probability of false discovery |

---

## References to Key Sections

- **Data quality**: Section 8.1.1
- **Data leakage prevention**: Section 8.1.2
- **Cross-validation problems**: Section 8.2.1
- **Walk-forward design**: Section 8.2.1
- **RAS theory**: Section 8.3
- **RAS procedures**: Section 8.3.2
- **Simulations**: Section 8.4.1
- **Empirical validation**: Section 8.4.2


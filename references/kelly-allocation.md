# Kelly Allocation: Dynamic Risk Allocation for Long-Term Growth

Based on Chapter 13 of Paleologo's *Elements of Quantitative Investing*

## Overview

The Kelly criterion provides a principled approach to capital allocation that maximizes long-term growth by balancing expected returns against volatility. Unlike single-period mean-variance optimization, Kelly strategies account for multi-period wealth dynamics and the compounding effects of reinvestment.

## Core Principle: Maximizing Expected Log Returns

The fundamental objective of Kelly allocation is to maximize the expected growth rate of capital over time:

$$\max_x E[\log(1 + xr)]$$

where $x$ is the investment fraction and $r$ is the return of a risky strategy.

This is equivalent to maximizing expected utility with logarithmic utility function, which has deep theoretical justification in information theory and optimal growth.

## Kelly Criterion Derivation

### Single Security Case

For a single risky asset with excess return $r$ having mean $\mu$ and variance $\sigma^2$, the Kelly optimal allocation solves:

$$\max_x g(x) := \max_x E[\log(1 + rx)]$$

**Exact Solution:**
$$x^* = \arg\max_x E[\log(1 + rx)]$$
$$g(x^*) = E[\log(1 + rx^*)]$$

**Quadratic Approximation** (using $\log(1+x) \approx x - x^2/2$):
$$x_1^* \approx \frac{\mu}{\mu^2 + \sigma^2}$$
$$g(x_1^*) \approx \frac{1}{2} \cdot \frac{SR^2}{SR^2 + 1}$$

**Further Simplified** (assuming $\mu \ll \sigma$):
$$x_2^* \approx \frac{SR}{\sigma}$$
$$g(x_2^*) \approx \frac{1}{2} SR^2$$

where $SR = \mu/\sigma$ is the Sharpe ratio.

### Key Relationship: Dollar Volatility and Capital

The optimal Kelly strategy maintains a constant ratio between dollar-deployed volatility and capital:

$$\frac{\text{(Dollar Volatility)}}{\text{(Capital)}} = SR$$

For example: $1B capital + annualized SR of 2 → deploy $2B of dollar volatility.

This equivalently means:
$$\text{Expected Strategy Return} = SR \times \frac{\text{Dollar Volatility}}{\text{Capital}} = SR^2$$

## Multi-Asset Kelly Extension

For a portfolio of assets with expected returns $\boldsymbol{\mu}$ and covariance matrix $\boldsymbol{\Sigma}$:

$$x^* = \boldsymbol{\Sigma}^{-1} \boldsymbol{\mu}$$

This represents the optimal leverage applied to the tangency portfolio. In the presence of a risk-free asset:

$$x^* = \frac{\boldsymbol{\Sigma}^{-1} \boldsymbol{\mu}}{1^T \boldsymbol{\Sigma}^{-1} \boldsymbol{\mu}}$$

## Mathematical Properties (Breiman 1961)

Kelly strategies possess remarkable theoretical properties:

1. **Growth Dominance**: For a Kelly strategy $X_t$ and any alternative strategy $Y_t$ with lower expected growth rate:
   $$\lim_{t \to \infty} \frac{X_t}{Y_t} = \infty \text{ (almost surely)}$$

2. **Necessary and Sufficient Growth Condition**: Let $g = E[\log(1 + r)]$. Then:
   - If $g > 0$: $X_t \to \infty$ (almost surely)
   - If $g < 0$: $X_t \to -\infty$ (almost surely)
   - If $g = 0$: $\limsup X_t = \infty, \liminf X_t = -\infty$ (almost surely)

3. **Shortest Path to Target Capital**: The expected time to reach capital level $C$ is $\log(C)/g$. Kelly strategies minimize this time.

## Fractional Kelly Strategies

Pure Kelly can lead to unacceptable drawdowns. Fractional Kelly allocates a fraction $0 < f < 1$ of the optimal amount:

$$x_{\text{frac}}^* = f \cdot x^*$$

### Three Interpretations

1. **Risk-Free Combination**: Fractional Kelly is equivalent to combining the risk-free asset with full Kelly. The portfolio volatility scales by $f$.

2. **Higher Risk Aversion**: Equivalent to solving:
   $$\max_x \mu x - \frac{\lambda}{2}(\sigma^2 + \mu^2)x^2, \quad \lambda > 1$$
   yielding $x_{\text{frac}}^* = x^*/\lambda$.

3. **Parameter Uncertainty**: When parameters are uncertain, the optimal allocation shrinks. For return uncertainty $\tau^2$ on expected return $\mu$:
   $$x_{\text{robust}}^* \approx \frac{\mu}{\mu^2 + \sigma^2 + \tau^2}$$
   This can be substantially lower than the point-estimate Kelly allocation.

### Practical Guidance

Set maximum tolerable percentage volatility per rebalancing interval to $p$:
$$x = \min\left(\frac{p}{\sigma}, SR\right)$$

**Example**: $1B capital, SR=2, weekly volatility limit 1% of capital
- Weekly SR: $2/\sqrt{52} \approx 0.27$
- Volatility constraint: $p/\sigma = 0.01/3 \approx 0.003$
- Fractional Kelly: $\min(0.003, 0.27) = 0.003$

## Grossman-Zhou Drawdown Control

Grossman and Zhou (1993) propose a dynamic allocation that controls maximum drawdown while maintaining growth optimization. Let:
- $M_t = \max_{s \in [0,t]} W_s$ = high watermark of wealth
- $d_t = 1 - W_t/M_t$ = current drawdown percentage
- $D$ = maximum tolerable drawdown percentage

The optimal policy is:

$$f_t = \frac{\mu}{\sigma^2}\left(1 - \frac{1-D}{1-d_t}\right)$$

**Properties:**
- At high watermark ($d_t = 0$): $f_t = \frac{\mu}{\sigma^2} \cdot D$ (fractional Kelly with fraction $D$)
- As drawdown approaches limit: $f_t \to 0$ (position reduces to risk-free)
- When $D = 1$ (infinite tolerance): reverts to standard Kelly

**Trade-off**: GZ controls drawdowns deterministically but may reduce realized Sharpe ratio due to volatility modulation. Changing volatility independent of expected returns reduces the Sharpe ratio.

## Critical Limitations and Warnings

1. **Sharpe Ratio Dependency on Capital**: As capital increases, Sharpe ratio of an active strategy typically decreases. The formula $x^* = SR/\sigma$ no longer holds; must solve:
   $$x = \frac{SR(x)}{\sigma}$$
   No analytical solution in general case.

2. **Transaction Costs**: Rebalancing to maintain fraction $x^*$ incurs trading costs. The required trade at each rebalancing after positive PnL is:
   $$\delta_t = \frac{1-x^*}{1+r_t} W_t r_t$$
   These costs compound and eventually dominate as wealth grows.

3. **Parameter Estimation Risk**: Kelly is sensitive to parameter misestimation. Small errors in $\mu$ or $\sigma$ lead to large errors in $x^*$. Parameter uncertainty justifies fractional Kelly.

4. **Drawdown Reality**: Pure Kelly can experience severe drawdowns. Historical simulation of S&P 500 (1926-2018) shows:
   - Optimal Kelly fraction: $x^* \approx 1.88$ to 2.2 (depending on approximation)
   - Maximum historical drawdown: 80%+ when fully leveraged
   - Unlevered S&P 500 (x=1): maximum historical drawdown ~57%

5. **Discrete vs. Continuous Rebalancing**: Approximation accuracy improves with rebalancing frequency. Daily Sharpe ratio of 0.24 (annualized 4) required for <2% error with $x_1^*$; daily SR of 0.15 (annualized 2.4) for $x_2^*$.

## Insight 13.1: Intuition Behind Kelly Strategies

- **Goal**: Achieve highest long-term capital growth
- **Simplicity**: Optimal strategy allocates constant fraction of capital
- **Bounds**: Lower bound ensures $g > 0$; upper bound is $x^*$
- **Sharpe-Proportional**: To first approximation, optimal fraction and volatility/capital ratio scale with Sharpe ratio

## Insight 13.2: Practitioners Use Fractional Kelly Implicitly

Successful investors typically allocate capital so that volatility/capital ratio is constant or slowly varying. This naturally implements fractional Kelly without explicit formula application. The ratio $\text{Dollar Volatility}/\text{Capital}$ serves as a simple heuristic that embeds Kelly principles.

## Comparison: Kelly, Fractional Kelly, Grossman-Zhou

All three are valid frameworks with different objectives:

| Criterion | Objective | Volatility | Drawdown Control |
|-----------|-----------|-----------|------------------|
| Full Kelly | Max long-term growth | Increasing with capital | None |
| Fractional Kelly | Growth + stability trade-off | Constant | Implicit (via reduced leverage) |
| Grossman-Zhou | Growth + explicit drawdown limit | Time-varying | Deterministic (almost sure) |

## Summary of Key Formulas

| Concept | Formula |
|---------|---------|
| Kelly objective | $\max_x E[\log(1+xr)]$ |
| Optimal Kelly (approx.) | $x^* \approx SR/\sigma$ |
| Expected return at Kelly | $E[r_{\text{Kelly}}] = SR^2$ |
| Volatility/Capital ratio | $(Dollar \, Vol)/(Capital) = SR$ |
| Fractional Kelly | $x_f = f \cdot x^*$ for $0 < f < 1$ |
| Robust Kelly (param uncertainty) | $x^* \approx \mu/(\mu^2+\sigma^2+\tau^2)$ |
| Grossman-Zhou policy | $f_t = (\mu/\sigma^2)(1 - (1-D)/(1-d_t))$ |
| Log growth at Kelly | $g(x^*) \approx SR^2/2$ |

## Practical Implementation Checklist

- [ ] Estimate Sharpe ratio and volatility of strategy
- [ ] Compute Kelly optimal allocation: $x^* = SR/\sigma$
- [ ] Assess parameter uncertainty and estimate robust Kelly
- [ ] Choose fractional Kelly factor $f$ based on risk tolerance
- [ ] Set rebalancing frequency (daily recommended for low-volatility strategies)
- [ ] Account for transaction costs in live deployment
- [ ] Monitor Sharpe ratio dependency on deployed capital
- [ ] Consider Grossman-Zhou overlay for drawdown control if needed
- [ ] Include standard errors/confidence intervals in performance reporting

## Bet Sizing from ML Classifiers (AFML Extension)

→ For full detail: `references/ml-pipeline-afml.md` §8

Kelly sizes positions at the portfolio level based on Sharpe ratio and volatility. AFML provides a complementary trade-level sizing approach based on ML classifier confidence:

**Sigmoid sizing**: Convert classifier probability p to position size via:
- m = (2p - 1) — maps probability to [-1, +1] signal
- size = m × (2Φ(z) - 1) / (2Φ(z_max) - 1) — normalizes via CDF

**Concurrency adjustment**: Scale new bet size inversely with number of concurrent active bets to prevent concentration.

**Integration with Kelly**: Use AFML bet sizing for individual trade entry decisions (how much confidence does the model have in this specific trade?). Use Kelly for overall portfolio leverage and capital allocation (how much total risk should the portfolio take?). The two operate at different levels of the investment pipeline and reinforce each other.

**Meta-labeling connection**: When the primary model is discretionary (quantamental), the meta-model's probability output directly feeds bet sizing — higher meta-label probability → larger position, implementing quantitative discipline on discretionary conviction.

---

## References

- Breiman (1961): Optimal Growth Investing
- Grossman and Zhou (1993): Optimal Investment Strategies with Drawdown Control
- MacLean et al. (1992, 2004, 2010): Comprehensive Kelly Theory
- Thorp (2006): Evidence for Fractional Kelly under Parameter Uncertainty

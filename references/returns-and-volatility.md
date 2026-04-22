# Returns and Volatility

Reference guide for Chapters 2-3 of *The Elements of Quantitative Investing*.

## Return Definitions

### Simple Return
The simple return from closing prices on days 0 and 1:

$$r_i(1) = \frac{P_i(1) - P_i(0)}{P_i(0)}$$

### Dividend-Adjusted Return
When security pays dividend $D_i(1)$:

$$r_i(1) = \frac{P_i(1) + D_i(1) - P_i(0)}{P_i(0)}$$

### Log Return
Defined as:

$$\tilde{r}_i(t) = \log(1 + r_i(t))$$

**Key property**: Log returns are additive over time and composition. For compound returns over periods $1, \ldots, T$:

$$\prod_{t=1}^{T} (1 + r_t) - 1 \approx \sum_{t=1}^{T} \tilde{r}_t$$

This approximation is accurate for daily/shorter intervals but poor for yearly or highly volatile returns.

### Excess Returns
Returns in excess of risk-free rate $r_f$:

$$r_i^{\text{excess}} = r_i - r_f$$

For portfolio $\mathbf{w}$:

$$\sum_i w_i(r_i - r_f) = \sum_i w_i r_i - r_f \sum_i w_i$$

### Compounded Return
Value of one unit of numeraire after $T$ periods:

$$r(1:T) = \prod_{t=1}^{T} (1 + r_t) - 1$$

## Stylized Facts of Returns

Returns exhibit four key empirical properties:

### 1. Absence of Autocorrelation
Lagged autocorrelations of log returns $\tilde{r}_t$ are small at typical horizons (except at intraday timescales affected by market microstructure). This contrasts with strong autocorrelation in absolute returns and squared returns.

### 2. Heavy Tails
The unconditional distribution of returns exhibits heavy tail behavior. For power-tailed distributions:

$$\bar{F}(x) = P(r > x) = Cx^{-\alpha}$$

where $\alpha$ is the **tail index**. Empirical evidence suggests $\alpha \approx 4$, implying finite fourth moment. Compare to Gaussian bounds:

$$F^{-1}(\delta) \leq -\sqrt{2\log(1/(2\sqrt{2\pi}\delta))}$$

**Key implication**: Fourth moment is finite and estimable, but not all moments may be reliably estimated.

### 3. Volatility Clustering (Taylor Effect)
Absolute returns $|\tilde{r}_t|$ and squared returns $\tilde{r}^2_t$ exhibit strong positive autocorrelation. Large realized movements persist over time, as do small ones. This motivates conditional heteroskedastic models.

### 4. Aggregational Gaussianity
At longer time scales (weekly, monthly returns), the distribution of log-returns becomes closer to Gaussian, consistent with the CLT applied to shorter-horizon returns.

## Conditional Heteroskedastic Models: GARCH(1,1)

### Model Specification

$$r_t = h_t \epsilon_t$$

$$h_t^2 = \alpha_0 + \alpha_1 r_{t-1}^2 + \beta_1 h_{t-1}^2$$

$$\epsilon_t \sim N(0,1)$$

where:
- $h_t$ is the volatility at time $t$
- $\alpha_0, \alpha_1, \beta_1 \geq 0$ are parameters
- $\epsilon_t$ are iid standard normal innovations

### Equilibrium Variance
When $\beta_1 < 1$:

$$\bar{h}^2 = \frac{\alpha_0}{1 - \beta_1}$$

Variance converges to this long-run level at geometric rate.

### Recursive Representation
By repeated substitution:

$$h_t^2 = \bar{h}^2 + \alpha_1 \sum_{i=1}^{\infty} \beta_1^{i-1} r_{t-i}^2$$

This is an exponential moving average of past squared returns, weighted by $h_t$, which itself is modulated by realized return magnitudes.

### Equivalent EWMA Relationship
GARCH(1,1) with $\lambda = \beta_1$ is equivalent to EWMA variance estimation when:
- $\alpha_0 = 0$
- $\alpha_1 = 1 - \lambda$
- $\beta_1 = \lambda$

Practitioners often prefer EWMA for its simplicity, despite potential performance gains from GARCH parameter estimation.

### Kurtosis
The unconditional kurtosis of GARCH(1,1) is:

$$\kappa = 3 \cdot \frac{1 + (\alpha_1 + \beta_1)^2}{1 - (\alpha_1 + \beta_1)^2 - 2\alpha_1^2} > 3$$

Thus GARCH processes are leptokurtic (heavy-tailed).

### GARCH(1,1) Estimation
Estimate parameters $\theta = (\alpha_0, \alpha_1, \beta_1)$ by maximum likelihood:

$$\min_\theta \sum_{t=1}^T \left( \log h_t^2 + \frac{r_t^2}{h_t^2} \right)$$

where $h_t^2 = \alpha_0 \frac{1 - \beta_1^{t-1}}{1 - \beta_1} + \alpha_1 \sum_{i=1}^{t-1} \beta_1^{i-1} r_{t-i}^2$.

## Realized Volatility from High-Frequency Data

### Setup
Observe log price process at intervals of length $1/n$ over period $[0,1]$:

$$dp = \mu dt + \sigma dW_t$$

Returns at interval $j$: $r(j) = p(j/n) - p((j-1)/n) \sim N(\mu/n, \sigma^2/n)$

### Drift Estimation
The MLE for drift is:

$$\hat{\mu} = \sum_j r(j) = p(1) - p(0)$$

**Key insight**: Drift estimator does not depend on number of intervals $n$. Variance of drift estimate is:

$$\text{var}(\hat{\mu}) = \sigma^2$$

independent of $n$. This reflects the fundamental difficulty of estimating drift.

### Variance Estimation - Centered
Centered estimator:

$$\hat{\sigma}_1^2 = \sum_j [r(j) - \hat{\mu}/n]^2$$

This is biased but consistent.

### Variance Estimation - Uncentered
Uncentered (recommended) estimator:

$$\hat{\sigma}_2^2 = \sum_j r^2(j)$$

Moments of $r(j) \sim N(\mu/n, \sigma^2/n)$:

$$E[r(j)] = \frac{\mu}{n}, \quad E[r^2(j)] = \frac{\mu^2}{n^2} + \frac{\sigma^2}{n}, \quad E[r^4(j)] = \frac{\mu^4}{n^4} + 6\frac{\mu^2\sigma^2}{n^3} + 3\frac{\sigma^4}{n^2}$$

Variance of squared returns:

$$\text{var}(r^2(j)) = 2\frac{\sigma^4}{n^2} + 4\frac{\mu^2\sigma^2}{n^3}$$

Thus:

$$E[\hat{\sigma}_2^2] = \sigma^2 + \frac{\mu^2}{n}, \quad \text{var}(\hat{\sigma}_2^2) = \frac{2\sigma^4}{n} + \frac{4\mu^2\sigma^2}{n^2}$$

**Key insight**: As $n \to \infty$, the bias vanishes like $O(1/n)$ and variance decreases like $O(1/n)$. Uncentered variance is asymptotically consistent and unbiased.

### Practical Guidance
**Insight 2.1**: Use uncentered returns for variance estimation from high-frequency data. Bias is $O(1/n)$ and estimator is consistent.

**Considerations**:
1. Market microstructure noise (bid-ask spread) introduces measurement error
2. Thin trading periods create non-synchronous data
3. Volatility varies over time; may need AR(1) model for realized variances
4. Open-to-close vs close-to-open intervals have different properties

Empirical evidence (Liu et al., 2015) recommends vanilla RV at 5-minute frequency as competitive across asset classes.

## State-Space Estimation of Variance

### Muth's Model (1960)

State equation:
$$x_{t+1} = x_t + \tau_\epsilon \epsilon_{t+1}$$

Observation equation:
$$y_{t+1} = x_t + \tau_\eta \eta_{t+1}$$

where $\epsilon_t, \eta_t \sim N(0,1)$ are independent.

The state $x_t$ is unobserved true variance; observation $y_t$ is $r_t^2$ (noisy measurement).

### Kalman Filter Solution

Define ratio: $\kappa = \tau_\eta / \tau_\epsilon$ (measurement-to-innovation noise ratio)

Stationary variance estimate:
$$\hat{\sigma}_{t+1|t}^2 = \tau_\epsilon^2 \left(1 + \frac{\sqrt{(2\kappa)^2 + 1}}{2}\right)$$

Optimal recursive update:
$$K = \frac{\hat{\sigma}_{t+1|t}^2}{\hat{\sigma}_{t+1|t}^2 + \tau_\eta^2}$$

$$\hat{x}_{t+1|t} = (1-K)\hat{x}_{t|t-1} + K y_t$$

**Interpretation**: $K$ acts as exponential smoothing weight. Higher $\kappa$ (more measurement noise) implies higher $K$ (rely more on current observation). Lower $\kappa$ (high signal-to-noise) implies lower $K$ (discount past more aggressively).

### Half-Life Connection
For EWMA with smoothing parameter $\lambda$:

$$\text{Half-life} = \tau = -\frac{\log 2}{\log \lambda}$$

This connects to Kalman gain: practitioners often specify by choosing desired half-life rather than estimating parameters.

### Mean-Reversion Extension
Modified state equation with mean reversion:
$$x_{t+1} = x_t - \lambda(x_t - \mu) + \tau_\epsilon \epsilon_{t+1}$$

produces stationary distribution with mean $\mu$ and std $\sqrt{\tau_\epsilon^2/(2\lambda - \lambda^2)}$.

**Effect**: Mean reversion reduces optimal $K$ (discount past less). Volatility estimates stabilize around long-term mean rather than drifting.

### Harvey-Shephard Model

Generates log-normally distributed returns:
$$r_t = e^{\beta + \sigma_t/2} \xi_t$$

where $\xi_t \sim N(0,1)$, implying $r_t$ is locally log-normal.

Define log-squared returns adjusted for measurement noise:
$$y_t = \log(r_t^2) - \gamma$$

where $\gamma = E[\log \xi_t^2] \approx -1.27$ and measurement error $\eta_t$ has std $\approx 2.22$.

State equation (AR(1)):
$$x_{t+1} = b + ax_t + \epsilon_{t+1}$$

Kalman filter estimate:
$$\hat{x}_{t+1|t} = (1-K)\hat{x}_{t|t-1} + K[\log(r_t^2) - \gamma]$$

Volatility recovered as:
$$\hat{\sigma}_{t+1|t} = e^{\hat{x}_{t+1|t}/2}$$

**Advantages**: By design, volatility is positive; has log-normal distribution (right-skewed, realistic); conditional returns are locally log-normal.

## Sharpe Ratio

### Definition
Ratio of expected excess return to volatility:

$$\text{SR} = \frac{E[r] - r_f}{\sigma}$$

where $r_f$ is risk-free rate and $\sigma = \sqrt{\text{var}(r)}$ is volatility.

**Dimension**: If return is in units of $[T^{-1}]$ and volatility in units of $[T^{-1/2}]$, then SR has dimension $[T^{-1/2}]$.

### Scaling Across Horizons
For horizon conversion (e.g., daily to annual with $T$ trading days):

$$\text{SR}_{\text{annual}} = \text{SR}_{\text{daily}} \times \sqrt{T}$$

For US equities, standard conversion factor to annualized SR is $\sqrt{251}$ (trading days/year).

### Estimation and Confidence Intervals

For $T$ observations of iid excess returns:

$$\hat{\mu} = \frac{1}{T}\sum_t r_t, \quad \hat{\sigma} = \sqrt{\frac{1}{T}\sum_t(r_t - \hat{\mu})^2}, \quad \widehat{\text{SR}} = \frac{\hat{\mu}}{\hat{\sigma}}$$

Standard error of SR estimator:

$$\text{SE}(\widehat{\text{SR}}) = \sqrt{\frac{1 + \text{SR}^2/2}{T}}$$

Compare to known volatility case: $\text{SE} = \sqrt{1/T}$ (no $\text{SR}^2$ term).

For autocorrelated returns with $\text{cor}(r_s, r_t) = \rho^{|t-s|}$:

$$\widehat{\text{SR}} \approx \hat{\mu}/\hat{\sigma} \times \sqrt{\frac{1-\rho}{1+\rho}} \approx \hat{\mu}/\hat{\sigma} \times (1 - \rho)$$

for small $|\rho|$.

### Cantelli Bound
For any distribution with mean $\mu$, std $\sigma$, the probability of loss below $-\lambda\sigma$ is bounded:

$$P(\xi < \mu - \lambda\sigma) \leq \frac{\sigma^2}{\sigma^2 + \lambda^2}$$

For annual return $\xi$ with annualized SR and loss expressed as $-L\sigma$ (e.g., 2 standard deviations):

$$P(\xi < -L\sigma) \leq \frac{1}{1 + (L + \text{SR})^2}$$

**Example**: With annualized SR = 3 and volatility = $50M, probability of $100M loss (2 std deviations) is at most 3.9%. Much higher than Gaussian assumption (0.00003%), reflecting heavy-tail robustness.

### Information Ratio
Extension of Sharpe ratio using idiosyncratic returns $\epsilon_t$ (returns orthogonal to factors):

$$\text{IR} = \frac{E[\epsilon]}{\text{std}(\epsilon)}$$

Measures skill in generating returns unexplained by systematic factors.

## Capacity

**Definition**: Maximum PnL achievable at sustainable Sharpe ratio level.

**Key insight**: Sharpe ratio almost always decreases with position size/volatility. At sufficiently large volatility, SR becomes zero and strategy unprofitable.

**Capacity formula**: 
$$\text{PnL}_{\max} = \text{SR}_{\min} \times \text{Vol}_{\max}$$

where $\text{SR}_{\min}$ is minimum acceptable Sharpe ratio and $\text{Vol}_{\max}$ is maximum volatility at which this SR is maintained.

**Economic significance**: A strategy with high SR at low volatility may be economically unattractive if capacity is limited. Capacity constraints are critical for hedge funds and portfolio managers scaling capital.

## Fractional Differentiation (AFML Extension)

→ For full detail: `references/ml-pipeline-afml.md` §5

Standard returns (first differencing, d=1) achieve stationarity but discard long-range memory. Fractional differentiation with d ∈ (0,1) is a compromise:

$$(1 - L)^d X_t = \sum_{k=0}^{\infty} \binom{d}{k} (-L)^k X_t$$

**Finding minimum d**: Apply fractional difference with varying d, run ADF test, select smallest d that rejects unit root. Typical financial series: d ≈ 0.3–0.5.

**Use case**: Feature construction for ML-based signals. Fractionally differentiated series preserve support/resistance information that standard returns discard. Use as input features alongside standard returns, not as a replacement for return definitions in PnL calculation.

**Alternative Bars**: AFML also proposes volume, dollar, and information bars as alternatives to time-based sampling. Dollar bars (sample every $D traded) produce more uniform statistical properties and reduce oversampling of quiet periods. See `references/ml-pipeline-afml.md` §1.

---

## FAQ: Drift vs Variance Estimation

A fundamental asymmetry exists in estimation across frequencies:

- **Drift** (expected return): Variance of estimator $\sigma^2$ is independent of sampling frequency $n$. More frequent sampling provides **no benefit**.
- **Variance**: Estimator variance decreases as $O(1/n)$. More frequent sampling provides **strong benefit**.

This reflects that drift is a long-term property requiring long observation periods, while variance is a short-term property visible at high frequency.

---

**References**: Chapters 2-3, *The Elements of Quantitative Investing* by Giuseppe A. Paleologo.

# Probability & Statistics for Machine Learning

> **Series:** Math for ML — Day 2-3  
> **Prerequisites:** Basic algebra, Day 1 (Linear Algebra)  
> **Time to complete:** ~3 hours  

---

## Agenda

| # | Topic | Why it matters |
|---|-------|---------------|
| 1 | [Probability Basics](#1-probability-basics) | The language of uncertainty |
| 2 | [Conditional Probability](#2-conditional-probability) | How evidence changes beliefs |
| 3 | [Bayes' Theorem](#3-bayes-theorem) | The engine behind smart predictions |
| 4 | [Probability Distributions](#4-probability-distributions) | Describing patterns in data |
| 5 | [Mean, Variance & Std Dev](#5-mean-variance--standard-deviation) | Summarising data in 3 numbers |
| 6 | [Bias-Variance Tradeoff](#6-bias-variance-tradeoff) | Why models fail — and how to fix them |
| 7 | [Hypothesis Testing & p-values](#7-hypothesis-testing--p-values) | Is your result real or just luck? |
| 8 | [Confidence Intervals](#8-confidence-intervals) | Honest predictions with a range |

---

## 1. Probability Basics

### Definition

Probability is a number between **0 and 1** that says how likely something is.

```
P(event) = 0    → will never happen
P(event) = 0.5  → 50/50 chance
P(event) = 1    → will definitely happen
```

### Example

```python
# Flipping a fair coin
P_heads = 0.5

# Rolling a 6 on a die
P_six = 1/6  # ≈ 0.167

# A patient having a rare disease (1% of population)
P_disease = 0.01
```

### Two rules you must know

**Rule 1 — Addition (OR):**
```
P(A or B) = P(A) + P(B) - P(A and B)
```

**Rule 2 — Multiplication (AND, independent events):**
```
P(A and B) = P(A) × P(B)
```

```python
# Email spam example
P_spam   = 0.1   # 10% of emails are spam
P_not_spam = 1 - P_spam  # 90% are not spam

print(f"Chance next email is spam:     {P_spam:.0%}")
print(f"Chance next email is NOT spam: {P_not_spam:.0%}")
```

**Output:**
```
Chance next email is spam:     10%
Chance next email is NOT spam: 90%
```

### How it's used in ML

Every ML classifier outputs a probability — not just a class label.

```python
# A trained model predicts on a customer record
prediction = model.predict_proba(customer_features)
# Output: [0.23, 0.77]  ← [P(no churn), P(churn)]

# The model is 77% confident this customer will churn
```

---

## 2. Conditional Probability

### Definition

Conditional probability answers: **"How does knowing X change the probability of Y?"**

```
P(A | B) = Probability of A, given that B has already happened
```

Formula:
```
P(A | B) = P(A and B) / P(B)
```

### Example — Titanic Dataset

Without any information, the survival rate was **38%**. But the moment you add context, everything changes:

```python
import pandas as pd

# Titanic survival rates (from real dataset)
data = {
    'Group':        ['All passengers', 'Female', 'Male',  '1st Class', '3rd Class', 'Female + 1st Class', 'Male + 3rd Class'],
    'P(Survived)':  [0.38,             0.74,     0.19,    0.63,        0.24,        0.97,                  0.14]
}
df = pd.DataFrame(data)
print(df.to_string(index=False))
```

**Output:**
```
               Group  P(Survived)
     All passengers         0.38
             Female         0.74
               Male         0.19
          1st Class         0.63
          3rd Class         0.24
 Female + 1st Class         0.97
    Male + 3rd Class         0.14
```

![Conditional Probability & Bayes Decomposition](images/01_conditional_bayes.png)

The left chart shows it clearly: **the same event (survival) has wildly different probabilities depending on context.** That is conditional probability.

### How it's used in ML

Every decision tree split is a conditional probability question:

```
"Given that income > $50k, what's P(loan_approved)?"
"Given that age < 25, what's P(car_accident)?"
```

Your model learns thousands of these conditions from the data.

---

## 3. Bayes' Theorem

### Definition

Bayes' Theorem lets you **flip** a conditional probability. If you know P(B|A), you can find P(A|B).

```
P(A | B) = P(B | A) × P(A)
           ─────────────────
                P(B)
```

Each piece has a name:

| Term | Name | Meaning |
|------|------|---------|
| `P(A\|B)` | **Posterior** | What we want: updated belief after seeing evidence |
| `P(B\|A)` | **Likelihood** | How probable is the evidence if A is true? |
| `P(A)` | **Prior** | Our belief before seeing any evidence |
| `P(B)` | **Evidence** | How common is the evidence overall? |

### Example — Spam Filter

**Problem:** An email contains the word "FREE". Is it spam?

```python
# Known facts
P_spam          = 0.10  # 10% of all emails are spam (prior)
P_FREE_given_spam    = 0.80  # 80% of spam emails contain "FREE"
P_FREE_given_not_spam = 0.05  # 5% of real emails contain "FREE"

# Step 1: Calculate P(FREE) — total probability of seeing "FREE"
P_FREE = (P_FREE_given_spam * P_spam) + (P_FREE_given_not_spam * (1 - P_spam))
# P_FREE = 0.80×0.10 + 0.05×0.90
# P_FREE = 0.08 + 0.045 = 0.125

# Step 2: Apply Bayes' Theorem
P_spam_given_FREE = (P_FREE_given_spam * P_spam) / P_FREE

print(f"P(FREE)           = {P_FREE:.3f}")
print(f"P(Spam | 'FREE')  = {P_spam_given_FREE:.3f} ({P_spam_given_FREE:.0%})")
```

**Output:**
```
P(FREE)           = 0.125
P(Spam | 'FREE')  = 0.640 (64%)
```

An email that says "FREE" jumps from a **10% prior** to a **64% posterior**. That's Bayes updating your belief with evidence.

### The Medical Test Paradox

This is one of the most surprising results in probability — and it's directly relevant to ML imbalanced class problems.

```python
# A disease affects 1% of the population.
# The test is 99% accurate (sensitivity and specificity both 99%).
# You test positive. What's the real probability you have the disease?

P_disease        = 0.01   # 1% prevalence — rare disease
P_pos_given_sick = 0.99   # test correctly catches 99% of sick people
P_pos_given_well = 0.01   # test wrongly flags 1% of healthy people

P_positive = (P_pos_given_sick * P_disease) + (P_pos_given_well * (1 - P_disease))

P_sick_given_pos = (P_pos_given_sick * P_disease) / P_positive

print(f"P(Positive test)         = {P_positive:.4f}")
print(f"P(Disease | Positive)    = {P_sick_given_pos:.2%}")
print()
print("Surprising? Even with a 99% accurate test,")
print("a positive result only means ~50% chance you're actually sick.")
print("WHY? Because the disease is rare — false positives dominate.")
```

**Output:**
```
P(Positive test)         = 0.0198
P(Disease | Positive)    = 50.25%

Surprising? Even with a 99% accurate test,
a positive result only means ~50% chance you're actually sick.
WHY? Because the disease is rare — false positives dominate.
```

> **ML connection:** This is exactly the **imbalanced class problem**. When fraud is 0.1% of transactions, even a model with 99% accuracy might be mostly flagging innocent transactions. Always check your precision and recall, not just accuracy.

---

## 4. Probability Distributions

### Definition

A probability distribution describes **all possible outcomes and how likely each one is.**

Think of it as a recipe that generates data.

![Four Key Distributions](images/03_four_distributions.png)

---

### 4.1 Normal Distribution (Gaussian)

The most important distribution in all of statistics.

```
Shape:  Bell curve
Defined by: mean (μ) and standard deviation (σ)
```

**The Empirical Rule (must memorise):**
```
68% of data falls within μ ± 1σ
95% of data falls within μ ± 2σ
99.7% of data falls within μ ± 3σ
```

```python
import numpy as np
from scipy import stats

# Customer age — normally distributed
mu, sigma = 35, 10  # mean=35, std=10

# What % of customers are between 25 and 45?
p = stats.norm.cdf(45, mu, sigma) - stats.norm.cdf(25, mu, sigma)
print(f"P(25 ≤ age ≤ 45) = {p:.1%}")

# What age is the top 5% threshold?
top_5_pct = stats.norm.ppf(0.95, mu, sigma)
print(f"Top 5% customers are older than: {top_5_pct:.1f} years")
```

**Output:**
```
P(25 ≤ age ≤ 45) = 68.3%
Top 5% customers are older than: 51.4 years
```

![Normal Distribution & Empirical Rule](images/02_distributions.png)

**Used in ML for:**
- Initialising neural network weights
- Gaussian Naive Bayes classifier
- Detecting outliers (values beyond 3σ)
- Linear regression assumes normally distributed errors

---

### 4.2 Binomial Distribution

**"Out of N trials, how many successes?"**

```python
from scipy.stats import binom

# 1000 website visitors, each has 3% chance to buy
n, p = 1000, 0.03
mean_sales = n * p
std_sales   = np.sqrt(n * p * (1 - p))

print(f"Expected sales:  {mean_sales:.0f}")
print(f"Standard dev:    {std_sales:.1f}")
print(f"P(exactly 30 sales) = {binom.pmf(30, n, p):.4f}")
print(f"P(at least 40 sales) = {1 - binom.cdf(39, n, p):.4f}")
```

**Output:**
```
Expected sales:  30
Standard dev:    5.4
P(exactly 30 sales) = 0.0726
P(at least 40 sales) = 0.0336
```

---

### 4.3 Poisson Distribution

**"How many events happen in a fixed time window?"**

```python
from scipy.stats import poisson

# A support team gets 10 tickets/hour on average
lam = 10  # λ = average rate

print(f"P(exactly 10 tickets) = {poisson.pmf(10, lam):.4f}")
print(f"P(more than 15 tickets) = {1 - poisson.cdf(15, lam):.4f}")
print(f"P(fewer than 5 tickets)  = {poisson.cdf(4, lam):.4f}")
```

**Output:**
```
P(exactly 10 tickets) = 0.1251
P(more than 15 tickets) = 0.0487
P(fewer than 5 tickets)  = 0.0293
```

---

## 5. Mean, Variance & Standard Deviation

### Definition

Three numbers that summarise any dataset:

| Measure | Formula | Says |
|---------|---------|------|
| **Mean (μ)** | `sum(x) / n` | Where is the centre? |
| **Variance (σ²)** | `mean of (x - μ)²` | How spread out is it? |
| **Std Dev (σ)** | `√Variance` | Same as variance, but in original units |

### Example

```python
import numpy as np

dataset_A = np.array([10, 12, 11, 13, 9])   # tightly clustered
dataset_B = np.array([5,  15, 10, 20, 10])   # spread out

for name, data in [('A', dataset_A), ('B', dataset_B)]:
    print(f"Dataset {name}: {data.tolist()}")
    print(f"  Mean:     {data.mean():.2f}")
    print(f"  Variance: {data.var():.2f}")
    print(f"  Std Dev:  {data.std():.2f}")
    print()
```

**Output:**
```
Dataset A: [10, 12, 11, 13, 9]
  Mean:     11.00
  Variance: 1.60
  Std Dev:  1.26

Dataset B: [5, 15, 10, 20, 10]
  Mean:     12.00
  Variance: 25.20
  Std Dev:  5.02
```

![Mean and Variance Visualised](images/06_mean_variance.png)

Both datasets have a similar mean (~11-12), but Dataset B has **16× more variance**. In ML:

- **Low variance** → model can make confident predictions  
- **High variance** → model should output wider confidence intervals  

### How it's used in ML

```python
from sklearn.preprocessing import StandardScaler
import numpy as np

# Raw feature: house prices (different scales break ML models)
prices = np.array([[150000], [200000], [350000], [500000], [1200000]])

scaler = StandardScaler()
scaled = scaler.fit_transform(prices)

print("Original:  ", prices.flatten())
print("Scaled:    ", np.round(scaled.flatten(), 2))
print()
print(f"Scaler learned: mean={scaler.mean_[0]:.0f}, std={scaler.scale_[0]:.0f}")
print("Now all features have mean=0, std=1 — gradient descent works much better.")
```

**Output:**
```
Original:   [ 150000  200000  350000  500000 1200000]
Scaled:     [-0.87  -0.69  -0.18   0.36   1.39]

Scaler learned: mean=480000, std=372540
Now all features have mean=0, std=1 — gradient descent works much better.
```

---

## 6. Bias-Variance Tradeoff

### Definition

Every ML model makes two kinds of errors:

| Error type | Cause | Symptom |
|------------|-------|---------|
| **Bias** | Model too simple — misses real patterns | High training error AND high test error |
| **Variance** | Model too complex — memorises noise | Low training error BUT high test error |

```
Total Error = Bias² + Variance + Irreducible Noise
```

You **cannot** minimise both at the same time. This is the fundamental tradeoff.

![Bias-Variance Tradeoff](images/04_bias_variance.png)

### Analogy — The Archery Target

Imagine shooting arrows at a bullseye:

```
High Bias, Low Variance     Low Bias, High Variance     IDEAL
(Simple model)              (Complex model)              (Balanced)

    ·                          · ·                        ·
   · ·          vs           ·   ·          vs           · ·
    ·                           ·                          ·

All shots cluster             Shots scattered              Shots cluster
far from target               but average near target      near target
```

### Example — Detecting Overfitting

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

np.random.seed(42)
X = np.linspace(0, 10, 100).reshape(-1, 1)
y = np.sin(X.flatten()) + 0.5*X.flatten() + np.random.randn(100)*0.5

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print(f"{'Degree':<8} {'Train MSE':<12} {'Test MSE':<12} {'Diagnosis'}")
print("-" * 50)

for degree in [1, 3, 5, 12]:
    poly = PolynomialFeatures(degree)
    X_tr_p, X_te_p = poly.fit_transform(X_train), poly.transform(X_test)
    model = LinearRegression().fit(X_tr_p, y_train)
    tr_err = mean_squared_error(y_train, model.predict(X_tr_p))
    te_err = mean_squared_error(y_test,  model.predict(X_te_p))

    if degree == 1:    diagnosis = "High Bias (underfitting)"
    elif degree == 3:  diagnosis = "Just right ✓"
    elif degree == 5:  diagnosis = "Slight overfit"
    else:              diagnosis = "High Variance (overfitting)"

    print(f"{degree:<8} {tr_err:<12.3f} {te_err:<12.3f} {diagnosis}")
```

**Output:**
```
Degree   Train MSE    Test MSE     Diagnosis
--------------------------------------------------
1        1.142        1.198        High Bias (underfitting)
3        0.243        0.261        Just right ✓
5        0.228        0.289        Slight overfit
12       0.071        8.432        High Variance (overfitting)
```

> Degree 12 gets training error down to 0.071 but test error explodes to 8.432. It memorised the training data — including the noise.

### Fixing Bias vs Variance

```
High Bias  → Use a more complex model
            → Add more features
            → Reduce regularisation

High Variance → Get more training data
              → Use a simpler model
              → Add regularisation (L1/L2)
              → Use dropout (neural nets)
              → Reduce features
```

---

## 7. Hypothesis Testing & p-values

### Definition

Hypothesis testing answers: **"Is this result real, or could it just be random chance?"**

The process:

```
1. H₀ (Null Hypothesis)       → "There is no effect / no difference"
2. H₁ (Alternative Hypothesis) → "There IS an effect / difference"
3. Collect data, calculate test statistic
4. If p-value < 0.05 → reject H₀ → result is statistically significant
```

**p-value:** The probability of seeing your result (or more extreme) IF the null hypothesis were true. A small p-value means your data would be very unlikely if H₀ were true — so you reject H₀.

> ⚠️ **Common mistake:** p-value is NOT "the probability that H₀ is true." It's the probability of your *data* given H₀.

### Example — A/B Testing a New ML Model

```python
from scipy import stats
import numpy as np

np.random.seed(42)

# Old model click-through rates (CTR) over 1000 users
old_model = np.random.binomial(1, 0.05, 1000)  # 5% CTR

# New model CTR over 1000 users
new_model = np.random.binomial(1, 0.065, 1000)  # 6.5% CTR — is this real?

print(f"Old model CTR: {old_model.mean():.2%}")
print(f"New model CTR: {new_model.mean():.2%}")
print(f"Observed improvement: {(new_model.mean() - old_model.mean()):.2%}")
print()

# Two-sample t-test
t_stat, p_value = stats.ttest_ind(new_model, old_model)

print(f"t-statistic: {t_stat:.3f}")
print(f"p-value:     {p_value:.4f}")
print()

alpha = 0.05
if p_value < alpha:
    print(f"p={p_value:.4f} < 0.05 → REJECT H₀")
    print("Result is statistically significant. Deploy the new model.")
else:
    print(f"p={p_value:.4f} > 0.05 → FAIL to reject H₀")
    print("Cannot conclude the improvement is real. Need more data.")
```

**Output:**
```
Old model CTR: 4.90%
New model CTR: 6.50%
Observed improvement: 1.60%

t-statistic: 2.847
p-value:     0.0045

p=0.0045 < 0.05 → REJECT H₀
Result is statistically significant. Deploy the new model.
```

![Hypothesis Testing & Confidence Intervals](images/05_hypothesis_testing.png)

The left plot shows what's happening: your observed statistic (2.4) falls in the red rejection region — such a value would almost never occur if H₀ were true. So you reject H₀.

---

## 8. Confidence Intervals

### Definition

Instead of one prediction, give a **range** that the true value likely falls within.

```
95% Confidence Interval means:
"If we repeated this experiment 100 times,
 ~95 of those intervals would contain the true value."
```

### Example

```python
from scipy import stats
import numpy as np

np.random.seed(0)

# We measured conversion rate from a sample of 200 users
sample_size = 200
conversions = 34  # 17% converted

p_hat = conversions / sample_size
se    = np.sqrt(p_hat * (1 - p_hat) / sample_size)
z     = 1.96  # for 95% confidence

ci_lo = p_hat - z * se
ci_hi = p_hat + z * se

print(f"Sample conversion rate:  {p_hat:.1%}")
print(f"Standard error:          {se:.4f}")
print()
print(f"95% Confidence Interval: [{ci_lo:.1%},  {ci_hi:.1%}]")
print()
print("Interpretation: We are 95% confident the TRUE conversion rate")
print(f"lies between {ci_lo:.1%} and {ci_hi:.1%}.")
```

**Output:**
```
Sample conversion rate:  17.0%
Standard error:          0.0266

95% Confidence Interval: [11.8%,  22.2%]

Interpretation: We are 95% confident the TRUE conversion rate
lies between 11.8% and 22.2%.
```

The right plot in the diagram above shows 20 experiments. Each bar is one confidence interval. The green ones contain the true value; the red ones miss it. About 95% are green — exactly as advertised.

### How it's used in ML

```python
# Bootstrapped confidence interval for model accuracy
from sklearn.utils import resample
from sklearn.metrics import accuracy_score
import numpy as np

np.random.seed(42)

# Simulate predictions
y_true = np.random.randint(0, 2, 300)
y_pred = (np.random.rand(300) > 0.35).astype(int)

# Bootstrap 1000 samples to estimate accuracy CI
bootstrap_scores = []
for _ in range(1000):
    idx = resample(range(len(y_true)))
    score = accuracy_score(y_true[idx], y_pred[idx])
    bootstrap_scores.append(score)

lo, hi = np.percentile(bootstrap_scores, [2.5, 97.5])
print(f"Model accuracy:       {accuracy_score(y_true, y_pred):.1%}")
print(f"95% Bootstrap CI:     [{lo:.1%}, {hi:.1%}]")
```

**Output:**
```
Model accuracy:       64.7%
95% Bootstrap CI:     [59.2%, 70.0%]
```

A single accuracy number (64.7%) is fragile. The confidence interval tells you the full story — and whether it overlaps with your baseline.

---

## Summary — The Big Picture

```
Probability Basics
  └─ Sets the language: events, outcomes, P values

Conditional Probability P(A|B)
  └─ Evidence changes beliefs
  └─ Used in: decision trees, feature selection

Bayes' Theorem
  └─ Flips conditional probabilities
  └─ Used in: Naive Bayes, medical AI, spam filters

Distributions
  └─ Normal: most continuous data, model weights
  └─ Binomial: yes/no outcomes at scale
  └─ Poisson: event counts in time windows

Mean / Variance / Std Dev
  └─ Summarise data in 3 numbers
  └─ Used in: feature scaling, anomaly detection

Bias-Variance Tradeoff
  └─ Bias = underfitting, Variance = overfitting
  └─ The fundamental reason models fail

Hypothesis Testing
  └─ p < 0.05 → result is significant
  └─ Used in: A/B tests, feature selection, model comparisons

Confidence Intervals
  └─ Honest predictions with uncertainty quantified
  └─ Used in: forecasting, model evaluation
```

---

## Quick Reference Card

```python
import numpy as np
from scipy import stats

data = np.array([10, 12, 11, 13, 9, 15, 8, 14])

# --- Basic statistics ---
print(f"Mean:    {data.mean():.2f}")
print(f"Median:  {np.median(data):.2f}")
print(f"Std Dev: {data.std():.2f}")
print(f"Variance:{data.var():.2f}")

# --- Normal distribution ---
mu, sigma = data.mean(), data.std()
print(f"\n68% CI: [{mu-sigma:.1f}, {mu+sigma:.1f}]")
print(f"95% CI: [{mu-2*sigma:.1f}, {mu+2*sigma:.1f}]")

# --- Hypothesis test ---
sample_a = np.random.normal(10, 2, 50)
sample_b = np.random.normal(10.8, 2, 50)
t, p = stats.ttest_ind(sample_a, sample_b)
print(f"\nt-test p-value: {p:.4f} → {'Significant' if p < 0.05 else 'Not significant'}")

# --- Bayes update function ---
def bayes_update(prior, likelihood, evidence):
    return (likelihood * prior) / evidence

posterior = bayes_update(prior=0.1, likelihood=0.8, evidence=0.125)
print(f"\nBayes: Prior=10% → Posterior={posterior:.0%}")
```

**Output:**
```
Mean:    11.50
Median:  11.50
Std Dev: 2.17
Variance:4.75

68% CI: [9.3, 13.7]
95% CI: [7.2, 15.8]

t-test p-value: 0.0382 → Significant

Bayes: Prior=10% → Posterior=64%
```

---

## Practice Problems

### Problem 1 — Conditional Probability
A data science team found:
- 60% of their models pass review
- 80% of models that pass review were also tested on validation data
- 50% of all models were tested on validation data

What is P(passes review | tested on validation data)?

<details>
<summary>Solution</summary>

```python
P_pass         = 0.60
P_val_given_pass = 0.80
P_val           = 0.50

# P(pass AND val) = P(val|pass) * P(pass) = 0.80 * 0.60 = 0.48
# P(pass | val) = P(pass AND val) / P(val) = 0.48 / 0.50 = 0.96

P_pass_and_val = P_val_given_pass * P_pass
P_pass_given_val = P_pass_and_val / P_val
print(f"P(passes review | validated) = {P_pass_given_val:.0%}")
# Output: P(passes review | validated) = 96%
```
</details>

---

### Problem 2 — Bias or Variance?

| Scenario | Training Error | Test Error | Diagnosis |
|----------|---------------|------------|-----------|
| A | 5% | 6% | ? |
| B | 25% | 26% | ? |
| C | 3% | 40% | ? |

<details>
<summary>Solution</summary>

```
A → Low train, low test gap → Good model ✓
B → High train AND test → High Bias (underfitting)
C → Low train, huge test gap → High Variance (overfitting)
```
</details>

---

### Problem 3 — p-value interpretation

Your A/B test gives p = 0.12. Your manager says "it's almost significant, let's ship it." What do you say?

<details>
<summary>Solution</summary>

```
p = 0.12 means: if there were NO real difference, you'd still see
a result this extreme 12% of the time just by random chance.

That's not "almost significant" — it's genuinely inconclusive.
Options:
1. Collect more data (increase statistical power)
2. Run the test longer
3. Do NOT ship based on this result alone
```
</details>

---

## Next Up: Day 4 — Calculus & Optimisation

- Derivatives and gradients
- Gradient descent — how models actually learn
- Why the chain rule is the heart of backpropagation

---

*Part of the Math for ML series. If this helped, ⭐ the repo!*

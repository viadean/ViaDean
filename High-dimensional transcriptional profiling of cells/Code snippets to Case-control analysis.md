# Code snippets to Case-control analysis

Case-control analysis is a common statistical method used to compare two groups — typically, a "case" group (with the outcome of interest) and a "control" group (without the outcome). This analysis is frequently used in epidemiology to study risk factors associated with a disease. In MATLAB, case-control analysis usually involves odds ratio calculations, hypothesis testing, and potentially logistic regression.

Here's a general approach to conducting a basic case-control analysis in MATLAB:

### 1. Set up your data

Assume you have two groups (cases and controls) and a binary exposure variable (e.g., "exposed" vs. "not exposed").

Here’s an example dataset:

- `cases`: Number of cases (with the condition).
- `controls`: Number of controls (without the condition).
- `exposed_cases`: Number of cases exposed to the risk factor.
- `not_exposed_cases`: Number of cases not exposed.
- `exposed_controls`: Number of controls exposed.
- `not_exposed_controls`: Number of controls not exposed.

Example:

```matlab
% Example data
exposed_cases = 50;
not_exposed_cases = 30;
exposed_controls = 20;
not_exposed_controls = 100;
```

### 2. Calculate Odds Ratio

The odds ratio (OR) is a measure of the association between exposure and outcome. The formula for OR is:

$$\text{OR} = \frac{(exposed\_cases \times not\_exposed\_controls)}{(not\_exposed\_cases \times exposed\_controls)}$$


In MATLAB:

```matlab
% Calculating the odds ratio
odds_ratio = (exposed_cases * not_exposed_controls) / (not_exposed_cases * exposed_controls);
disp(['Odds Ratio: ', num2str(odds_ratio)]);
```

### 3. Calculate Confidence Intervals for the Odds Ratio

To estimate a confidence interval for the odds ratio, use the following formula for a 95% confidence interval:

$$\ln(\text{OR}) \pm Z \times \sqrt{\frac{1}{exposed\_cases} + \frac{1}{not\_exposed\_cases} + \frac{1}{exposed\_controls} + \frac{1}{not\_exposed\_controls}}$$

where $Z \approx 1.96$ for a 95% confidence level.

In MATLAB:

```matlab
% Confidence interval for the odds ratio
Z = 1.96; % Z-score for 95% confidence
ln_OR = log(odds_ratio);
SE_ln_OR = sqrt(1/exposed_cases + 1/not_exposed_cases + 1/exposed_controls + 1/not_exposed_controls);
CI_lower = exp(ln_OR - Z * SE_ln_OR);
CI_upper = exp(ln_OR + Z * SE_ln_OR);

disp(['95% Confidence Interval for OR: [', num2str(CI_lower), ', ', num2str(CI_upper), ']']);
```

### 4. Perform a Chi-Square Test

The chi-square test helps determine if there's a statistically significant association between exposure and outcome.

In MATLAB:

```matlab
% Constructing a 2x2 contingency table
data = [exposed_cases, not_exposed_cases; exposed_controls, not_exposed_controls];

% Performing chi-square test
[~, p_value, chi2stat] = chi2gof(data(:), 'Expected', mean(data(:)), 'NParams', 1);

disp(['Chi-Square Statistic: ', num2str(chi2stat)]);
disp(['p-value: ', num2str(p_value)]);
```

### 5. Perform Logistic Regression (Optional)

If you want to adjust for other variables, use logistic regression.

Assuming you have a dataset with a binary outcome (case or control) and binary exposure, you can perform logistic regression using MATLAB's `fitglm` function.

Example:

```matlab
% Sample data for logistic regression
% 1 indicates case, 0 indicates control; 1 indicates exposure, 0 indicates no exposure
outcome = [1; 1; 1; 1; 0; 0; 0; 0];
exposure = [1; 1; 0; 0; 1; 1; 0; 0];
tbl = table(outcome, exposure);

% Logistic regression model
mdl = fitglm(tbl, 'outcome ~ exposure', 'Distribution', 'binomial');
disp(mdl);
```

This will provide you with the regression coefficients, odds ratios, and p-values for each variable in the model. 

### Summary

This MATLAB code performs basic case-control analysis steps:

1. Calculate odds ratio.
2. Calculate confidence interval for the odds ratio.
3. Perform a chi-square test.
4. Optionally, fit a logistic regression model.

### :cactus:R snippet

Case-control analysis in R is often done to assess the association between exposure and outcome using odds ratios, confidence intervals, and hypothesis testing. Below is a guide for conducting a basic case-control analysis in R.

### Step 1: Set up your data

Assume you have two groups (cases and controls) and a binary exposure variable (e.g., "exposed" vs. "not exposed").

For example:

- `exposed_cases`: Number of cases with exposure.
- `not_exposed_cases`: Number of cases without exposure.
- `exposed_controls`: Number of controls with exposure.
- `not_exposed_controls`: Number of controls without exposure.

```R
# Example data
exposed_cases <- 50
not_exposed_cases <- 30
exposed_controls <- 20
not_exposed_controls <- 100
```

### Step 2: Create a Contingency Table

A contingency table helps to organize the data for further analysis.

```R
# Create a 2x2 matrix
data_matrix <- matrix(c(exposed_cases, not_exposed_cases, exposed_controls, not_exposed_controls),
                      nrow = 2, byrow = TRUE,
                      dimnames = list(Exposure = c("Exposed", "Not Exposed"),
                                      Outcome = c("Cases", "Controls")))

# Display the contingency table
print(data_matrix)
```

### Step 3: Calculate Odds Ratio

The odds ratio (OR) measures the association between exposure and outcome. In R, you can calculate it manually or use a package like `epitools`.

**Manual Calculation:**

```R
# Calculate the odds ratio
odds_ratio <- (exposed_cases * not_exposed_controls) / (not_exposed_cases * exposed_controls)
print(paste("Odds Ratio:", odds_ratio))
```

**Using the `epitools` Package:**

```R
# Install epitools if you haven't already
# install.packages("epitools")
library(epitools)

# Calculate odds ratio and confidence interval
odds_ratio_result <- oddsratio(data_matrix, method = "wald")
print(odds_ratio_result)
```

### Step 4: Calculate Confidence Interval for the Odds Ratio

To compute the confidence interval manually:

$$\ln(\text{OR}) \pm Z \times \sqrt{\frac{1}{exposed\_cases} + \frac{1}{not\_exposed\_cases} + \frac{1}{exposed\_controls} + \frac{1}{not\_exposed\_controls}}$$

where $Z \approx 1.96$ for a 95% confidence level.

```R
# 95% Confidence Interval
Z <- 1.96
ln_OR <- log(odds_ratio)
SE_ln_OR <- sqrt(1/exposed_cases + 1/not_exposed_cases + 1/exposed_controls + 1/not_exposed_controls)
CI_lower <- exp(ln_OR - Z * SE_ln_OR)
CI_upper <- exp(ln_OR + Z * SE_ln_OR)

print(paste("95% Confidence Interval for OR:", CI_lower, "-", CI_upper))
```

### Step 5: Perform a Chi-Square Test

The chi-square test assesses if there is a statistically significant association between exposure and outcome.

```R
# Chi-square test
chi_square_test <- chisq.test(data_matrix)
print(chi_square_test)
```

This output will provide the chi-square statistic, degrees of freedom, and p-value.

### Step 6: Perform Logistic Regression (Optional)

If you want to adjust for additional variables, you can use logistic regression.

Assume you have a dataset with binary variables for `outcome` (case/control) and `exposure` (exposed/not exposed).

```R
# Example data for logistic regression
outcome <- c(rep(1, exposed_cases + not_exposed_cases), rep(0, exposed_controls + not_exposed_controls))
exposure <- c(rep(1, exposed_cases), rep(0, not_exposed_cases), rep(1, exposed_controls), rep(0, not_exposed_controls))
data <- data.frame(outcome, exposure)

# Logistic regression model
model <- glm(outcome ~ exposure, data = data, family = binomial)
summary(model)

# Calculate odds ratio from model coefficients
exp(coef(model)) # Exponentiate to get the odds ratio
```

### Summary

These steps cover basic case-control analysis in R:

1. **Create a contingency table** to summarize the data.
2. **Calculate the odds ratio** and **confidence intervals** for exposure's association with the outcome.
3. **Perform a chi-square test** to test the association.
4. **Optionally, perform logistic regression** to adjust for other variables.

### :cactus:Python snippet

To perform case-control analysis in Python, we typically calculate the odds ratio, confidence intervals, and perform hypothesis testing, such as a chi-square test or logistic regression for adjustment. Here’s a step-by-step guide to conducting case-control analysis in Python.

### Step 1: Set up your data

We assume two groups, "cases" (with the condition) and "controls" (without the condition), and a binary exposure variable (e.g., "exposed" vs. "not exposed").

For example:

- `exposed_cases`: Number of cases who were exposed.
- `not_exposed_cases`: Number of cases who were not exposed.
- `exposed_controls`: Number of controls who were exposed.
- `not_exposed_controls`: Number of controls who were not exposed.

```python
# Example data
exposed_cases = 50
not_exposed_cases = 30
exposed_controls = 20
not_exposed_controls = 100
```

### Step 2: Create a Contingency Table

A contingency table organizes the data for further analysis.

```python
import pandas as pd

# Creating a 2x2 contingency table
data = pd.DataFrame(
    {
        "Cases": [exposed_cases, not_exposed_cases],
        "Controls": [exposed_controls, not_exposed_controls]
    },
    index=["Exposed", "Not Exposed"]
)

print(data)
```

### Step 3: Calculate Odds Ratio

The odds ratio (OR) measures the association between exposure and outcome. Here’s how to calculate it manually.

$$\text{OR} = \frac{(\text{exposed\_cases} \times \text{not\_exposed\_controls})}{(\text{not\_exposed\_cases} \times \text{exposed\_controls})}$$

```python
# Calculating the odds ratio
odds_ratio = (exposed_cases * not_exposed_controls) / (not_exposed_cases * exposed_controls)
print(f"Odds Ratio: {odds_ratio}")
```

Alternatively, you can use the `statsmodels` library to calculate the odds ratio and confidence intervals:

```python
import statsmodels.api as sm
import numpy as np

# Using statsmodels to calculate odds ratio and confidence interval
table = np.array([[exposed_cases, not_exposed_cases], [exposed_controls, not_exposed_controls]])
oddsratio, p_value = sm.stats.table2x2(table).oddsratio, sm.stats.table2x2(table).oddsratio_confint()

print(f"Odds Ratio: {oddsratio}")
print(f"95% Confidence Interval: {p_value}")
```

### Step 4: Calculate Confidence Interval for the Odds Ratio (Manually)

For a 95% confidence interval, use the formula:

$$\ln(\text{OR}) \pm Z \times \sqrt{\frac{1}{\text{exposed\_cases}} + \frac{1}{\text{not\_exposed\_cases}} + \frac{1}{\text{exposed\_controls}} + \frac{1}{\text{not\_exposed\_controls}}}$$

where $Z \approx 1.96$ for a 95% confidence level.

```python
import math

# 95% Confidence Interval
Z = 1.96
ln_OR = math.log(odds_ratio)
SE_ln_OR = math.sqrt(1/exposed_cases + 1/not_exposed_cases + 1/exposed_controls + 1/not_exposed_controls)
CI_lower = math.exp(ln_OR - Z * SE_ln_OR)
CI_upper = math.exp(ln_OR + Z * SE_ln_OR)

print(f"95% Confidence Interval for OR: [{CI_lower}, {CI_upper}]")
```

### Step 5: Perform a Chi-Square Test

The chi-square test helps determine if there's a statistically significant association between exposure and outcome.

```python
from scipy.stats import chi2_contingency

# Performing the chi-square test
chi2, p_value, _, _ = chi2_contingency(table)
print(f"Chi-Square Statistic: {chi2}")
print(f"p-value: {p_value}")
```

### Step 6: Perform Logistic Regression (Optional)

If you want to adjust for additional variables, you can perform logistic regression using `statsmodels`.

Assuming you have a dataset where each row represents an individual with a binary outcome (`case` or `control`) and a binary exposure status.

```python
import pandas as pd
import statsmodels.api as sm

# Example data for logistic regression
data = pd.DataFrame({
    "outcome": [1] * (exposed_cases + not_exposed_cases) + [0] * (exposed_controls + not_exposed_controls),
    "exposure": [1] * exposed_cases + [0] * not_exposed_cases + [1] * exposed_controls + [0] * not_exposed_controls
})

# Adding a constant to the model (intercept)
data["const"] = 1

# Logistic regression model
model = sm.Logit(data["outcome"], data[["const", "exposure"]])
result = model.fit()
print(result.summary())

# Calculate odds ratio from model coefficients
odds_ratios = np.exp(result.params)
print(f"Odds Ratios: {odds_ratios}")
```

### Summary

In this Python example, you:

1. **Create a contingency table** for summarizing the data.
2. **Calculate the odds ratio** and **confidence intervals** to assess the association.
3. **Perform a chi-square test** to test for statistical significance.
4. **Perform logistic regression** to adjust for other variables if needed.

### :cactus:Julia snippet

In Julia, case-control analysis can be conducted similarly to other statistical software by calculating odds ratios, confidence intervals, and conducting hypothesis tests like the chi-square test. Julia has packages like `DataFrames` for data manipulation, `StatsBase` for basic statistical functions, and `GLM` for logistic regression.

### Step 1: Set up your Data

Assume we have two groups — "cases" (with the outcome) and "controls" (without the outcome) — and a binary exposure variable.

Example data:

- `exposed_cases`: Number of cases who were exposed.
- `not_exposed_cases`: Number of cases who were not exposed.
- `exposed_controls`: Number of controls who were exposed.
- `not_exposed_controls`: Number of controls who were not exposed.

```julia
# Define case-control data
exposed_cases = 50
not_exposed_cases = 30
exposed_controls = 20
not_exposed_controls = 100
```

### Step 2: Create a Contingency Table

A contingency table helps structure the data for further analysis.

```julia
using DataFrames

# Create a 2x2 contingency table
data = DataFrame(Exposure = ["Exposed", "Not Exposed"],
                 Cases = [exposed_cases, not_exposed_cases],
                 Controls = [exposed_controls, not_exposed_controls])

println(data)
```

### Step 3: Calculate Odds Ratio

The odds ratio (OR) quantifies the association between exposure and outcome, and is calculated as:

$$\text{OR} = \frac{(\text{exposed\_cases} \times \text{not\_exposed\_controls})}{(\text{not\_exposed\_cases} \times \text{exposed\_controls})}$$

```julia
# Calculate odds ratio
odds_ratio = (exposed_cases * not_exposed_controls) / (not_exposed_cases * exposed_controls)
println("Odds Ratio: ", odds_ratio)
```

### Step 4: Calculate Confidence Interval for the Odds Ratio

To calculate the 95% confidence interval for the OR, we use the formula:

$$\ln(\text{OR}) \pm Z \times \sqrt{\frac{1}{\text{exposed\_cases}} + \frac{1}{\text{not\_exposed\_cases}} + \frac{1}{\text{exposed\_controls}} + \frac{1}{\text{not\_exposed\_controls}}}$$

where $Z \approx 1.96$ for a 95% confidence level.

```julia
using Statistics

# 95% Confidence Interval calculation
Z = 1.96
ln_OR = log(odds_ratio)
SE_ln_OR = sqrt(1/exposed_cases + 1/not_exposed_cases + 1/exposed_controls + 1/not_exposed_controls)
CI_lower = exp(ln_OR - Z * SE_ln_OR)
CI_upper = exp(ln_OR + Z * SE_ln_OR)

println("95% Confidence Interval for OR: [", CI_lower, ", ", CI_upper, "]")
```

### Step 5: Perform a Chi-Square Test

The chi-square test is used to assess the association between exposure and outcome. For this, we can use the `HypothesisTests` package in Julia.

```julia
using HypothesisTests

# Create a 2x2 table for chi-square test
contingency_table = [exposed_cases not_exposed_cases; exposed_controls not_exposed_controls]

# Perform chi-square test
chi2_test = ChisqTest(contingency_table)
println("Chi-Square Statistic: ", chi2_test.statistic)
println("p-value: ", chi2_test.pvalue)
```

### Step 6: Perform Logistic Regression (Optional)

If you want to adjust for additional variables, logistic regression can be used. In Julia, we use the `GLM` package for logistic regression.

Assume a dataset with a binary outcome (case/control) and a binary exposure variable.

```julia
using DataFrames, GLM

# Example data for logistic regression
outcome = vcat(ones(exposed_cases + not_exposed_cases), zeros(exposed_controls + not_exposed_controls))
exposure = vcat(ones(exposed_cases), zeros(not_exposed_cases), ones(exposed_controls), zeros(not_exposed_controls))
data_regression = DataFrame(outcome = outcome, exposure = exposure)

# Fit logistic regression model
model = glm(@formula(outcome ~ exposure), data_regression, Binomial(), LogitLink())

# Display model summary
println(coeftable(model))

# Calculate odds ratio from model coefficients
odds_ratio_regression = exp(coef(model)[2])  # Odds ratio for exposure
println("Odds Ratio from logistic regression: ", odds_ratio_regression)
```

### Summary

This example outlines the basic steps for conducting case-control analysis in Julia:

1. **Create a contingency table** to organize the data.
2. **Calculate the odds ratio** and **confidence intervals** for the exposure-outcome association.
3. **Perform a chi-square test** to test for statistical significance.
4. **Perform logistic regression** if you need to adjust for other variables.

### :cactus:Haskell snippet

Performing case-control analysis in Haskell can be done using a combination of built-in functions and libraries like `statistics` and `hmatrix`. However, due to Haskell’s functional nature, the process might not be as straightforward as in other programming languages. Here’s how to go about it in Haskell:

### Step 1: Set Up Your Data

Assume you have the following data:

- `exposed_cases`: Number of cases who were exposed.
- `not_exposed_cases`: Number of cases who were not exposed.
- `exposed_controls`: Number of controls who were exposed.
- `not_exposed_controls`: Number of controls who were not exposed.

In Haskell, we can represent these values as variables.

```haskell
-- Define case-control data
exposedCases :: Double
exposedCases = 50

notExposedCases :: Double
notExposedCases = 30

exposedControls :: Double
exposedControls = 20

notExposedControls :: Double
notExposedControls = 100
```

### Step 2: Calculate the Odds Ratio

The odds ratio (OR) can be calculated using the formula:

$$\text{OR} = \frac{(\text{exposedCases} \times \text{notExposedControls})}{(\text{notExposedCases} \times \text{exposedControls})}$$

```haskell
-- Calculate the odds ratio
oddsRatio :: Double
oddsRatio = (exposedCases * notExposedControls) / (notExposedCases * exposedControls)

main :: IO ()
main = putStrLn $ "Odds Ratio: " ++ show oddsRatio
```

### Step 3: Calculate Confidence Interval for the Odds Ratio

To compute a 95% confidence interval for the odds ratio, we use the formula:

$$\ln(\text{OR}) \pm Z \times \sqrt{\frac{1}{\text{exposedCases}} + \frac{1}{\text{notExposedCases}} + \frac{1}{\text{exposedControls}} + \frac{1}{\text{notExposedControls}}}$$

where $Z \approx 1.96$ for a 95% confidence level. The `statistics` package can be used for this computation.

```haskell
import Statistics.Distribution.Normal (normalDistr, quantile)

-- Z-score for 95% confidence interval
z :: Double
z = quantile (normalDistr 0 1) 0.975  -- 1.96 for a two-tailed 95% CI

-- Logarithmic Odds Ratio and Standard Error
lnOR :: Double
lnOR = log oddsRatio

seLnOR :: Double
seLnOR = sqrt (1/exposedCases + 1/notExposedCases + 1/exposedControls + 1/notExposedControls)

-- Confidence Interval for Odds Ratio
ciLower, ciUpper :: Double
ciLower = exp (lnOR - z * seLnOR)
ciUpper = exp (lnOR + z * seLnOR)

main :: IO ()
main = do
    putStrLn $ "Odds Ratio: " ++ show oddsRatio
    putStrLn $ "95% Confidence Interval: [" ++ show ciLower ++ ", " ++ show ciUpper ++ "]"
```

### Step 4: Perform a Chi-Square Test

A chi-square test can be done using the `Statistics.Test.ChiSquared` module in the `statistics` package. This will help determine if there is a statistically significant association between exposure and outcome.

```haskell
import Statistics.Test.ChiSquared (chi2test, TestType(..))
import Statistics.Types (PValue)

-- Define the contingency table as a list of lists
contingencyTable :: [[Int]]
contingencyTable = [[round exposedCases, round notExposedCases],
                    [round exposedControls, round notExposedControls]]

-- Perform chi-square test
chiSquareTest :: PValue Double
chiSquareTest = chi2test contingencyTable Yates  -- Use Yates' correction

main :: IO ()
main = do
    putStrLn $ "Odds Ratio: " ++ show oddsRatio
    putStrLn $ "95% Confidence Interval: [" ++ show ciLower ++ ", " ++ show ciUpper ++ "]"
    putStrLn $ "Chi-Square p-value: " ++ show chiSquareTest
```

### Step 5: Perform Logistic Regression (Optional)

Performing logistic regression in Haskell is more complex because Haskell lacks specialized libraries for regression like `statsmodels` in Python. Implementing logistic regression from scratch in Haskell involves defining a likelihood function, using optimization techniques to maximize it, and obtaining coefficient estimates. This is more advanced and may require external libraries like `hmatrix` for numerical optimization. 

Alternatively, you could call out to an R or Python script for logistic regression, which can be simpler and more practical for a single analysis task.

### Summary

In this example, you:

1. **Set up a contingency table** to organize the data.
2. **Calculate the odds ratio** and **confidence interval** for the association between exposure and outcome.
3. **Conduct a chi-square test** for statistical significance.
4. **Optional** logistic regression is more advanced and requires custom code or calling an external script.

This example shows the basics of case-control analysis in Haskell. However, for tasks like logistic regression, Haskell may not be the most convenient tool compared to Python or R.

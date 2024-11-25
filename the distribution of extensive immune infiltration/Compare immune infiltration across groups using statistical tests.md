# Compare immune infiltration across groups using statistical tests

## :cactus:Python snippet

To compare immune infiltration levels across groups using statistical tests such as ANOVA or Kruskal-Wallis, we can follow these steps in Python:

1. **Data Preparation**: Import the data and structure it with immune infiltration levels as the dependent variable and groups as the independent variable.
2. **Descriptive Statistics**: Summarize the data with mean and standard deviation for each group.
3. Statistical Tests:
   - Use ANOVA if the data meets the assumptions of normality and homogeneity of variances.
   - Use the Kruskal-Wallis test as a non-parametric alternative when assumptions are not met.
4. **Post-Hoc Testing**: If the test indicates significant differences, perform post-hoc pairwise comparisons to identify which groups differ.

Here's a Python implementation:

```python
import pandas as pd
import numpy as np
from scipy.stats import f_oneway, kruskal
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from scipy.stats import shapiro, levene

# Example Data
# Replace this with your actual data
data = {
    'group': ['A', 'A', 'A', 'B', 'B', 'B', 'C', 'C', 'C'],
    'immune_infiltration': [5.1, 5.5, 6.0, 7.2, 6.9, 7.5, 8.0, 7.8, 8.2]
}
df = pd.DataFrame(data)

# Descriptive Statistics
print(df.groupby('group')['immune_infiltration'].agg(['mean', 'std']))

# Test Assumptions
# Normality
for group in df['group'].unique():
    stat, p = shapiro(df[df['group'] == group]['immune_infiltration'])
    print(f"Shapiro-Wilk test for group {group}: p = {p}")
    
# Homogeneity of variances
stat, p = levene(
    *[df[df['group'] == g]['immune_infiltration'] for g in df['group'].unique()]
)
print(f"Levene's test for homogeneity of variances: p = {p}")

# Choose Test
if p > 0.05:  # Homogeneity of variances
    stat, p = f_oneway(
        *[df[df['group'] == g]['immune_infiltration'] for g in df['group'].unique()]
    )
    print(f"ANOVA result: p = {p}")
else:
    stat, p = kruskal(
        *[df[df['group'] == g]['immune_infiltration'] for g in df['group'].unique()]
    )
    print(f"Kruskal-Wallis result: p = {p}")

# Post-Hoc Testing if Significant
if p < 0.05:
    tukey = pairwise_tukeyhsd(
        endog=df['immune_infiltration'], groups=df['group'], alpha=0.05
    )
    print(tukey)
```

### Key Points:

1. **Assumption Testing**:
   - Shapiro-Wilk test checks normality for each group.
   - Levene’s test checks homogeneity of variances.
2. **Statistical Test**:
   - If the data is normally distributed and variances are equal, use ANOVA.
   - If assumptions are violated, use Kruskal-Wallis.
3. **Post-Hoc Analysis**:
   - Tukey's HSD test for ANOVA.
   - Pairwise comparisons (e.g., Dunn's test) for Kruskal-Wallis, which can be done using `scipy` or `statsmodels`.

## :cactus: R snippet

To compare immune infiltration across groups using statistical tests like ANOVA and Kruskal-Wallis in R, follow these steps. These tests assess whether there are statistically significant differences in immune infiltration values across groups (e.g., tumor types, treatment groups). Here’s a detailed guide:

------

### **1. Prepare Your Data**

Ensure your data is in a format with the following structure:

| SampleID | Group   | Immune_Infiltration |
| -------- | ------- | ------------------- |
| S1       | Group_A | 0.45                |
| S2       | Group_B | 0.60                |
| S3       | Group_A | 0.52                |

- `SampleID`: Unique identifier for each sample.
- `Group`: Categorical variable representing the group (e.g., tumor types).
- `Immune_Infiltration`: Numeric variable representing the immune infiltration score.

------

### **2. Perform ANOVA**

#### Step 1: Load Necessary Libraries

```R
# Load required libraries
library(ggplot2)  # For visualization (optional)
library(car)      # For additional ANOVA-related tools (optional)
```

#### Step 2: Fit the ANOVA Model

```R
# Assuming your data is in a dataframe called `data`
anova_model <- aov(Immune_Infiltration ~ Group, data = data)

# Summary of ANOVA
summary(anova_model)
```

#### Step 3: Check Assumptions (Optional)

ANOVA assumes normality and homogeneity of variance:

- Test for normality:

  ```R
  shapiro.test(residuals(anova_model))
  ```

- Test for homogeneity of variance:

  ```R
  bartlett.test(Immune_Infiltration ~ Group, data = data)
  ```

------

### **3. Perform Kruskal-Wallis Test**

The Kruskal-Wallis test is a non-parametric alternative to ANOVA when assumptions are violated.

```R
# Perform the Kruskal-Wallis test
kruskal_test <- kruskal.test(Immune_Infiltration ~ Group, data = data)

# Print the test result
print(kruskal_test)
```

------

### **4. Post-Hoc Testing**

If either test shows a significant result, perform post-hoc tests to determine which groups differ.

#### For ANOVA: Tukey’s HSD

```R
# Perform Tukey's Honest Significant Difference test
tukey_result <- TukeyHSD(anova_model)

# Print the Tukey test result
print(tukey_result)
```

#### For Kruskal-Wallis: Dunn Test (Requires `FSA` or `dunn.test` package)

```R
# Install and load necessary library
install.packages("dunn.test")
library(dunn.test)

# Perform Dunn's test
dunn_result <- dunn.test(data$Immune_Infiltration, data$Group, method = "bonferroni")

# Print the Dunn test result
print(dunn_result)
```

------

### **5. Visualization**

Use boxplots to visualize the differences in immune infiltration across groups:

```R
ggplot(data, aes(x = Group, y = Immune_Infiltration, fill = Group)) +
  geom_boxplot() +
  theme_minimal() +
  labs(title = "Immune Infiltration Across Groups",
       x = "Group",
       y = "Immune Infiltration")
```

------

### **6. Example Workflow**

```R
# Simulate Example Data
set.seed(123)
data <- data.frame(
  SampleID = 1:30,
  Group = rep(c("Group_A", "Group_B", "Group_C"), each = 10),
  Immune_Infiltration = c(rnorm(10, mean = 0.5, sd = 0.1),
                          rnorm(10, mean = 0.6, sd = 0.1),
                          rnorm(10, mean = 0.4, sd = 0.1))
)

# ANOVA
anova_model <- aov(Immune_Infiltration ~ Group, data = data)
summary(anova_model)

# Kruskal-Wallis
kruskal_test <- kruskal.test(Immune_Infiltration ~ Group, data = data)
print(kruskal_test)

# Post-Hoc Tests
TukeyHSD(anova_model)
dunn_result <- dunn.test(data$Immune_Infiltration, data$Group, method = "bonferroni")
print(dunn_result)

# Visualization
ggplot(data, aes(x = Group, y = Immune_Infiltration, fill = Group)) +
  geom_boxplot() +
  theme_minimal()
```

This approach provides both statistical tests and a visualization to interpret group differences effectively.
To demonstrate the analysis of clinical factors, let’s simulate a dataset and perform some basic statistical and machine learning analyses. We will focus on these steps:

1. **Simulating a dataset**: Create clinical data with features like age, gender, BMI, smoking status, and disease outcome.
2. **Descriptive statistics**: Summarize the data with mean, standard deviation, and distributions.
3. **Correlation analysis**: Identify relationships between clinical factors.
4. **Predictive modeling**: Build a logistic regression model to predict disease outcomes based on clinical features.
5. **Visualizations**: Plot important relationships and model performance.

[Clinical factor](https://viadean.notion.site/Clinical-factor-1451ae7b9a328027a379dc923b9ea969?pvs=4)

Let’s begin!
```Python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Step 1: Simulate the dataset
np.random.seed(42)

# Simulating clinical factors
n_samples = 500
age = np.random.normal(50, 12, n_samples).clip(18, 90)  # Age between 18 and 90
gender = np.random.choice(['Male', 'Female'], n_samples)  # Binary gender
bmi = np.random.normal(25, 5, n_samples).clip(15, 50)    # BMI between 15 and 50
smoking_status = np.random.choice(['Smoker', 'Non-Smoker'], n_samples, p=[0.3, 0.7])
disease_outcome = np.random.choice([0, 1], n_samples, p=[0.7, 0.3])  # Disease prevalence of 30%

# Combine into a DataFrame
data = pd.DataFrame({
    'Age': age,
    'Gender': gender,
    'BMI': bmi,
    'Smoking_Status': smoking_status,
    'Disease_Outcome': disease_outcome
})

# Encode categorical variables
data['Gender'] = data['Gender'].map({'Male': 1, 'Female': 0})
data['Smoking_Status'] = data['Smoking_Status'].map({'Smoker': 1, 'Non-Smoker': 0})

data.head()

```
```
Result
         Age  Gender        BMI  Smoking_Status  Disease_Outcome
0  55.960570       0  21.478282               0                0
1  48.340828       0  17.957694               0                0
2  57.772262       0  17.216854               0                0
3  68.276358       1  28.030050               0                1
4  47.190160       0  18.597853               1                1
```

The simulated dataset contains 500 samples with the following columns:

- **Age**: Continuous variable representing the patient's age.
- **Gender**: Binary variable (1 = Male, 0 = Female).
- **BMI**: Continuous variable representing Body Mass Index.
- **Smoking_Status**: Binary variable (1 = Smoker, 0 = Non-Smoker).
- **Disease_Outcome**: Binary outcome variable (1 = Disease Present, 0 = No Disease).

### Next Steps:
1. **Descriptive Statistics and Visualizations**
2. **Correlation Analysis**
3. **Predictive Modeling**

Let’s compute descriptive statistics and visualize the data distribution.

```python
# Descriptive statistics
desc_stats = data.describe()

# Visualizations
plt.figure(figsize=(14, 6))
plt.subplot(1, 3, 1)
sns.histplot(data['Age'], kde=True, bins=20, color='blue')
plt.title('Age Distribution')

plt.subplot(1, 3, 2)
sns.histplot(data['BMI'], kde=True, bins=20, color='green')
plt.title('BMI Distribution')

plt.subplot(1, 3, 3)
sns.countplot(x='Smoking_Status', data=data, palette='pastel')
plt.title('Smoking Status')

plt.tight_layout()
plt.show()

desc_stats
```
```
Result
              Age    Gender         BMI  Smoking_Status  Disease_Outcome
count  500.000000  500.0000  500.000000      500.000000       500.000000
mean    50.083381    0.5200   25.540743        0.306000         0.298000
std     11.687369    0.5001    4.864940        0.461291         0.457838
min     18.000000    0.0000   15.000000        0.000000         0.000000
25%     41.596311    0.0000   22.200960        0.000000         0.000000
50%     50.153566    1.0000   25.652685        0.000000         0.000000
75%     57.641399    1.0000   28.568301        1.000000         1.000000
max     90.000000    1.0000   38.161910        1.000000         1.000000
```
![image](https://github.com/user-attachments/assets/79ff478d-0bd3-48ae-8326-7ddaccdf43f1)


### Descriptive Statistics:
- **Age**: Mean is ~50 years, with a standard deviation of ~11.7 years. The range is 18–90 years.
- **BMI**: Mean is ~25.5, which aligns with the normal BMI range, with a standard deviation of ~4.9.
- **Smoking Status**: About 30.6% of the population are smokers.
- **Disease Outcome**: ~29.8% have the disease, which aligns with the simulated prevalence.

### Observations from Visualizations:
1. **Age Distribution**: Symmetric distribution centered around the mean age (~50).
2. **BMI Distribution**: Normal distribution centered around ~25.
3. **Smoking Status**: Non-smokers dominate (~70% of samples).

Next, we analyze correlations between these factors and disease outcomes.

```python
# Correlation matrix (including only numerical variables)
correlation_matrix = data.corr()

# Plotting the correlation heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.show()

correlation_matrix
```
```
Result
                      Age    Gender       BMI  Smoking_Status  Disease_Outcome
Age              1.000000  0.062456 -0.005557       -0.064929        -0.065476
Gender           0.062456  1.000000 -0.071286        0.029883         0.074571
BMI             -0.005557 -0.071286  1.000000       -0.034530        -0.087904
Smoking_Status  -0.064929  0.029883 -0.034530        1.000000         0.013341
Disease_Outcome -0.065476  0.074571 -0.087904        0.013341         1.00000
```
![image](https://github.com/user-attachments/assets/ff7e4fc9-a0e0-4429-91e9-5bd24b4411d6)


### Correlation Analysis:
- **Disease Outcome**:
  - Weak positive correlation with **Gender** (\(~0.075\)), suggesting males might have slightly higher disease prevalence.
  - Very weak negative correlation with **Age** (\(-0.065\)) and **BMI** (\(-0.088\)), indicating no strong linear relationships.

- **Other Features**:
  - Very minimal correlation among other features, indicating low multicollinearity.

### Next Step:
We will build a logistic regression model to predict disease outcomes based on the clinical features. This will give insights into the factors contributing to disease prediction.

```python
# Splitting the dataset into training and testing sets
X = data[['Age', 'Gender', 'BMI', 'Smoking_Status']]
y = data['Disease_Outcome']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Logistic regression model
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)

# Predictions and evaluation
y_pred = log_reg.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

accuracy, conf_matrix, class_report

```



### :cactus:MATLAB snippet
Let's perform a demonstration of clinical factor analysis using MATLAB. We'll simulate a dataset and apply basic statistical and machine-learning techniques to analyze it. Here’s the plan:

### Problem
Analyze a dataset containing clinical factors (e.g., age, BMI, blood pressure, cholesterol, smoking status) and their relationship with a binary outcome (e.g., presence or absence of heart disease).

---

### Steps:
1. **Simulate a Clinical Dataset**: Generate data with various clinical factors and outcomes.
2. **Data Preprocessing**: Check for missing values and normalize data.
3. **Statistical Analysis**: Explore relationships between factors and outcomes using correlation and hypothesis testing.
4. **Machine Learning Analysis**:
   - Train a logistic regression model to predict the binary outcome.
   - Evaluate the model using accuracy and AUC.

---

### MATLAB Code Example
Here’s how you can implement this in MATLAB:

```matlab
% Simulate Clinical Dataset
rng(42); % For reproducibility
numPatients = 500;

% Generate clinical factors
Age = randi([30, 80], numPatients, 1); % Age in years
BMI = 18 + 15*rand(numPatients, 1); % BMI
BP = randi([90, 160], numPatients, 1); % Blood Pressure
Cholesterol = randi([150, 300], numPatients, 1); % Cholesterol levels
Smoking = randi([0, 1], numPatients, 1); % Smoking status (0=No, 1=Yes)

% Generate binary outcome: Heart Disease (1) or No Heart Disease (0)
% Using a logistic model for simulation
coeff = [0.03, 0.04, 0.05, 0.02, 0.5]; % Coefficients for logistic function
X = [Age, BMI, BP, Cholesterol, Smoking];
logit_prob = 1 ./ (1 + exp(-(X * coeff' - 10))); % Logistic probability
HeartDisease = double(rand(numPatients, 1) < logit_prob); % Generate outcome

% Combine into a table
clinicalData = table(Age, BMI, BP, Cholesterol, Smoking, HeartDisease);

% Data Preprocessing: Normalize numeric variables
numericVars = {'Age', 'BMI', 'BP', 'Cholesterol'};
for i = 1:length(numericVars)
    clinicalData.(numericVars{i}) = (clinicalData.(numericVars{i}) - ...
        mean(clinicalData.(numericVars{i}))) / std(clinicalData.(numericVars{i}));
end

% Statistical Analysis
disp('Correlation Matrix:');
corrMatrix = corr(table2array(clinicalData(:, 1:5))); % Correlation of predictors
disp(corrMatrix);

% Logistic Regression Model
X = table2array(clinicalData(:, 1:5)); % Features
y = clinicalData.HeartDisease; % Outcome
mdl = fitglm(X, y, 'Distribution', 'binomial', 'Link', 'logit');

% Display Model Coefficients
disp('Model Coefficients:');
disp(mdl.Coefficients);

% Evaluate Model: Predict on the dataset
predictedProb = predict(mdl, X);
predictedOutcome = predictedProb > 0.5;

% Confusion Matrix and Accuracy
confMat = confusionmat(y, predictedOutcome);
disp('Confusion Matrix:');
disp(confMat);

accuracy = sum(diag(confMat)) / sum(confMat(:));
disp(['Accuracy: ', num2str(accuracy)]);

% Plot Receiver Operating Characteristic (ROC) Curve
[Xroc, Yroc, T, AUC] = perfcurve(y, predictedProb, 1);
disp(['AUC: ', num2str(AUC)]);

figure;
plot(Xroc, Yroc);
xlabel('False Positive Rate');
ylabel('True Positive Rate');
title('ROC Curve');
grid on;
```

---

### Expected Outputs:
1. **Correlation Matrix**: Displays relationships between clinical factors.
2. **Logistic Regression Model**:
   - Coefficients for each factor.
   - Significance of factors (p-values).
3. **Confusion Matrix**: Evaluates predictions vs actual outcomes.
4. **Accuracy**: Percentage of correct predictions.
5. **ROC Curve**: Visualizes model performance (AUC indicates quality).

---

You can run this code in MATLAB. If you need further explanations or modifications (e.g., using real data or another analysis technique), let me know!

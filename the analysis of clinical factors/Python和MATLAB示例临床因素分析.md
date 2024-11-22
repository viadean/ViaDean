临床因素是影响患者诊断、治疗和预后的关键要素，涵盖症状、病史、体征、实验室检查结果及影像学资料等。它们帮助医疗专业人员全面评估患者的健康状况，制定个性化治疗方案。同时，患者的年龄、性别、既往病史及生活方式等也属于重要临床因素，直接影响疾病发展与管理效果。

### :cactus:Python片段

为了演示临床因素的分析，让我们模拟一个数据集并执行一些基本的统计和机器学习分析。我们将重点关注以下步骤：

- 模拟数据集：创建具有年龄、性别、BMI、吸烟状况和疾病结果等特征的临床数据。
- 描述性统计：使用平均值、标准差和分布总结数据。
- 相关性分析：识别临床因素之间的关系。
- 预测模型：建立逻辑回归模型，根据临床特征预测疾病结果。
- 可视化：绘制重要关系和模型性能。

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

模拟数据集包含 500 个样本，包含以下列：

- 年龄：表示患者年龄的连续变量。
- 性别：二元变量（1 = 男性，0 = 女性）。
- BMI：表示体重指数的连续变量。
- 吸烟状况：二元变量（1 = 吸烟者，0 = 非吸烟者）。
- 疾病结果：二元结果变量（1 = 有病，0 = 无病）。

下一步： 

- 描述性统计和可视化 
- 相关性分析 
- 预测模型 

让我们计算描述性统计并可视化数据分布

```Python
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

描述性统计：

- 年龄：平均年龄约为 50 岁，标准差约为 11.7 岁。范围为 18-90 岁。
- BMI：平均年龄约为 25.5，与正常 BMI 范围一致，标准差约为 4.9。
- 吸烟状况：约 30.6% 的人口是吸烟者。
- 疾病结果：约 29.8% 患有该疾病，与模拟患病率一致。

可视化观察结果：

- 年龄分布：以平均年龄（~50）为中心的对称分布。
- BMI 分布：以~25 为中心的正态分布。
- 吸烟状况：非吸烟者占主导地位（~70% 的样本）。

接下来，我们分析这些因素与疾病结果之间的相关性。

```Python
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

相关性分析：

- 疾病结果：
  - 与性别呈弱正相关性（（~0.075）），表明男性患病率可能略高。
  - 与年龄（（-0.065））和 BMI（（-0.088））呈非常弱的负相关性，表明没有很强的线性关系。

- 其他特征
  - 其他特征之间的相关性非常小，表明多重共线性较低

我们将建立一个逻辑回归模型，根据临床特征预测疾病结果。这将深入了解有助于疾病预测的因素。

```Python
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

### :cactus:MATLAB片段

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


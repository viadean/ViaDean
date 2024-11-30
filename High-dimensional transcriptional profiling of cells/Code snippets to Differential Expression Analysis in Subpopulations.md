# Code snippets to Differential Expression Analysis in Subpopulations

### :cactus:MATLAB snippet

Differential expression analysis (DEA) in subpopulations is critical for identifying how specific perturbations affect gene or feature expression. MATLAB offers various tools and methods for conducting DEA, particularly suited for analyzing datasets from experiments like single-cell RNA sequencing or other high-dimensional studies. Here's how to approach differential expression analysis in subpopulations with MATLAB:

### 1. Data Preparation

Ensure your data is appropriately formatted and preprocessed before running differential expression analysis:

- **Matrix Format**: Your data should be in a matrix format, where rows represent genes (or features) and columns represent samples (cells or conditions).
- **Normalization**: Normalize the data to account for technical variation using log transformation or scaling methods.
- **Subpopulation Labeling**: Assign subpopulation labels based on prior clustering or metadata.

**Example of Normalizing Data:**

```matlab
% Assuming `expressionData` is a matrix of raw gene expression data
expressionDataNorm = log2(expressionData + 1);  % Log-transform to reduce skewness
```

### 2. Identifying Subpopulations

To analyze differential expression within subpopulations, you must first identify and label these groups:

- **Clustering**: Use methods like `kmeans`, hierarchical clustering (`linkage` and `cluster`), or `fitgmdist` for subpopulation identification.
- **Manual Annotation**: If you have metadata available, assign subpopulation labels based on known characteristics.

**Example of Clustering for Subpopulation Identification:**

```matlab
% Perform K-means clustering to identify subpopulations
[idx, ~] = kmeans(expressionDataNorm', 3);  % Transpose for samples as rows
subpopulationLabels = idx;
```

### 3. Differential Expression Analysis

To perform DEA between subpopulations:

- **Statistical Testing**: Use `ttest2` for simple two-group comparisons or ANOVA (`anova1`) for multi-group comparisons.
- **Built-in Functions**: MATLAB’s Statistics and Machine Learning Toolbox provides functions like `ranksum` for non-parametric testing.

**Example of Differential Expression Analysis Between Two Subpopulations:**

```matlab
% Identify subpopulations (e.g., subpop1 and subpop2)
group1 = expressionDataNorm(:, subpopulationLabels == 1);
group2 = expressionDataNorm(:, subpopulationLabels == 2);

% Initialize vector to store p-values
numGenes = size(expressionDataNorm, 1);
pValues = zeros(numGenes, 1);

% Perform t-tests for each gene
for i = 1:numGenes
    [~, p] = ttest2(group1(i, :), group2(i, :), 'Vartype', 'unequal');
    pValues(i) = p;
end

% Adjust p-values using the Benjamini-Hochberg procedure for FDR control
pValuesAdj = mafdr(pValues, 'BHFDR', true);
```

### 4. Filtering for Significantly Differentially Expressed Genes

Set a significance threshold (e.g., `p < 0.05` or `adjusted p < 0.05`) to filter out significantly differentially expressed genes.

**Example of Filtering Results:**

```matlab
% Identify significantly differentially expressed genes
significantGenes = find(pValuesAdj < 0.05);
disp(['Number of significant genes: ', num2str(length(significantGenes))]);

% Visualize the top significant genes
topGenes = expressionDataNorm(significantGenes, :);
figure;
heatmap(topGenes, 'Colormap', parula);
title('Top Significant Differentially Expressed Genes');
```

### 5. Advanced Differential Expression Analysis

For more advanced analysis:

- **Mixed Models**: Use `fitlme` for linear mixed-effects models to handle complex experimental designs.
- **Permutation Tests**: Implement custom permutation tests for robust hypothesis testing.
- **Multi-Condition Comparison**: Use `anova1` or `multcompare` for comparing multiple subpopulations.

**Example of ANOVA for Multi-Group Comparison:**

```matlab
% ANOVA for each gene across multiple subpopulations
groupingVar = categorical(subpopulationLabels);
pValuesAnova = zeros(numGenes, 1);

for i = 1:numGenes
    pValuesAnova(i) = anova1(expressionDataNorm(i, :)', groupingVar, 'off');
end

% Adjust p-values for multiple testing
pValuesAnovaAdj = mafdr(pValuesAnova, 'BHFDR', true);
```

### 6. Visualization of Results

Visualize the differential expression results to interpret findings effectively:

- **Volcano Plots**: Create a volcano plot to display significance versus fold change.
- **Heatmaps**: Display expression levels of significant genes across subpopulations using `heatmap`.
- **Boxplots**: Plot specific gene expressions across subpopulations with `boxplot`.

**Example of a Volcano Plot:**

```matlab
% Compute fold changes between subpopulations
meanExprGroup1 = mean(group1, 2);
meanExprGroup2 = mean(group2, 2);
foldChange = log2(meanExprGroup2 ./ meanExprGroup1);

% Plot volcano plot
figure;
scatter(foldChange, -log10(pValuesAdj), 10, 'filled');
xlabel('Log2 Fold Change');
ylabel('-Log10 Adjusted P-value');
title('Volcano Plot of Differential Expression');
grid on;
```

### 7. Reporting and Biological Interpretation

- **Enrichment Analysis**: Perform Gene Ontology (GO) or pathway analysis to interpret the biological significance of differentially expressed genes.
- **Annotation**: Use external databases or MATLAB's bioinformatics tools to map significant genes to known pathways or functions.

**Example of GO Enrichment:**
Use MATLAB's `getgenbank` or integrate with external bioinformatics tools to perform enrichment analysis.

### 8. Automation and Custom Functions

To streamline your analysis, create custom functions for repetitive tasks such as running t-tests or generating plots.

**Example Custom Function for T-tests:**

```matlab
function pValues = runTTest(data, labels, group1ID, group2ID)
    numGenes = size(data, 1);
    pValues = zeros(numGenes, 1);
    
    for i = 1:numGenes
        [~, p] = ttest2(data(i, labels == group1ID), data(i, labels == group2ID), 'Vartype', 'unequal');
        pValues(i) = p;
    end
end
```

### Summary

MATLAB provides a comprehensive platform for differential expression analysis in subpopulations with tools for:

- **Preprocessing**: Data normalization and subpopulation labeling.
- **Statistical Testing**: Implementing t-tests, ANOVA, and permutation tests.
- **Visualization**: Creating meaningful plots to interpret results.
- **Biological Insight**: Incorporating GO or pathway analysis for deeper understanding.

### :cactus:Python snippet

Performing differential expression analysis (DEA) in subpopulations with Python involves using specialized libraries for data manipulation, statistical testing, and visualization. This guide covers steps for analyzing differential expression to detect selective perturbation effects.

### 1. Data Preparation

Ensure the data is properly formatted:

- **Data Structure**: Use a `pandas` DataFrame where rows are genes and columns are samples (cells or conditions).
- **Normalization**: Normalize the data using log transformation or scaling techniques.
- **Subpopulation Labels**: Assign labels to subpopulations based on clustering results or metadata.

**Example of Data Normalization:**

```python
import pandas as pd
import numpy as np

# Load data into a DataFrame (rows: genes, columns: samples)
expression_data = pd.read_csv('expression_data.csv', index_col=0)

# Log2 transformation for normalization
expression_data_log = np.log2(expression_data + 1)
```

### 2. Identifying Subpopulations

To identify subpopulations, use clustering or pre-defined labels:

- **Clustering**: Use `KMeans`, `AgglomerativeClustering`, or `DBSCAN` from `sklearn`.
- **Metadata**: Use existing labels if provided.

**Clustering Example (K-means):**

```python
from sklearn.cluster import KMeans

# Perform clustering (e.g., K-means for 3 subpopulations)
kmeans = KMeans(n_clusters=3, random_state=42)
subpop_labels = kmeans.fit_predict(expression_data_log.T)
expression_data_log['Subpopulation'] = subpop_labels
```

### 3. Differential Expression Analysis

Use statistical tests to find differentially expressed genes between subpopulations:

- **T-tests**: Use `scipy.stats.ttest_ind`.
- **Mann-Whitney U Test**: A non-parametric test with `scipy.stats.mannwhitneyu` for non-normally distributed data.
- **Multiple Testing Correction**: Adjust p-values with `statsmodels.stats.multitest`.

**Example of a Two-Group Comparison:**

```python
from scipy import stats
from statsmodels.stats.multitest import multipletests

# Compare subpopulation 0 and subpopulation 1
group1 = expression_data_log.loc[:, expression_data_log['Subpopulation'] == 0].iloc[:, :-1]
group2 = expression_data_log.loc[:, expression_data_log['Subpopulation'] == 1].iloc[:, :-1]

# Perform t-test for each gene
p_values = []
for gene in expression_data_log.index:
    _, p = stats.ttest_ind(group1.loc[gene], group2.loc[gene], equal_var=False)
    p_values.append(p)

# Adjust p-values using Benjamini-Hochberg FDR
p_values_adj = multipletests(p_values, alpha=0.05, method='fdr_bh')[1]

# Identify significant genes
significant_genes = expression_data_log.index[p_values_adj < 0.05]
print(f"Number of significant genes: {len(significant_genes)}")
```

### 4. Filtering for Significant Genes

Filter the genes based on adjusted p-values and set a fold-change threshold:

- **Fold Change**: Compute fold changes between group means.
- **Significance Threshold**: Filter by `p-values_adj < 0.05` and a fold-change cutoff (e.g., `|log2FC| > 1`).

**Computing Fold Change:**

```python
# Calculate mean expression for each group
mean_group1 = group1.mean(axis=1)
mean_group2 = group2.mean(axis=1)

# Compute log2 fold change
log2_fold_change = np.log2(mean_group2 / mean_group1)

# Filter significant genes by p-value and fold change
filtered_genes = expression_data_log.index[(p_values_adj < 0.05) & (abs(log2_fold_change) > 1)]
print(f"Number of significant genes after filtering: {len(filtered_genes)}")
```

### 5. Visualization of Results

Visualize the differentially expressed genes using:

- **Volcano Plots**: Combine fold change and p-values to visualize the overall significance.
- **Heatmaps**: Plot expression levels of significant genes.
- **Boxplots**: Display specific gene expressions across subpopulations.

**Volcano Plot Example:**

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.scatter(log2_fold_change, -np.log10(p_values_adj), c=(p_values_adj < 0.05), cmap='coolwarm', alpha=0.75)
plt.xlabel('Log2 Fold Change')
plt.ylabel('-Log10 Adjusted P-value')
plt.title('Volcano Plot of Differential Expression')
plt.axhline(-np.log10(0.05), color='gray', linestyle='--', lw=1)  # Significance threshold
plt.show()
```

**Heatmap Example:**

```python
import seaborn as sns

# Select significant genes for heatmap
significant_data = expression_data_log.loc[filtered_genes, :-1]  # Remove subpopulation column

plt.figure(figsize=(12, 8))
sns.heatmap(significant_data, cmap='viridis', yticklabels=True)
plt.title('Heatmap of Significant Genes')
plt.show()
```

### 6. Advanced Analysis

- **Statistical Models**: Use `statsmodels` for linear models (`OLS`) and generalized linear models (`GLM`) to adjust for covariates.
- **Permutation Tests**: Implement custom permutation tests to validate findings.
- **Machine Learning for Feature Selection**: Use `RandomForestClassifier` or other classifiers to identify important features contributing to subpopulation differences.

**Example Using `statsmodels` for Linear Regression:**

```python
import statsmodels.api as sm

# Prepare data for linear regression
X = pd.get_dummies(expression_data_log['Subpopulation'], drop_first=True)
y = expression_data_log.loc['GENE_NAME']  # Replace 'GENE_NAME' with actual gene name

model = sm.OLS(y, sm.add_constant(X))
results = model.fit()
print(results.summary())
```

### 7. Biological Interpretation and Reporting

- **Gene Ontology (GO) Enrichment**: Use `gseapy` or external tools for functional annotation.
- **Pathway Analysis**: Map differentially expressed genes to biological pathways using `BioPython` or integrate with web tools like DAVID or GSEA.

**Example of GO Enrichment with `gseapy`:**

```python
import gseapy as gp

# Run enrichment analysis
enrich_results = gp.enrichr(gene_list=list(filtered_genes), gene_sets='GO_Biological_Process', organism='Human')
enrich_results.res2d.head()
```

### Summary

Python provides a powerful suite for performing differential expression analysis in subpopulations:

1. **Preprocessing**: Normalize and label the data.
2. **Clustering**: Identify subpopulations with `scikit-learn`.
3. **Statistical Analysis**: Use `scipy` and `statsmodels` for hypothesis testing.
4. **Visualization**: Create volcano plots, heatmaps, and boxplots.
5. **Interpretation**: Perform GO and pathway enrichment to understand biological significance.

This approach provides comprehensive tools for detecting and analyzing perturbation effects on subpopulations in biological data.

### :cactus:R snippet

Differential expression analysis (DEA) in subpopulations for detecting selective perturbation effects is a common task in bioinformatics, particularly with single-cell RNA-seq or other high-dimensional data. R, with its robust ecosystem of statistical and bioinformatics packages, is well-suited for such analyses. Here’s a detailed guide to performing DEA in subpopulations using R.

### 1. Data Preparation

Properly format and preprocess the data before analysis:

- **Data Structure**: Use a `matrix` or `data.frame` where rows represent genes and columns represent samples.
- **Normalization**: Normalize the data using methods like `log` transformation or scaling with packages like `DESeq2`.
- **Subpopulation Labels**: Assign subpopulation labels based on clustering or prior metadata.

**Example of Data Loading and Normalization:**

```r
library(DESeq2)

# Load data (rows: genes, columns: samples)
counts <- as.matrix(read.csv("expression_data.csv", row.names = 1))

# Create a DESeqDataSet object (assumes metadata is available for subpopulation labels)
metadata <- read.csv("metadata.csv", row.names = 1)
dds <- DESeqDataSetFromMatrix(countData = counts, colData = metadata, design = ~ subpopulation)

# Normalize data
dds <- DESeq(dds)
normalized_counts <- counts(dds, normalized = TRUE)
```

### 2. Identifying Subpopulations

To identify subpopulations:

- **Clustering**: Use `kmeans`, `hclust`, or `Seurat` for more advanced single-cell data analysis.
- **Labeling**: Use pre-existing subpopulation labels from metadata or generate them using clustering results.

**Example of Clustering to Identify Subpopulations:**

```r
library(stats)

# Transpose normalized data for clustering (samples as rows)
kmeans_result <- kmeans(t(normalized_counts), centers = 3)
metadata$subpopulation <- as.factor(kmeans_result$cluster)
```

### 3. Differential Expression Analysis

Use `DESeq2` or `edgeR` to perform DEA between subpopulations.

**Example of DEA with `DESeq2`:**

```r
# Redefine design to compare specific subpopulations
dds$subpopulation <- relevel(dds$subpopulation, ref = "1")  # Set reference group
dds <- DESeq(dds)

# Perform DEA for a specific contrast (e.g., subpopulation 2 vs 1)
results <- results(dds, contrast = c("subpopulation", "2", "1"))
```

**Example of DEA with `edgeR`:**

```r
library(edgeR)

# Prepare data for edgeR
group <- metadata$subpopulation
y <- DGEList(counts = counts, group = group)
y <- calcNormFactors(y)

# Create design matrix
design <- model.matrix(~ 0 + group)
colnames(design) <- levels(group)
y <- estimateDisp(y, design)

# Fit the model and test
fit <- glmFit(y, design)
lrt <- glmLRT(fit, contrast = c(-1, 1, 0))  # Adjust contrast as needed

# Get top genes
topTags(lrt)
```

### 4. Filtering and Adjusting for Multiple Testing

Filter significant genes using adjusted p-values and fold-change thresholds:

- **Multiple Testing Correction**: Use `p.adjust` for FDR correction.
- **Significance Threshold**: Set a threshold (e.g., `p < 0.05` and `|log2FoldChange| > 1`).

**Example of Filtering Significant Genes:**

```r
# Filter by adjusted p-value and log2 fold-change
significant_genes <- subset(results, padj < 0.05 & abs(log2FoldChange) > 1)
print(paste("Number of significant genes:", nrow(significant_genes)))
```

### 5. Visualization of Results

Visualize the differential expression results:

- **Volcano Plots**: Use `ggplot2` or `EnhancedVolcano` for comprehensive volcano plots.
- **Heatmaps**: Use `pheatmap` to visualize the expression of significant genes.
- **Boxplots**: Display specific gene expressions across subpopulations.

**Volcano Plot Example:**

```r
library(EnhancedVolcano)

EnhancedVolcano(results,
                lab = rownames(results),
                x = 'log2FoldChange',
                y = 'padj',
                pCutoff = 0.05,
                FCcutoff = 1,
                title = 'Volcano Plot of Differential Expression',
                subtitle = 'Subpopulation 2 vs 1')
```

**Heatmap Example:**

```r
library(pheatmap)

# Select significant genes for heatmap
selected_genes <- rownames(significant_genes)
pheatmap(normalized_counts[selected_genes, ], annotation_col = metadata)
```

### 6. Advanced Analysis

For more complex analyses:

- **Multiple Conditions**: Adjust the design formula in `DESeq2` to include multiple conditions or covariates (e.g., `~ condition + subpopulation`).
- **Permutation Tests**: Implement custom permutation tests for added robustness.
- **Gene Set Enrichment Analysis**: Use `clusterProfiler` for GO and pathway analysis.

**Example of GO Enrichment with `clusterProfiler`:**

```r
library(clusterProfiler)

# Run enrichment analysis
gene_list <- rownames(subset(significant_genes, log2FoldChange > 1))
enrich_res <- enrichGO(gene = gene_list, OrgDb = org.Hs.eg.db, keyType = "SYMBOL", ont = "BP")
dotplot(enrich_res)
```

### 7. Reporting and Interpretation

- **Summarize Results**: Create summary tables with gene names, log2 fold changes, and p-values.
- **Pathway Analysis**: Integrate results with pathway databases to interpret biological significance.
- **Reproducibility**: Save your code and use `knitr` or `rmarkdown` for creating detailed analysis reports.

**Summary Table Example:**

```r
write.csv(as.data.frame(significant_genes), file = "DEA_results.csv")
```

### Summary

R provides a comprehensive workflow for differential expression analysis in subpopulations:

1. **Preprocess Data**: Normalize and label your dataset.
2. **Clustering**: Identify subpopulations using `kmeans`, `Seurat`, or hierarchical methods.
3. **DEA**: Use `DESeq2`, `edgeR`, or similar packages for differential expression analysis.
4. **Filtering and Correction**: Adjust for multiple testing and filter by significance.
5. **Visualization**: Create volcano plots, heatmaps, and boxplots.
6. **Enrichment Analysis**: Use `clusterProfiler` or `topGO` for deeper biological insights.

This approach helps in identifying and interpreting the effects of selective perturbations across subpopulations in biological data.

### :cactus:C++ snippet

Performing differential expression analysis (DEA) in subpopulations using C++ is more challenging compared to languages like R or Python due to the lack of specialized bioinformatics libraries. However, it is possible to accomplish this by leveraging core C++ functionalities and integrating with statistical libraries. Here’s a step-by-step approach to conducting DEA in subpopulations with C++:

### 1. Data Preparation

Prepare your gene expression data in an appropriate format:

- **Matrix Representation**: Use a 2D array or `std::vector<std::vector<double>>` for storing gene expression data, where rows represent genes and columns represent samples.
- **Normalization**: Implement normalization functions for data preprocessing, such as log transformation.

**Example of Data Loading and Normalization:**

```cpp
#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <sstream>

// Function to read CSV data into a 2D vector
std::vector<std::vector<double>> readCSV(const std::string& filename) {
    std::vector<std::vector<double>> data;
    std::ifstream file(filename);
    std::string line;
    
    while (std::getline(file, line)) {
        std::vector<double> row;
        std::stringstream ss(line);
        std::string cell;
        
        while (std::getline(ss, cell, ',')) {
            row.push_back(std::stod(cell));
        }
        data.push_back(row);
    }
    return data;
}

// Log2 transformation function
void logTransform(std::vector<std::vector<double>>& data) {
    for (auto& row : data) {
        for (auto& val : row) {
            val = std::log2(val + 1);  // Add 1 to avoid log(0)
        }
    }
}
```

### 2. Identifying Subpopulations

For subpopulation identification, you can implement clustering algorithms or use external tools to generate clustering results that can be imported into C++.

**Example of K-Means Clustering in C++:**
Use libraries such as [OpenCV](https://opencv.org/) for k-means clustering, or implement the algorithm manually.

### 3. Statistical Testing for Differential Expression

Use statistical tests to identify differentially expressed genes:

- **T-tests**: Implement or use libraries for conducting Welch’s t-tests between subpopulations.
- **Multiple Testing Correction**: Implement methods like Benjamini-Hochberg for controlling the false discovery rate (FDR).

**Example of Implementing a T-Test Function:**

```cpp
#include <vector>
#include <cmath>

// Function to calculate mean
double mean(const std::vector<double>& data) {
    double sum = 0;
    for (double val : data) sum += val;
    return sum / data.size();
}

// Function to calculate standard deviation
double standardDeviation(const std::vector<double>& data, double mean) {
    double sum = 0;
    for (double val : data) sum += (val - mean) * (val - mean);
    return std::sqrt(sum / (data.size() - 1));
}

// Welch's t-test for two independent samples
double tTest(const std::vector<double>& group1, const std::vector<double>& group2) {
    double mean1 = mean(group1);
    double mean2 = mean(group2);
    double sd1 = standardDeviation(group1, mean1);
    double sd2 = standardDeviation(group2, mean2);
    int n1 = group1.size();
    int n2 = group2.size();

    double t_stat = (mean1 - mean2) / std::sqrt((sd1 * sd1 / n1) + (sd2 * sd2 / n2));
    return t_stat;
}
```

### 4. Adjusting P-values for Multiple Testing

Implement the Benjamini-Hochberg procedure to control the false discovery rate:

```cpp
#include <vector>
#include <algorithm>

// Structure to hold p-value and original index
struct PValue {
    double value;
    int index;
};

// Adjust p-values using the Benjamini-Hochberg method
std::vector<double> adjustPValues(std::vector<double> pValues) {
    int n = pValues.size();
    std::vector<PValue> pValStruct;
    
    for (int i = 0; i < n; ++i) {
        pValStruct.push_back({pValues[i], i});
    }
    
    // Sort p-values in ascending order
    std::sort(pValStruct.begin(), pValStruct.end(), [](PValue a, PValue b) {
        return a.value < b.value;
    });

    std::vector<double> adjustedPValues(n);
    for (int i = 0; i < n; ++i) {
        adjustedPValues[pValStruct[i].index] = pValStruct[i].value * n / (i + 1);
        if (i > 0) {
            adjustedPValues[pValStruct[i].index] = std::min(adjustedPValues[pValStruct[i].index], adjustedPValues[pValStruct[i - 1].index]);
        }
    }
    
    return adjustedPValues;
}
```

### 5. Visualization of Results

For visualization, export results and use tools like Python's `matplotlib` or R's `ggplot2`:

- **Volcano Plots**: Generate scatter plots of log2 fold change vs. -log10 adjusted p-values.
- **Heatmaps**: Use C++ plotting libraries like [matplotlib-cpp](https://github.com/lava/matplotlib-cpp) or export data for external visualization.

### 6. Advanced Analysis

For more sophisticated analysis:

- **Permutation Tests**: Implement custom permutation tests for non-parametric DEA.
- **Integration with External Libraries**: Use libraries like [Eigen](https://eigen.tuxfamily.org/dox/) for matrix operations and [Armadillo](http://arma.sourceforge.net/) for advanced linear algebra.

### 7. Biological Interpretation

Connect significant genes to biological pathways:

- **Enrichment Analysis**: Use external tools or call web APIs for gene set enrichment analysis.
- **Pathway Analysis**: Integrate with databases like KEGG or use libraries for gene ontology.

### Summary

Performing DEA in subpopulations using C++ requires:

1. **Data Preprocessing**: Reading and normalizing data.
2. **Clustering**: Implement clustering or use external results.
3. **Statistical Testing**: Conduct t-tests and adjust p-values.
4. **Visualization**: Export data for visualization with external tools.
5. **Integration**: Combine C++ with other languages/tools for enriched functionality.

This method leverages the computational efficiency of C++ while integrating with specialized tools for visualization and biological interpretation.

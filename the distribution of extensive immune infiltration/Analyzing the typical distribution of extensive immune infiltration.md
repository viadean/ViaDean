# Analyzing the typical distribution of extensive immune infiltration 

Extensive immune infiltration refers to the widespread presence of immune cells within a tissue, often observed in inflammatory conditions, infections, or tumors. Typically, this infiltration is dominated by lymphocytes (T cells, B cells), macrophages, and dendritic cells, which localize near sites of antigen presentation, damaged cells, or neoplastic growth. In cancers, immune cells may cluster around the tumor periphery (immune-excluded) or penetrate deeply (inflamed phenotype). Chronic inflammation often shows diffuse infiltration across tissues. The distribution reflects the immune system's attempt to combat threats or maintain homeostasis, influenced by cytokine gradients, chemokine signals, and tissue-specific factors shaping immune cell recruitment and retention.

## :cactus:Python snippet

### :fallen_leaf:using scRNA-seq data

Single-cell RNA sequencing (scRNA-seq) data analysis involves many steps, from data preprocessing to visualization and downstream analysis. Here’s a Python example demonstrating some basic tasks like loading, preprocessing, and visualizing scRNA-seq data using the `Scanpy` library.

### Example: Basic scRNA-seq Analysis Pipeline with `Scanpy`

```python
import scanpy as sc
import matplotlib.pyplot as plt

# Step 1: Load a dataset
# For simplicity, use a built-in dataset from Scanpy
adata = sc.datasets.pbmc3k_processed()  # Preprocessed PBMC dataset

# Inspect the data
print(adata)  # Observations (cells) × Variables (genes)

# Step 2: Preprocess data
# Filter genes and cells based on minimum expression
sc.pp.filter_genes(adata, min_cells=10)  # Keep genes expressed in at least 10 cells
sc.pp.filter_cells(adata, min_genes=200)  # Keep cells with at least 200 expressed genes

# Normalize the data
sc.pp.normalize_total(adata, target_sum=1e4)  # Normalize total counts per cell
sc.pp.log1p(adata)  # Logarithmic transformation

# Identify highly variable genes
sc.pp.highly_variable_genes(adata, n_top_genes=2000)
adata = adata[:, adata.var['highly_variable']]  # Subset to these genes

# Scale the data
sc.pp.scale(adata)

# Step 3: Dimensionality Reduction
# Perform PCA
sc.tl.pca(adata, svd_solver='arpack')
sc.pl.pca(adata, color='CST3')  # Plot PCA with a gene marker

# Compute neighborhood graph
sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40)

# Perform UMAP for visualization
sc.tl.umap(adata)
sc.pl.umap(adata, color=['louvain', 'CST3'])  # Visualize clusters and gene expression

# Step 4: Clustering
# Use the Louvain algorithm for clustering
sc.tl.louvain(adata)
sc.pl.umap(adata, color='louvain')  # Visualize clusters

# Step 5: Differential Expression Analysis
# Find marker genes for clusters
sc.tl.rank_genes_groups(adata, groupby='louvain', method='wilcoxon')
sc.pl.rank_genes_groups(adata, n_genes=10, sharey=False)
```

### Explanation of Steps:

1. **Loading Data**:
   - Uses a preprocessed scRNA-seq dataset (PBMC 3k cells) as an example. Alternatively, use custom datasets in `.h5ad` or `.mtx` formats.
2. **Preprocessing**:
   - Filters low-quality genes and cells.
   - Normalizes expression counts and applies logarithmic transformation.
   - Identifies highly variable genes for downstream analysis.
3. **Dimensionality Reduction**:
   - Reduces data dimensionality using PCA.
   - Visualizes data with UMAP, preserving local structures.
4. **Clustering**:
   - Groups cells into clusters based on their expression profiles.
5. **Differential Expression Analysis**:
   - Identifies marker genes distinguishing clusters using statistical tests.

This pipeline demonstrates the fundamentals of scRNA-seq data analysis. Customize it with your data for more advanced workflows like trajectory analysis, integration, or cell type annotation.

### :fallen_leaf:Using Bulk RNA-seq data

Bulk RNA-seq data analysis involves quantifying gene expression across a population of cells, identifying differentially expressed genes, and performing downstream analyses like functional enrichment. Below is an example Python pipeline for bulk RNA-seq analysis using some common Python libraries:

### **Example Workflow**

This example demonstrates how to preprocess RNA-seq count data, perform differential expression analysis, and visualize the results.

------

#### **1. Install Required Libraries**

Make sure you have the following Python packages installed:

```bash
pip install numpy pandas matplotlib seaborn statsmodels
```

------

#### **2. Load Bulk RNA-Seq Data**

Suppose we have a dataset with raw gene counts (rows are genes, columns are samples) and a metadata file describing sample conditions.

```python
import pandas as pd
import numpy as np

# Load RNA-seq count data
counts_file = "gene_counts.csv"  # Replace with your file
metadata_file = "sample_metadata.csv"  # Replace with your file

counts = pd.read_csv(counts_file, index_col=0)  # Rows: Genes, Columns: Samples
metadata = pd.read_csv(metadata_file)          # Metadata: Sample info

# Inspect data
print("Counts shape:", counts.shape)
print("Metadata shape:", metadata.shape)
```

------

#### **3. Normalize Counts (e.g., TPM or DESeq2-style Normalization)**

Normalize raw counts for downstream analysis.

```python
# TPM normalization example
def tpm_normalization(counts, lengths):
    """
    Compute Transcripts Per Million (TPM).
    `counts` is a DataFrame of raw counts.
    `lengths` is a Series of gene lengths.
    """
    rpk = counts.div(lengths, axis=0)  # Reads Per Kilobase
    scaling_factors = rpk.sum(axis=0) / 1e6
    tpm = rpk.div(scaling_factors, axis=1)
    return tpm

# Load gene lengths (in kilobases)
gene_lengths = pd.read_csv("gene_lengths.csv", index_col=0, squeeze=True)
tpm = tpm_normalization(counts, gene_lengths)

print("Normalized Counts (TPM):")
print(tpm.head())
```

------

#### **4. Differential Expression Analysis**

Use `statsmodels` or other statistical libraries for differential expression analysis (e.g., DESeq2 in R is often used for robust DE analysis).

```python
import statsmodels.api as sm
from statsmodels.formula.api import ols

# Example: Perform t-tests for a simple case-control comparison
condition = metadata["condition"]  # Assuming "condition" is a column in metadata
groups = condition.unique()

# Ensure only two groups for t-test
assert len(groups) == 2, "Only two groups supported for this example!"

# Perform t-test for each gene
from scipy.stats import ttest_ind

group1 = counts.loc[:, metadata[condition == groups[0]].index]
group2 = counts.loc[:, metadata[condition == groups[1]].index]

pvals = []
for gene in counts.index:
    _, pval = ttest_ind(group1.loc[gene], group2.loc[gene], equal_var=False)
    pvals.append(pval)

# Add p-values to the results DataFrame
results = pd.DataFrame({
    "Gene": counts.index,
    "p-value": pvals,
    # Apply multiple testing correction if needed (e.g., Benjamini-Hochberg)
}).set_index("Gene")

print(results.head())
```

------

#### **5. Visualization**

Create plots such as heatmaps, volcano plots, or PCA to visualize results.

```python
import seaborn as sns
import matplotlib.pyplot as plt

# Volcano Plot
results["-log10(p-value)"] = -np.log10(results["p-value"])
plt.figure(figsize=(8, 6))
sns.scatterplot(data=results, x="log2FC", y="-log10(p-value)", alpha=0.7)
plt.axhline(-np.log10(0.05), color="red", linestyle="--", label="p=0.05")
plt.title("Volcano Plot")
plt.xlabel("Log2 Fold Change")
plt.ylabel("-Log10(p-value)")
plt.legend()
plt.show()
```

------

#### **6. Functional Enrichment Analysis**

Use libraries like `gprofiler-official` for pathway and GO term enrichment.

```bash
pip install gprofiler-official
from gprofiler import GProfiler

gp = GProfiler(return_dataframe=True)
enrichment_results = gp.profile(organism='hsapiens', query=results.index[:100])  # Top 100 genes
print(enrichment_results.head())
```

------

### Wrap up

This pipeline:

1. Loads RNA-seq count data and metadata.
2. Normalizes counts using TPM.
3. Identifies differentially expressed genes with statistical tests.
4. Visualizes results with volcano plots.
5. Optionally performs functional enrichment analysis.

Replace file names and paths with your data to adapt this script. For larger or more complex datasets, consider using dedicated tools such as DESeq2 or edgeR in R.

### :fallen_leaf:Using tissue microarray data

Using tissue microarray (TMA) data in Python typically involves analyzing and visualizing pathological data, often stored in formats like spreadsheets or databases. TMA data may include information on tissue samples, histological scores, and associated clinical data. Here's an example pipeline:

### Scenario:

1. Import a CSV file containing TMA data.
2. Perform basic exploratory data analysis (EDA).
3. Generate visualizations like heatmaps for TMA core scores.

Here’s a Python example:

------

#### Example: Analyzing Tissue Microarray Data

```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Step 1: Load TMA data
# Assume the CSV file contains columns like 'CoreID', 'MarkerA_Score', 'MarkerB_Score', 'PatientID', etc.
file_path = "tma_data.csv"  # Replace with your file path
tma_data = pd.read_csv(file_path)

# View the first few rows
print(tma_data.head())

# Step 2: Basic EDA
# Check for missing values
print(tma_data.isnull().sum())

# Summary statistics
print(tma_data.describe())

# Step 3: Create a heatmap for marker scores
# Pivot the data for heatmap visualization (e.g., rows as 'CoreID', columns as 'Markers')
heatmap_data = tma_data.pivot(index='CoreID', columns='PatientID', values='MarkerA_Score')

# Plot the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(heatmap_data, cmap='viridis', annot=True, cbar=True)
plt.title("Heatmap of MarkerA Scores Across Tissue Cores")
plt.xlabel("Patient ID")
plt.ylabel("Core ID")
plt.show()

# Step 4: Correlation analysis (optional)
# Analyze correlation between scores of different markers
correlation_matrix = tma_data[['MarkerA_Score', 'MarkerB_Score']].corr()
print("Correlation Matrix:")
print(correlation_matrix)

# Visualize correlation
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title("Correlation Between Marker Scores")
plt.show()
```

------

#### Explanation:

1. **Loading Data:** The `pandas` library handles data import and cleaning.
2. **EDA:** Quick checks for missing values and statistics ensure data quality.
3. **Heatmap Visualization:** A heatmap is useful for spatial relationships of TMA core scores across patients or samples.
4. **Correlation Analysis:** Understanding marker relationships helps in biomarker studies.

------

#### Sample Data Format (`tma_data.csv`):

| CoreID | PatientID | MarkerA_Score | MarkerB_Score |
| ------ | --------- | ------------- | ------------- |
| Core01 | Patient1  | 2.5           | 3.0           |
| Core02 | Patient1  | 1.8           | 2.4           |
| Core03 | Patient2  | 3.1           | 3.5           |
| Core04 | Patient2  | 2.2           | 2.8           |

This approach can be extended by incorporating advanced analysis, such as machine learning for predictive insights, or interfacing with digital pathology tools for automated TMA analysis. 

## :herb:Advanced Analyses 

Spatial transcriptomics integrates spatially resolved gene expression data with tissue histology, enabling the mapping of cell types, including immune cells, onto tissue sections. Python is widely used for such tasks, often leveraging libraries like **Scanpy**, **Squidpy**, and **Seaborn** for data analysis and visualization.

Here’s a step-by-step guide for mapping immune cells onto tissue sections using spatial transcriptomics:

------

### **1. Prerequisites**

- Install the necessary Python libraries:

  ```bash
  pip install scanpy squidpy pandas matplotlib seaborn scikit-learn
  ```

### **2. Load and Prepare Data**

You need:

- **Spatial transcriptomics data** (e.g., spatial gene expression matrix).
- **Tissue image** (e.g., histological image like H&E staining).

Load the spatial transcriptomics data (often in `.h5ad` format) and preprocess it:

```python
import scanpy as sc
import squidpy as sq
import matplotlib.pyplot as plt

# Load the spatial data
adata = sc.read_h5ad("spatial_data.h5ad")

# View basic info
print(adata)
```

------

### **3. Annotate Immune Cells**

Use marker genes to identify immune cells (e.g., T-cells, B-cells). If cell-type annotations are already available, skip this step.

```python
# Define immune cell marker genes
marker_genes = {
    "T_cells": ["CD3D", "CD3E", "CD4", "CD8A"],
    "B_cells": ["MS4A1"],
    "Macrophages": ["CD68", "CD163"],
}

# Score for immune markers
sc.tl.score_genes(adata, gene_list=marker_genes["T_cells"], score_name="T_cell_score")
sc.tl.score_genes(adata, gene_list=marker_genes["B_cells"], score_name="B_cell_score")
sc.tl.score_genes(adata, gene_list=marker_genes["Macrophages"], score_name="Macrophage_score")
```

------

### **4. Visualize Immune Cell Scores on the Tissue**

Map the expression of immune cell scores to spatial coordinates.

```python
# Plot spatial distribution of scores
sc.pl.spatial(
    adata, 
    color=["T_cell_score", "B_cell_score", "Macrophage_score"],
    size=1.5, 
    cmap="viridis"
)
```

------

### **5. Cluster and Identify Cell Types**

Cluster the cells and assign immune cell types based on marker gene scores.

```python
# Perform clustering
sc.tl.leiden(adata, resolution=0.5)

# Visualize clusters
sc.pl.spatial(adata, color="leiden", size=1.5)

# Annotate clusters with immune cell types (manual or automated)
adata.obs["cell_type"] = adata.obs["leiden"].map({
    "0": "T_cells",
    "1": "B_cells",
    "2": "Macrophages",
    # Add other mappings based on clustering
})
```

------

### **6. Use Squidpy for Advanced Spatial Analysis**

Squidpy provides tools for analyzing spatial interactions between cell types.

```python
# Compute neighborhood graph
sq.gr.spatial_neighbors(adata)

# Visualize spatial cell type distribution
sq.pl.spatial_scatter(adata, color="cell_type", size=1.5)
```

------

### **7. Overlay on Histological Image**

If your data includes a tissue image, overlay immune cells for enhanced visualization.

```python
# Overlay cell types on tissue image
sq.pl.spatial_scatter(
    adata, 
    img_key="hires",  # Adjust for your image key
    color="cell_type", 
    size=1.5,
    alpha_img=0.7
)
```

------

### **8. Save and Share Results**

Save annotated data and visualizations for downstream analysis.

```python
# Save annotated data
adata.write("annotated_spatial_data.h5ad")

# Save plots
plt.savefig("immune_cell_mapping.png")
```

------

### Additional Considerations

1. Data Sources:
   - Ensure high-quality spatial transcriptomics data (e.g., Visium by 10x Genomics).
   - Use annotated datasets for training/validation.
2. Validation:
   - Validate immune cell assignments using reference datasets like CellMarker or PanglaoDB.
3. Scalability:
   - For large datasets, use subsampling or efficient computation libraries.

This pipeline provides a robust framework for mapping immune cells onto tissue sections in spatial transcriptomics using Python.


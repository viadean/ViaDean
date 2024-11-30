# Code snippets to Single-cell analysis methods

Single-cell analysis methods encompass a range of advanced techniques aimed at examining the molecular and genetic features of individual cells. Here are some key methods:

1. **Single-cell RNA sequencing (scRNA-seq)**: Analyzes the transcriptome of individual cells, revealing gene expression patterns and cellular heterogeneity.
2. **Single-cell DNA sequencing**: Allows for the study of genetic variations, including mutations and copy number variations, in single cells.
3. **Single-cell ATAC-seq**: Assesses chromatin accessibility to understand how genes are regulated at the single-cell level.
4. **Single-cell proteomics**: Uses mass spectrometry or antibody-based techniques to quantify proteins in single cells, providing insights into functional states.
5. **Flow cytometry and fluorescence-activated cell sorting (FACS)**: Enables the sorting and analysis of cells based on specific markers or properties.
6. **Single-cell epigenomics**: Investigates DNA modifications like methylation patterns in individual cells.
7. **Spatial transcriptomics**: Combines single-cell analysis with spatial information, mapping gene expression to specific tissue locations.

These techniques are vital for studying cell diversity, development, and disease mechanisms in various biological and medical research contexts.

MATLAB provides a range of tools and functions for analyzing single-cell data, particularly through specialized toolboxes and user-developed functions. Here are some commonly used methods and resources for single-cell analysis in MATLAB:

1. **Bioinformatics Toolbox**:
   - MATLAB’s **Bioinformatics Toolbox** offers functions for handling biological data, including gene expression matrices and sequence analysis. 
   - It can be used for data preprocessing, normalization, and clustering of single-cell data.

2. **Single-Cell Data Processing**:
   - **Data import and preprocessing**: MATLAB can handle large single-cell datasets imported from common formats (e.g., CSV, HDF5). Users can clean, filter, and normalize data using custom scripts.
   - **Dimensionality reduction**: Functions such as `pca()` and `tsne()` help visualize high-dimensional single-cell data in 2D or 3D space.

3. **Clustering and Classification**:
   - MATLAB has built-in functions like `kmeans()`, `hierarchical clustering`, and `spectral clustering` for cell-type identification based on gene expression patterns.
   - Advanced machine learning tools (e.g., `fitcnb`, `fitctree`, or neural networks from Deep Learning Toolbox) can classify cell types or predict cell states.

4. **Visualization**:
   - **Heatmaps**: `heatmap()` is used to create expression heatmaps.
   - **Scatter plots**: Useful for 2D/3D representations of reduced data (e.g., using `scatter3()`).
   - **Interactive visualizations**: MATLAB apps and scripts can create interactive plots for deeper data exploration.

5. **Toolboxes and Custom Scripts**:
   - **Single-cell MATLAB Toolboxes**: Community-developed toolboxes (such as those shared on MATLAB File Exchange or GitHub) can extend MATLAB’s capabilities in single-cell analysis.
   - **Integration with Python/R**: MATLAB can call Python/R functions, allowing use of popular single-cell packages like `Seurat` (R) or `Scanpy` (Python).

6. **Statistical Analysis**:
   - MATLAB’s statistical functions can perform differential expression analysis and other statistical tests to compare groups of cells.

7. **MATLAB Scripts for Single-cell RNA-seq**:
   - Some researchers provide scripts for single-cell RNA-seq data processing, clustering, and analysis, which can be customized to specific research needs.

These capabilities make MATLAB a flexible platform for handling single-cell data, offering robust analysis and visualization tools that can be integrated into larger pipelines.

### :cactus:MATLAB code snippets

Below are some common single-cell analysis methods using MATLAB code snippets to demonstrate how you can perform these tasks:

### 1. **Data Import and Preprocessing**

Load single-cell RNA-seq data (e.g., a gene expression matrix):

```matlab
% Load data from a CSV file
data = readmatrix('single_cell_data.csv');

% Normalize data (log normalization)
normalizedData = log1p(data);
```

### 2. **Dimensionality Reduction**

Perform PCA and visualize the result:

```matlab
% Perform PCA
[coeff, score, ~] = pca(normalizedData);

% Plot the first two principal components
scatter(score(:, 1), score(:, 2), 10, 'filled');
title('PCA of Single-cell Data');
xlabel('PC1');
ylabel('PC2');
```

Run t-SNE for better visualization of cell clusters:

```matlab
% Run t-SNE
Y = tsne(normalizedData);

% Plot t-SNE results
scatter(Y(:, 1), Y(:, 2), 10, 'filled');
title('t-SNE Visualization');
xlabel('t-SNE 1');
ylabel('t-SNE 2');
```

### 3. **Clustering**

Apply k-means clustering to identify potential cell types:

```matlab
% Perform k-means clustering
numClusters = 5;  % Adjust as needed
[idx, C] = kmeans(normalizedData, numClusters);

% Plot with cluster labels
gscatter(Y(:, 1), Y(:, 2), idx);
title('t-SNE with k-means Clustering');
xlabel('t-SNE 1');
ylabel('t-SNE 2');
```

### 4. **Visualization**

Generate a heatmap for visualizing gene expression:

```matlab
% Create a heatmap for the top 100 genes (rows) across all cells (columns)
heatmap(normalizedData(1:100, :));
title('Gene Expression Heatmap');
```

### 5. **Differential Expression Analysis**

Use basic statistical functions to compare two groups of cells:

```matlab
% Split data into two groups (e.g., clusters 1 and 2)
group1 = normalizedData(idx == 1, :);
group2 = normalizedData(idx == 2, :);

% Perform a t-test for differential expression
pValues = arrayfun(@(i) ttest2(group1(:, i), group2(:, i)), 1:size(normalizedData, 2));

% Find significantly differentially expressed genes
significantGenes = find(pValues < 0.05);
disp('Significant genes:');
disp(significantGenes);
```

### 6. **Interactive Visualization with MATLAB App**

Use MATLAB's built-in apps for exploring data interactively:

```matlab
% Launch the basic scatter plot app for interactive exploration
scatter3(score(:, 1), score(:, 2), score(:, 3), 15, idx, 'filled');
xlabel('PC1');
ylabel('PC2');
zlabel('PC3');
title('3D PCA Plot');
```

These snippets give a foundation for single-cell data analysis in MATLAB. They can be customized or expanded based on specific research needs or integrated with more complex machine learning or statistical modeling as required.

### :cactus:R code snippets

R is widely used in single-cell analysis due to its rich ecosystem of packages designed for genomics and bioinformatics. Below are some popular single-cell analysis methods in R, along with code snippets for implementation:

### 1. **Data Import and Preprocessing**

Load single-cell RNA-seq data using the `Seurat` package:

```r
# Load necessary library
library(Seurat)

# Load data (assuming data is in a CSV file)
data <- read.csv("single_cell_data.csv", row.names = 1)

# Create a Seurat object
seurat_obj <- CreateSeuratObject(counts = data)

# Normalize the data
seurat_obj <- NormalizeData(seurat_obj)

# Find variable features
seurat_obj <- FindVariableFeatures(seurat_obj)
```

### 2. **Dimensionality Reduction**

Perform PCA and visualize the results:

```r
# Run PCA
seurat_obj <- ScaleData(seurat_obj)
seurat_obj <- RunPCA(seurat_obj)

# Visualize PCA results
DimPlot(seurat_obj, reduction = "pca")
```

Run t-SNE for better visualization of cell clusters:

```r
# Run t-SNE
seurat_obj <- RunTSNE(seurat_obj, dims = 1:10)

# Plot t-SNE results
DimPlot(seurat_obj, reduction = "tsne", group.by = "ident")
```

### 3. **Clustering**

Cluster cells and visualize them:

```r
# Find clusters
seurat_obj <- FindNeighbors(seurat_obj, dims = 1:10)
seurat_obj <- FindClusters(seurat_obj, resolution = 0.5)

# Visualize clusters using UMAP
seurat_obj <- RunUMAP(seurat_obj, dims = 1:10)
DimPlot(seurat_obj, reduction = "umap", label = TRUE)
```

### 4. **Differential Expression Analysis**

Identify differentially expressed genes between clusters:

```r
# Find markers for a specific cluster compared to all remaining cells
cluster_markers <- FindMarkers(seurat_obj, ident.1 = 0, min.pct = 0.25)

# Print top differentially expressed genes
head(cluster_markers)
```

### 5. **Visualization of Gene Expression**

Visualize the expression of specific genes:

```r
# Feature plot for specific genes
FeaturePlot(seurat_obj, features = c("Gene1", "Gene2"))

# Violin plot for gene expression
VlnPlot(seurat_obj, features = c("Gene1"), group.by = "ident")
```

### 6. **Integration and Comparison**

Integrate multiple single-cell datasets for comparative analysis:

```r
# Load additional single-cell datasets
data2 <- read.csv("second_single_cell_data.csv", row.names = 1)
seurat_obj2 <- CreateSeuratObject(counts = data2)

# Integrate datasets
seurat_list <- list(seurat_obj, seurat_obj2)
seurat_combined <- merge(seurat_list[[1]], y = seurat_list[2:length(seurat_list)])

# Perform integrated analysis
seurat_combined <- NormalizeData(seurat_combined)
seurat_combined <- FindVariableFeatures(seurat_combined)
seurat_combined <- ScaleData(seurat_combined)
seurat_combined <- RunPCA(seurat_combined)
DimPlot(seurat_combined, reduction = "pca", group.by = "orig.ident")
```

### 7. **Epigenomic Analysis with Additional Packages**

Use packages like **Signac** for single-cell ATAC-seq data:

```r
# Load Signac for ATAC-seq analysis
library(Signac)

# Create Seurat object for ATAC-seq and follow similar steps for data preprocessing
```

These methods showcase the flexibility and power of R in handling single-cell data, making it a robust option for researchers in genomics and transcriptomics.

### :cactuPython snippet

Python is a powerful language for single-cell analysis, thanks to its extensive libraries and frameworks that streamline bioinformatics workflows. Here’s an overview of single-cell analysis methods in Python, along with relevant code snippets:

### 1. **Data Import and Preprocessing**

Using `Scanpy` for handling single-cell RNA-seq data:

```python
import scanpy as sc

# Load data (e.g., an HDF5 or CSV file)
adata = sc.read_csv('single_cell_data.csv')

# Basic preprocessing: filtering and normalization
sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=3)
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

# Identify highly variable genes
sc.pp.highly_variable_genes(adata)
adata = adata[:, adata.var['highly_variable']]
```

### 2. **Dimensionality Reduction**

Perform PCA and visualize results:

```python
# Run PCA
sc.pp.scale(adata, max_value=10)
sc.tl.pca(adata, svd_solver='arpack')

# Plot PCA results
sc.pl.pca(adata, color='batch')
```

Run UMAP or t-SNE for better visualization:

```python
# Run UMAP
sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40)
sc.tl.umap(adata)

# Plot UMAP
sc.pl.umap(adata, color=['gene1', 'gene2', 'leiden'])  # 'leiden' represents clusters
```

### 3. **Clustering**

Cluster cells using the Leiden or Louvain algorithms:

```python
# Clustering using Leiden
sc.tl.leiden(adata, resolution=0.5)

# Plot clusters on UMAP
sc.pl.umap(adata, color='leiden')
```

### 4. **Differential Expression Analysis**

Identify marker genes for specific clusters:

```python
# Find marker genes for each cluster
sc.tl.rank_genes_groups(adata, 'leiden', method='t-test')

# Visualize top marker genes
sc.pl.rank_genes_groups(adata, n_genes=20, sharey=False)
```

### 5. **Visualization of Gene Expression**

Plot expression of specific genes across cells:

```python
# UMAP with gene expression overlay
sc.pl.umap(adata, color=['GeneX'])

# Violin plot for specific genes
sc.pl.violin(adata, ['GeneX'], groupby='leiden')
```

### 6. **Integration and Batch Correction**

Integrate multiple datasets using `Scanpy` or other libraries like `Harmony`:

```python
# Concatenate multiple datasets
adata1 = sc.read_h5ad('dataset1.h5ad')
adata2 = sc.read_h5ad('dataset2.h5ad')
adata = adata1.concatenate(adata2, batch_key='batch')

# Batch correction using Harmony
import harmonypy as hm

# Run Harmony
adata.obsm['X_pca'] = hm.run_harmony(adata.obsm['X_pca'], adata.obs, 'batch')
sc.pp.neighbors(adata, use_rep='X_pca')
sc.tl.umap(adata)
sc.pl.umap(adata, color='batch')
```

### 7. **Epigenomic Analysis**

Use `AnnData` and packages like `scATAC-pro` for ATAC-seq data:

```python
# Import libraries
import anndata as ad

# Load ATAC-seq data and follow preprocessing and analysis steps similarly
```

### 8. **Integration with Machine Learning**

Leverage machine learning for cell-type classification or trajectory inference:

```python
from sklearn.ensemble import RandomForestClassifier

# Prepare data for classification
X = adata.X
y = adata.obs['cell_type']  # Assume cell type labels exist

# Train a RandomForest model
clf = RandomForestClassifier()
clf.fit(X, y)

# Predict on new data
predictions = clf.predict(X_new)
```

These code snippets showcase the utility of Python libraries such as `Scanpy`, `scikit-learn`, and `Harmony` for various single-cell analysis tasks. The workflow covers everything from data preprocessing and visualization to clustering, differential expression, and integration.



# Code snippets to High-dimensional transcriptional profiling



### :cactus:MATLAB snippet

MATLAB offers tools and workflows for analyzing high-dimensional transcriptional profiling data from techniques like single-cell RNA sequencing (scRNA-seq) or bulk RNA-seq, focusing on gene expression analysis, clustering, visualization, and dimensionality reduction. Here's an overview of using MATLAB to analyze high-dimensional transcriptional data from individual cells, from preprocessing to visualization.

### 1. Loading and Preprocessing Data

In MATLAB, you can start by loading raw gene expression data, which is typically stored as a matrix where rows represent genes and columns represent cells. Several formats, such as CSV, MAT, or HDF5, can be read using MATLAB's `readmatrix` or `importdata` functions.

```matlab
% Example: Load gene expression matrix
expressionData = readmatrix('gene_expression.csv');  % Load raw data
geneNames = readcell('gene_names.csv');  % Load gene names
cellNames = readcell('cell_names.csv');  % Load cell names

% Basic Preprocessing
expressionData = log1p(expressionData); % Log transformation to normalize data
```

### 2. Quality Control and Filtering

Filter low-quality cells and genes with low counts to remove noise. Basic metrics include:

- Removing cells with low gene counts (indicating dead or unhealthy cells).
- Removing genes with low expression across cells.

```matlab
% Set thresholds for quality control
minGeneCount = 200;
minCellCount = 5;

% Filter cells and genes
geneFilter = sum(expressionData > 0, 2) >= minGeneCount; % Genes expressed in enough cells
cellFilter = sum(expressionData > 0, 1) >= minCellCount; % Cells with enough genes expressed
expressionData = expressionData(geneFilter, cellFilter);
geneNames = geneNames(geneFilter);
cellNames = cellNames(cellFilter);
```

### 3. Normalization and Scaling

Normalization controls for differences in library size and sequencing depth, and scaling centers the data.

```matlab
% Normalize expression to account for sequencing depth
normalizedData = expressionData ./ sum(expressionData, 1) * median(sum(expressionData, 1));

% Z-score scaling for gene-wise normalization
scaledData = zscore(normalizedData, 0, 2); % Normalize each gene's expression
```

### 4. Dimensionality Reduction (PCA, t-SNE, UMAP)

Dimensionality reduction helps visualize and interpret high-dimensional data. MATLAB’s Statistics and Machine Learning Toolbox offers methods like PCA, and there are packages for t-SNE and UMAP.

```matlab
% Perform PCA
[coeff, score, latent] = pca(scaledData');

% Visualize first two principal components
figure;
scatter(score(:,1), score(:,2), 10, 'filled');
title('PCA of Gene Expression');
xlabel('PC1');
ylabel('PC2');

% Perform t-SNE
tsneData = tsne(scaledData', 'Algorithm', 'barneshut', 'NumDimensions', 2);
figure;
scatter(tsneData(:,1), tsneData(:,2), 10, 'filled');
title('t-SNE of Gene Expression');
xlabel('t-SNE 1');
ylabel('t-SNE 2');
```

For UMAP, use the MATLAB Toolbox "umap" available on MATLAB File Exchange.

### 5. Clustering Cells into Subpopulations

Clustering reveals distinct cell populations based on transcriptional similarity. MATLAB’s `kmeans` or `cluster` functions are useful, and hierarchical clustering can visualize relationships among clusters.

```matlab
% k-means clustering
numClusters = 5; % Set the number of clusters
[idx, C] = kmeans(score(:, 1:10), numClusters); % Clustering on the top 10 PCs

% Visualize clusters
figure;
gscatter(score(:,1), score(:,2), idx);
title('Cell Clusters in PCA Space');
xlabel('PC1');
ylabel('PC2');
```

### 6. Differential Gene Expression

To identify marker genes that characterize each cluster, compare gene expression across clusters using t-tests or ANOVA.

```matlab
% Perform differential expression analysis
cluster1 = expressionData(:, idx == 1);
cluster2 = expressionData(:, idx == 2);
[~, pvals] = ttest2(cluster1', cluster2', 'Vartype', 'unequal');
adjustedPvals = mafdr(pvals, 'BHFDR', true); % FDR correction
```

### 7. Visualization and Annotation

Heatmaps and dot plots help visualize expression patterns. MATLAB’s `heatmap` function can be customized to highlight key genes and clusters.

```matlab
% Select top marker genes for clusters
topGenes = geneNames(adjustedPvals < 0.05); % Significant genes
expressionSubset = scaledData(strcmp(geneNames, topGenes), :);

% Generate heatmap
figure;
heatmap(expressionSubset, 'Colormap', parula, 'GridVisible', 'off');
title('Expression Heatmap of Marker Genes');
xlabel('Cells');
ylabel('Genes');
```

### 8. Gene Set Enrichment and Pathway Analysis (Optional)

For pathway enrichment, you can use Gene Set Enrichment Analysis (GSEA) with MATLAB’s bioinformatics toolbox if gene sets are preloaded.

```matlab
% Load gene sets and perform GSEA (requires additional libraries for gene sets)
geneSet = load('pathway_genes.mat');
gseaResults = geneont('File', 'gene_ontology.obo');
```

### Summary

MATLAB provides robust options for preprocessing, clustering, and visualizing high-dimensional transcriptional data, especially with basic gene expression analysis workflows. While other languages like Python offer more single-cell-specific libraries, MATLAB is suitable for custom analysis pipelines and advanced visualization, particularly for researchers already comfortable with its environment. For high-level analysis, combining MATLAB with tools like R (e.g., Seurat) or Python (e.g., Scanpy) can be a powerful approach to extract biological insights from single-cell RNA-seq data.

### :cactus:Python snippet

In Python, high-dimensional transcriptional profiling of cells, especially for single-cell RNA sequencing (scRNA-seq) data, is typically handled using libraries like **Scanpy** and **Seurat (through the SeuratDisk package to bridge R and Python)**. Scanpy is particularly powerful, as it provides comprehensive preprocessing, visualization, and clustering tools within a single framework. Here's a guide to using Python for high-dimensional transcriptional profiling:

### 1. Installing and Importing Necessary Packages

To get started, install Scanpy and other essential libraries:

```bash
pip install scanpy[leiden] anndata matplotlib seaborn
```

Then, import them in your Python environment:

```python
import scanpy as sc
import anndata as ad
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
```

### 2. Loading and Preprocessing Data

Typically, scRNA-seq data is stored as a matrix (cells × genes). If you have a CSV file or other common formats, you can load it into an AnnData object (the primary data structure in Scanpy):

```python
# Load data from a CSV file where rows are genes and columns are cells
adata = sc.read_csv('gene_expression.csv').transpose()

# Load gene and cell names (optional)
adata.var_names = np.loadtxt('gene_names.csv', dtype=str)  # Genes
adata.obs_names = np.loadtxt('cell_names.csv', dtype=str)  # Cells
```

### 3. Quality Control and Filtering

For scRNA-seq, it’s essential to filter cells and genes based on counts. This step removes dead or low-quality cells and non-informative genes.

```python
# Filter genes that are expressed in less than 3 cells and cells with fewer than 200 genes
sc.pp.filter_genes(adata, min_cells=3)
sc.pp.filter_cells(adata, min_genes=200)

# Calculate QC metrics and filter based on quality
adata.var['mt'] = adata.var_names.str.startswith('MT-')  # Mitochondrial genes
sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)

# Filter cells based on mitochondrial gene content and total counts
adata = adata[adata.obs['pct_counts_mt'] < 5, :]
```

### 4. Normalization and Scaling

Normalize gene expression counts and scale each gene to have unit variance:

```python
# Normalize data so each cell has the same total count
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)  # Log-transform data

# Scale data to unit variance and zero mean for PCA
sc.pp.scale(adata, max_value=10)
```

### 5. Dimensionality Reduction (PCA, t-SNE, UMAP)

Dimensionality reduction is essential for visualizing high-dimensional data in 2D or 3D.

```python
# PCA
sc.tl.pca(adata, svd_solver='arpack')

# Plot the explained variance ratio of the PCs
sc.pl.pca_variance_ratio(adata, log=True)

# Compute neighborhood graph for clustering and embedding
sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40)

# UMAP
sc.tl.umap(adata)
sc.pl.umap(adata, color='total_counts')

# t-SNE (optional)
sc.tl.tsne(adata)
sc.pl.tsne(adata, color='total_counts')
```

### 6. Clustering Cells into Subpopulations

Scanpy provides several clustering methods, including the **Leiden algorithm**, which is particularly effective for scRNA-seq data.

```python
# Cluster cells using the Leiden algorithm
sc.tl.leiden(adata, resolution=0.5)  # Adjust resolution as needed
sc.pl.umap(adata, color='leiden')  # Visualize clusters on UMAP plot
```

### 7. Differential Gene Expression

Identify marker genes that are differentially expressed between clusters.

```python
# Identify marker genes for each cluster
sc.tl.rank_genes_groups(adata, 'leiden', method='t-test')  # Can use other methods like 'wilcoxon'
sc.pl.rank_genes_groups(adata, n_genes=10, sharey=False)
```

### 8. Visualization of Gene Expression

To visualize expression levels of specific genes or marker genes, use Violin plots or heatmaps.

```python
# Violin plot for a specific gene
sc.pl.violin(adata, ['GeneA', 'GeneB'], groupby='leiden')

# Heatmap for top marker genes across clusters
sc.pl.rank_genes_groups_heatmap(adata, n_genes=10, groupby='leiden', cmap='viridis')
```

### 9. Gene Set Enrichment and Pathway Analysis

Gene Set Enrichment Analysis (GSEA) can link transcriptional changes to biological pathways. For this, Scanpy doesn’t have native support, so you may use packages like **gseapy**.

```bash
pip install gseapy
import gseapy as gp

# Run enrichment analysis for a specific list of genes
gene_list = adata.var_names[adata.var['highly_variable']].tolist()
enr = gp.enrichr(gene_list=gene_list, gene_sets='KEGG_2016', organism='Human', outdir='enrichr_results')
```

### 10. Integrating and Analyzing Additional Data Modalities

For more advanced analyses, consider integrating other omics data (e.g., spatial transcriptomics or ATAC-seq). Libraries like **scvi-tools** and **Anndata** enable complex multimodal analyses.

### Full Example of Workflow

```python
import scanpy as sc
import matplotlib.pyplot as plt

# Load and preprocess
adata = sc.read_csv('gene_expression.csv').transpose()
sc.pp.filter_genes(adata, min_cells=3)
sc.pp.filter_cells(adata, min_genes=200)
adata.var['mt'] = adata.var_names.str.startswith('MT-')
sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], inplace=True)
adata = adata[adata.obs['pct_counts_mt'] < 5, :]
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.pp.scale(adata, max_value=10)

# Dimensionality reduction and clustering
sc.tl.pca(adata)
sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40)
sc.tl.umap(adata)
sc.tl.leiden(adata, resolution=0.5)

# Plotting
sc.pl.umap(adata, color=['leiden', 'total_counts', 'pct_counts_mt'])

# Differential expression
sc.tl.rank_genes_groups(adata, 'leiden', method='t-test')
sc.pl.rank_genes_groups(adata, n_genes=5, sharey=False)
```

### Summary

This pipeline provides an efficient way to process and analyze high-dimensional transcriptional data in Python using Scanpy. The workflow can be extended for multimodal analysis by integrating with additional libraries and tools.

### :cactus:R snippet

In R, high-dimensional transcriptional profiling of cells is often handled using **Seurat**, which is a comprehensive toolkit for single-cell RNA-seq (scRNA-seq) analysis. It supports various steps from preprocessing to clustering, visualization, and differential gene expression. Below is a structured guide to using Seurat for scRNA-seq data in R.

### 1. Installing and Loading Necessary Packages

Install **Seurat** and other essential libraries:

```r
if (!requireNamespace("BiocManager", quietly = TRUE))
    install.packages("BiocManager")
BiocManager::install("Seurat")
```

Then, load the libraries:

```r
library(Seurat)
library(dplyr)
library(Matrix)
library(ggplot2)
```

### 2. Loading Data

Typically, scRNA-seq data is provided as a count matrix (genes × cells). You can load this directly into Seurat:

```r
# Loading raw counts matrix (e.g., from a CSV file)
counts <- read.csv("gene_expression.csv", row.names = 1)
seurat_obj <- CreateSeuratObject(counts = counts, project = "scRNAseq", min.cells = 3, min.features = 200)
```

### 3. Quality Control and Filtering

Filter cells and genes based on quality metrics like the number of genes per cell, counts per cell, and mitochondrial gene percentage.

```r
# Calculate mitochondrial gene percentage (assuming "MT-" prefix for mitochondrial genes)
seurat_obj[["percent.mt"]] <- PercentageFeatureSet(seurat_obj, pattern = "^MT-")

# Plot QC metrics
VlnPlot(seurat_obj, features = c("nFeature_RNA", "nCount_RNA", "percent.mt"), ncol = 3)

# Filter cells based on QC metrics
seurat_obj <- subset(seurat_obj, subset = nFeature_RNA > 200 & nFeature_RNA < 2500 & percent.mt < 5)
```

### 4. Normalization and Scaling

Normalize the data and scale it for principal component analysis (PCA):

```r
# Normalize data
seurat_obj <- NormalizeData(seurat_obj, normalization.method = "LogNormalize", scale.factor = 10000)

# Find variable features
seurat_obj <- FindVariableFeatures(seurat_obj, selection.method = "vst", nfeatures = 2000)

# Scaling the data (centering and scaling)
seurat_obj <- ScaleData(seurat_obj, features = rownames(seurat_obj))
```

### 5. Dimensionality Reduction (PCA, UMAP, t-SNE)

Perform PCA and use it to visualize the data in lower dimensions (e.g., with UMAP or t-SNE).

```r
# Run PCA
seurat_obj <- RunPCA(seurat_obj, features = VariableFeatures(object = seurat_obj))

# Examine PCA results
print(seurat_obj[["pca"]], dims = 1:5, nfeatures = 5)
VizDimLoadings(seurat_obj, dims = 1:2, reduction = "pca")

# Plot PCA results
DimPlot(seurat_obj, reduction = "pca")

# Run UMAP and t-SNE for visualization
seurat_obj <- RunUMAP(seurat_obj, dims = 1:10)
seurat_obj <- RunTSNE(seurat_obj, dims = 1:10)

# Plot UMAP and t-SNE
DimPlot(seurat_obj, reduction = "umap")
DimPlot(seurat_obj, reduction = "tsne")
```

### 6. Clustering

Cluster cells using the **Louvain** or **Leiden** algorithm, which Seurat supports through its `FindClusters` function.

```r
# Find neighbors and clusters
seurat_obj <- FindNeighbors(seurat_obj, dims = 1:10)
seurat_obj <- FindClusters(seurat_obj, resolution = 0.5)

# Plot clusters on UMAP
DimPlot(seurat_obj, reduction = "umap", group.by = "seurat_clusters")
```

### 7. Differential Gene Expression

Identify marker genes for each cluster, which can help determine cell types.

```r
# Find all markers for each cluster
cluster_markers <- FindAllMarkers(seurat_obj, only.pos = TRUE, min.pct = 0.25, logfc.threshold = 0.25)

# View top markers for each cluster
top_markers <- cluster_markers %>% group_by(cluster) %>% top_n(n = 10, wt = avg_log2FC)
print(top_markers)

# Visualize top marker genes
FeaturePlot(seurat_obj, features = c("GeneA", "GeneB"), cols = c("lightgrey", "blue"))
VlnPlot(seurat_obj, features = c("GeneA", "GeneB"))
```

### 8. Gene Set Enrichment and Pathway Analysis

You can analyze enriched pathways or gene sets using external libraries, such as **clusterProfiler** or **msigdbr** for accessing the Molecular Signatures Database (MSigDB):

```r
BiocManager::install("clusterProfiler")
library(clusterProfiler)
library(msigdbr)

# Run enrichment analysis on a list of significant genes
significant_genes <- top_markers$gene
enrich_results <- enrichGO(gene = significant_genes, OrgDb = org.Hs.eg.db, ont = "BP", pAdjustMethod = "BH", pvalueCutoff = 0.01)

# Visualize results
dotplot(enrich_results)
```

### 9. Visualize Gene Expression

Visualize expression patterns of specific genes across clusters using **Violin plots** and **Heatmaps**.

```r
# Violin plots for specific genes
VlnPlot(seurat_obj, features = c("GeneA", "GeneB"), group.by = "seurat_clusters")

# Heatmap of top marker genes
DoHeatmap(seurat_obj, features = top_markers$gene) + NoLegend()
```

### 10. Saving and Sharing Results

Save the processed Seurat object or export the cluster markers for further analysis.

```r
# Save Seurat object
saveRDS(seurat_obj, file = "seurat_analysis.rds")

# Export markers to CSV
write.csv(cluster_markers, file = "cluster_markers.csv")
```

### Full Workflow Example

```r
library(Seurat)
library(dplyr)
library(Matrix)
library(ggplot2)

# Load data
counts <- read.csv("gene_expression.csv", row.names = 1)
seurat_obj <- CreateSeuratObject(counts = counts, project = "scRNAseq", min.cells = 3, min.features = 200)

# Quality control
seurat_obj[["percent.mt"]] <- PercentageFeatureSet(seurat_obj, pattern = "^MT-")
seurat_obj <- subset(seurat_obj, subset = nFeature_RNA > 200 & nFeature_RNA < 2500 & percent.mt < 5)

# Normalize, find variable features, scale data
seurat_obj <- NormalizeData(seurat_obj)
seurat_obj <- FindVariableFeatures(seurat_obj)
seurat_obj <- ScaleData(seurat_obj)

# PCA, UMAP, and clustering
seurat_obj <- RunPCA(seurat_obj)
seurat_obj <- RunUMAP(seurat_obj, dims = 1:10)
seurat_obj <- FindNeighbors(seurat_obj, dims = 1:10)
seurat_obj <- FindClusters(seurat_obj, resolution = 0.5)

# Visualization and marker identification
DimPlot(seurat_obj, reduction = "umap", group.by = "seurat_clusters")
cluster_markers <- FindAllMarkers(seurat_obj, only.pos = TRUE)

# Save results
write.csv(cluster_markers, file = "cluster_markers.csv")
```

### Summary

This pipeline covers a typical workflow for high-dimensional transcriptional profiling in R using Seurat. The workflow includes data loading, quality control, normalization, dimensionality reduction, clustering, differential expression, and visualization, allowing you to identify and interpret cellular subpopulations.

### :cactus:Julia snippet

In Julia, the high-dimensional transcriptional profiling of cells can be handled with the **SingleCell** and **BioJulia** ecosystems, along with data analysis and visualization packages such as **DataFrames.jl**, **Plots.jl**, and **Clustering.jl**. Below is a guide on performing single-cell RNA-seq (scRNA-seq) analysis in Julia.

### 1. Setting Up the Environment

Install the necessary packages:

```julia
using Pkg
Pkg.add(["DataFrames", "CSV", "SingleCell", "Plots", "Clustering", "StatsBase", "Distances"])
```

Then, load the packages:

```julia
using DataFrames
using CSV
using SingleCell
using Plots
using Clustering
using StatsBase
using Distances
```

### 2. Loading Data

Load your gene expression data (typically a counts matrix with genes as rows and cells as columns) into Julia. The **DataFrames.jl** and **CSV.jl** packages can handle this easily:

```julia
# Load the data from a CSV file
data = CSV.read("gene_expression.csv", DataFrame)
# Convert the data into a matrix if needed
counts_matrix = Matrix(data[:, 2:end])  # Assuming the first column is gene names
genes = data[:, 1]  # Store gene names separately
```

### 3. Preprocessing and Quality Control

For quality control, filter cells and genes based on criteria such as minimum expression level, number of detected genes per cell, or mitochondrial gene content.

```julia
# Calculate QC metrics (e.g., gene count per cell)
cell_counts = sum(counts_matrix, dims=1)
gene_counts = sum(counts_matrix, dims=2)

# Filter cells and genes based on these metrics
filtered_cells = findall(cell_counts .> 200)  # Example threshold of 200 detected genes
filtered_genes = findall(gene_counts .> 10)   # Example threshold of 10 total counts

# Filter the matrix
counts_matrix_filtered = counts_matrix[filtered_genes, filtered_cells]
genes_filtered = genes[filtered_genes]
```

### 4. Normalization and Scaling

Normalize the data, which involves adjusting for differences in library size across cells.

```julia
# Normalize data by total counts per cell
normalized_counts = counts_matrix_filtered ./ cell_counts[filtered_cells]

# Log transformation for stabilization
log_normalized_counts = log1p.(normalized_counts)
```

### 5. Dimensionality Reduction (PCA)

Use Principal Component Analysis (PCA) to reduce the dimensionality of the data for clustering and visualization.

```julia
using LinearAlgebra

# Center the data by subtracting the mean
data_centered = log_normalized_counts .- mean(log_normalized_counts, dims=2)

# Compute the covariance matrix
cov_matrix = cov(data_centered)

# Perform PCA using eigen decomposition
eigenvalues, eigenvectors = eigen(cov_matrix)
pca_data = data_centered' * eigenvectors[:, 1:10]  # First 10 PCs
```

### 6. Clustering Cells

Use k-means clustering or other clustering algorithms to identify cell subpopulations.

```julia
# Run k-means clustering on the PCA-reduced data
k = 5  # Example: 5 clusters
assignments = kmeans(pca_data, k)

# Visualize clusters using the first two principal components
scatter(pca_data[:, 1], pca_data[:, 2], color=assignments.assignments, legend=false)
```

### 7. Visualization with t-SNE or UMAP

Currently, **TSne.jl** or **UMAP.jl** packages are not as fully featured as in Python, but you can use approximate t-SNE or UMAP implementations in Julia.

```julia
using MultivariateStats  # For TSne

# Run t-SNE on the PCA data
tsne_data = tsne(pca_data)

# Plot the t-SNE results
scatter(tsne_data[:, 1], tsne_data[:, 2], color=assignments.assignments, legend=false)
```

### 8. Differential Gene Expression

To find marker genes for each cluster, calculate the mean expression in each cluster and identify genes with significant differences between clusters.

```julia
# Compute mean expression per cluster
mean_expression = reduce(hcat, [mean(log_normalized_counts[:, assignments.assignments .== i], dims=2) for i in 1:k])

# Identify top genes for each cluster by variance across clusters
top_genes = sortperm(var(mean_expression, dims=2), rev=true)[1:10]  # Top 10 genes

# Plot expression of top genes
bar(genes_filtered[top_genes], mean_expression[top_genes, :], xlabel="Genes", ylabel="Expression")
```

### Full Workflow Example

Here's the workflow consolidated:

```julia
using DataFrames, CSV, SingleCell, Plots, Clustering, StatsBase, Distances, LinearAlgebra

# Load data
data = CSV.read("gene_expression.csv", DataFrame)
counts_matrix = Matrix(data[:, 2:end])
genes = data[:, 1]

# Quality control
cell_counts = sum(counts_matrix, dims=1)
gene_counts = sum(counts_matrix, dims=2)
filtered_cells = findall(cell_counts .> 200)
filtered_genes = findall(gene_counts .> 10)
counts_matrix_filtered = counts_matrix[filtered_genes, filtered_cells]
genes_filtered = genes[filtered_genes]

# Normalization
normalized_counts = counts_matrix_filtered ./ cell_counts[filtered_cells]
log_normalized_counts = log1p.(normalized_counts)

# PCA
data_centered = log_normalized_counts .- mean(log_normalized_counts, dims=2)
cov_matrix = cov(data_centered)
eigenvalues, eigenvectors = eigen(cov_matrix)
pca_data = data_centered' * eigenvectors[:, 1:10]

# Clustering
k = 5
assignments = kmeans(pca_data, k)

# Visualization
scatter(pca_data[:, 1], pca_data[:, 2], color=assignments.assignments, legend=false, title="PCA Clusters")

# Differential Expression
mean_expression = reduce(hcat, [mean(log_normalized_counts[:, assignments.assignments .== i], dims=2) for i in 1:k])
top_genes = sortperm(var(mean_expression, dims=2), rev=true)[1:10]
bar(genes_filtered[top_genes], mean_expression[top_genes, :], xlabel="Genes", ylabel="Expression")
```

### Summary

This workflow provides a structured approach to perform high-dimensional transcriptional profiling in Julia. This approach handles loading data, quality control, normalization, dimensionality reduction, clustering, and differential gene expression analysis. Although Julia’s single-cell ecosystem is less developed than Python’s, this approach utilizes general data science libraries and Julia's high-performance capabilities to conduct this analysis.
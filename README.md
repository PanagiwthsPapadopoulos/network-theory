# 🔍 Distance Quality Maximization for Graph Clustering  

## 📌 Project Overview  
This project implements an algorithm to maximize the **Distance Quality Function (Q_d)**, a metric used for clustering nodes in a graph. The algorithm aims to find an optimal partition of the graph such that the observed distances between nodes within clusters are minimized while the expected distances in a random graph are maximized. This approach improves the clustering quality for graphs with complex structures and multiple connected components.  

## 📂 Files and Directories  
- 🖧 **network.py**: The main script that implements the algorithm for maximizing distance quality and includes various clustering methods.  
- 📜 **README.md**: This file. Provides an overview and instructions.  
- 📄 **NetworkTheory.pdf**: The task description  

## ✨ Features  
### 📏 **Distance Quality Function**  
- 📊 Implementation of the **Q_d** metric to evaluate clustering quality.  
- ⚡ Supports both direct computation and an **approximated version using ERCQ**.  

### 🏗 **Clustering Techniques**  
- 🏆 **Proto-cluster initialization** based on:  
  - 🔝 High-degree nodes  
  - 🔄 High betweenness centrality  
  - 📈 Clustering coefficient  
- 🔗 **Expansion of clusters** using neighboring nodes.  
- 🔄 **Refinement through cluster merging** to improve **Q_d**.  

### 🎲 **Random Graph Generation**  
- 🛠 Generate **random graphs** with preserved degree distributions for comparison.  

### 🎨 **Visualization**  
- 🖼 **Graph visualizations** with clusters **color-coded** for clarity.  

## ⚙️ How the Algorithm Works  
### 🏁 **Proto-cluster Initialization**  
Nodes with specific structural properties are selected to form **initial clusters (proto-clusters)**.  

### 🔍 **Cluster Expansion**  
Unassigned nodes are iteratively added to the nearest cluster based on **distance minimization criteria**.  

### 🔄 **Cluster Refinement through Merging**  
Clusters are merged if **Q_d improves**, balancing between **over-merging and under-merging**.  

### ✅ **Final Output**  
The algorithm outputs a **partition of the graph into clusters** with the highest **Q_d**.  

## 📦 Prerequisites  
- 🐍 **Python 3.10 or higher**  
- 📚 Required libraries: **networkx, matplotlib, numpy**  

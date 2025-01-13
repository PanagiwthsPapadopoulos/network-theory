# README: Distance Quality Maximization for Graph Clustering

## Project Overview  
This project implements an algorithm to maximize the **Distance Quality Function (Q_d)**, a metric used for clustering nodes in a graph. The algorithm aims to find an optimal partition of the graph such that the observed distances between nodes within clusters are minimized while the expected distances in a random graph are maximized. This approach improves the clustering quality for graphs with complex structures and multiple connected components.

---

## Files and Directories  

- **`network.py`**: The main script that implements the algorithm for maximizing distance quality and includes various clustering methods.  
- **`README.md`**: This file. Provides an overview and instructions.  
- **`NetworkTheory.pdf`**: The task description  

---

## Features  

1. **Distance Quality Function**  
   - Implementation of the Q_d metric to evaluate clustering quality.
   - Supports both direct computation and an **approximated version** using ERCQ.

2. **Clustering Techniques**  
   - Proto-cluster initialization based on:  
     - High-degree nodes  
     - High betweenness centrality  
     - Clustering coefficient  
   - Expansion of clusters using neighboring nodes.  
   - Refinement through cluster merging to improve Q_d.  

3. **Random Graph Generation**  
   - Generate random graphs with preserved degree distributions for comparison.  

4. **Visualization**  
   - Graph visualizations with clusters color-coded for clarity.  

---

## How the Algorithm Works  

1. **Proto-cluster Initialization**  
   Nodes with specific structural properties are selected to form initial clusters (proto-clusters).

2. **Cluster Expansion**  
   Unassigned nodes are iteratively added to the nearest cluster based on distance minimization criteria.

3. **Cluster Refinement through Merging**  
   Clusters are merged if the Q_d improves, with a balance between over-merging and under-merging.

4. **Final Output**  
   The algorithm outputs a partition of the graph into clusters with the highest Q_d.

---

## Prerequisites  

- Python 3.10 or higher  
- Required libraries: `networkx`, `matplotlib`, `numpy`


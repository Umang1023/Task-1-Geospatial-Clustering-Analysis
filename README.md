# Geospatial Clustering Analysis

## Project Overview

The project consists of two main parts:

1. **Implementation Problem**: A Python-based solution for clustering geospatial data (latitude and longitude coordinates) using various clustering algorithms. The implementation includes data loading, preprocessing, model training, evaluation, and visualization.

2. **System Design Problem**: A system design document outlining a cloud-based solution for a quick commerce company that wants to optimize order assignment to delivery personnel using geospatial clustering.

## Implementation Details

### Features

- Data loading and preprocessing
- Implementation of four clustering algorithms:
  - K-Means
  - DBSCAN (Density-Based Spatial Clustering of Applications with Noise)
  - Agglomerative Clustering
  - Gaussian Mixture Model (GMM)
- Model evaluation using metrics:
  - Silhouette Score
  - Davies-Bouldin Index
  - Calinski-Harabasz Score
- Visualization of clustering results:
  - 2D scatter plots
  - Interactive map visualization using Folium

### Code Structure

The implementation follows Object-Oriented Programming principles with the following classes:

- `DataLoader`: Handles data loading from CSV files
- `Standardizer`: Standardizes feature values
- `ClusteringModels`: Implements and trains different clustering models
- `EvaluationMetrics`: Evaluates clustering performance
- `MapPlotter`: Visualizes clusters on an interactive map
- `ModelPlotter`: Creates scatter plots of clustering results

## Approach and Methodology

### Data Understanding and Preparation

1. **Data Loading**: The dataset contains geographical coordinates (latitude and longitude) for various locations. The data is loaded using pandas and initially explored to understand its structure and distribution.

2. **Feature Standardization**: Geographical coordinates are standardized using `StandardScaler` to ensure that the clustering algorithms are not biased by the scale of the data. This is particularly important for distance-based algorithms like K-Means and DBSCAN.

### Clustering Model Selection

Four different clustering algorithms were implemented to compare their performance on geospatial data:

1. **K-Means**: A centroid-based algorithm that partitions data into K clusters. K=8 was chosen as an initial parameter based on exploratory data analysis.

2. **DBSCAN**: A density-based algorithm that identifies clusters as dense regions separated by sparse regions. Parameters eps=0.7 and min_samples=10 were selected to identify meaningful clusters in the dataset.

3. **Agglomerative Clustering**: A hierarchical clustering approach that builds nested clusters by merging or splitting them. A parameter of 5 clusters was chosen for this implementation.

4. **Gaussian Mixture Model (GMM)**: A probabilistic model that assumes data is generated from a mixture of Gaussian distributions. 4 components were used in the implementation.

### Model Evaluation

The clustering models were evaluated using three different metrics:

1. **Silhouette Score**: Measures how similar an object is to its own cluster compared to other clusters. Higher values indicate better-defined clusters.

2. **Davies-Bouldin Index**: Evaluates the average similarity between clusters. Lower values indicate better clustering.

3. **Calinski-Harabasz Score**: Measures the ratio of between-cluster dispersion to within-cluster dispersion. Higher values indicate better-defined clusters.

### Visualization

Two visualization approaches were implemented:

1. **2D Scatter Plots**: Simple visualizations of longitude vs. latitude with clusters colored differently.

2. **Interactive Map**: Using Folium, clusters are visualized on an actual map, providing a more intuitive understanding of the geographical distribution.

## Assumptions

1. **Euclidean Distance**: The implementation assumes that Euclidean distance is an appropriate measure for geographical data. For more accurate results, haversine distance could be considered.

2. **Optimal Number of Clusters**: The number of clusters for K-Means, Agglomerative Clustering, and GMM were chosen based on initial exploration. A more rigorous approach would involve techniques like the elbow method or silhouette analysis.

3. **Feature Importance**: The implementation assumes that latitude and longitude have equal importance in the clustering process. In real-world scenarios, different weights might be assigned to these features.

## Hurdles and Challenges

1. **Parameter Tuning**: Finding optimal parameters for each clustering algorithm, especially for DBSCAN (eps and min_samples), was challenging and required multiple iterations.

2. **Evaluation Metrics**: Determining the most appropriate evaluation metrics for geospatial clustering was not straightforward since traditional metrics may not fully capture the geographical context.

3. **Visualization Limitations**: Creating meaningful visualizations of geospatial clusters that effectively communicate the results required careful consideration of color schemes and marker styles.

4. **Path Management**: Ensuring correct file paths for data loading and saving visualizations across different environments required careful attention to detail.

## Results and Findings

Based on the evaluation metrics and visual inspection, the [KMeans] was identified as the most effective clustering method for this dataset. Key findings include:

**Metric Performance**: [KMeans] achieved the highest Silhouette Score ([0.5641613628678133]), lowest Davies-Bouldin Index ([0.6421843177277118]).

## Installation and Usage

### Prerequisites

- Python 3.8+
- Required Python packages (see `requirements.txt`)

### Setup

1. Clone the repository:
   ```
   git clone https://github.com/Umang1023/Task-1-Geospatial-Clustering-Analysis.git
   cd geospatial-clustering
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Update the file paths in `main.py` to match your local environment.

4. Run the script:
   ```
   python main.py
   ```




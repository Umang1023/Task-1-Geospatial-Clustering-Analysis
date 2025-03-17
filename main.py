import pandas as pd
import numpy as np
import webbrowser
import os
import matplotlib.pyplot as plt
import folium
from folium.plugins import MarkerCluster
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

# Class for handling data loading
class DataLoader:
    def __init__(self, file_path, separator=';'):
        self.file_path = file_path
        self.separator = separator
        self.df = None

    def load_data(self):
        self.df = pd.read_csv(self.file_path, sep=self.separator)
        return self.df

# Class for standardizing features
class Standardizer:
    def __init__(self):
        self.scaler = StandardScaler()
    
    def fit_transform(self, df, features):
        X = df[features].values
        return self.scaler.fit_transform(X)

# Class for implementing clustering models
class ClusteringModels:
    def __init__(self, X_scaled):
        self.X_scaled = X_scaled
        self.models = {
            'KMeans': KMeans(n_clusters=8, random_state=42),
            'DBSCAN': DBSCAN(eps=0.7, min_samples=10),
            'Agglomerative': AgglomerativeClustering(n_clusters=5),
            'GMM': GaussianMixture(n_components=4, random_state=42)
        }
        self.labels = {}
    
    def fit_models(self):
        for name, model in self.models.items():
            model.fit(self.X_scaled)
            self.labels[name] = model.labels_ if hasattr(model, 'labels_') else model.predict(self.X_scaled)
        return self.labels

# Class for evaluating clustering models
class EvaluationMetrics:
    @staticmethod
    def evaluate(X_scaled, labels):
        scores = {}
        for name, model_labels in labels.items():
            if len(set(model_labels)) > 1:
                scores[name] = {
                    'Silhouette Score': silhouette_score(X_scaled, model_labels),
                    'Davies-Bouldin Index': davies_bouldin_score(X_scaled, model_labels),
                    'Calinski-Harabasz Score': calinski_harabasz_score(X_scaled, model_labels)
                }
        return scores


class MapPlotter:
    def __init__(self, df):
        self.df = df

    def plot_map(self, filename="D:/ML Assignment/CSV/clusters_map.html"):  # Use full path
        map_center = [self.df["Latitude"].mean(), self.df["Longitude"].mean()]
        map_ = folium.Map(location=map_center, zoom_start=12)
        marker_cluster = MarkerCluster().add_to(map_)

        points = self.df[["Latitude", "Longitude"]].values.tolist()
        for point in points:
            folium.CircleMarker(
                location=point,
                radius=5,
                color="blue",
                fill=True,
                fill_color="blue",
                fill_opacity=0.7,
                weight=1
            ).add_to(marker_cluster)
        
        map_.save(filename)
        print(f"Map saved as {filename}")

        # ✅ Open using absolute path
        webbrowser.open(f"file://{os.path.abspath(filename)}")

# Class for plotting clustering results
class ModelPlotter:
    def __init__(self, df):
        self.df = df

    def plot_clusters(self, labels, title, cmap='viridis'):
        plt.figure(figsize=(8, 6))
        plt.scatter(self.df['Longitude'], self.df['Latitude'], c=labels, cmap=cmap, alpha=0.5)
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.title(title)
        plt.show()

# Main execution
if __name__ == "__main__":
    # Load data
    data_loader = DataLoader(r'D:\ML Assignment\CSV\ML.csv')
    df = data_loader.load_data()
    
    # Standardize features
    standardizer = Standardizer()
    X_scaled = standardizer.fit_transform(df, ['Latitude', 'Longitude'])
    
    # Train clustering models
    clustering = ClusteringModels(X_scaled)
    labels = clustering.fit_models()
    
    # Evaluate models
    evaluator = EvaluationMetrics()
    scores = evaluator.evaluate(X_scaled, labels)
    for model, metrics in scores.items():
        print(f"{model} Clustering:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
    
    # Plot results
    plotter = ModelPlotter(df)
    for model_name, model_labels in labels.items():
        plotter.plot_clusters(model_labels, f"{model_name} Clustering")
    
    # Plot clusters on a map
    map_plotter = MapPlotter(df)
    map_plotter.plot_map()

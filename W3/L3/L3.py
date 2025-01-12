import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import warnings
from pyclustering.cluster.xmeans import xmeans
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer

#there was an error with the libraries, this was the fix
np.warnings = warnings

def iris():
    iris = load_iris()
    data = iris.data
    target = iris.target
    feature_names = iris.feature_names

    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans_labels = kmeans.fit_predict(data_scaled)

    initial_centers = kmeans_plusplus_initializer(data_scaled, 2).initialize()
    xmeans_instance = xmeans(data_scaled, initial_centers, 20)
    xmeans_instance.process()
    xmeans_labels = np.array([cluster_idx for cluster_idx, cluster in enumerate(xmeans_instance.get_clusters()) for _ in cluster])

    df = pd.DataFrame(data, columns=feature_names)
    df['kmeans_labels'] = kmeans_labels
    df['xmeans_labels'] = xmeans_labels

    kmeans_mi_scores = mutual_info_classif(data, kmeans_labels, discrete_features=False)
    xmeans_mi_scores = mutual_info_classif(data, xmeans_labels, discrete_features=False)

    feature_importance_kmeans = pd.DataFrame({
        'Feature': feature_names,
        'Mutual Information with K-Means Labels': kmeans_mi_scores
    })

    feature_importance_xmeans = pd.DataFrame({
        'Feature': feature_names,
        'Mutual Information with X-Means Labels': xmeans_mi_scores
    })

    print("Feature importance (K-Means):")
    print(feature_importance_kmeans.sort_values(by='Mutual Information with K-Means Labels', ascending=False))

    print("\nFeature importance (X-Means):")
    print(feature_importance_xmeans.sort_values(by='Mutual Information with X-Means Labels', ascending=False))

    plt.figure(figsize=(12, 5))

    #kmean
    plt.subplot(1, 2, 1)
    plt.scatter(data_scaled[:, 0], data_scaled[:, 1], c=kmeans_labels, cmap='viridis', s=50)
    plt.title('K-Means Clustering')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')

    #xmeans
    plt.subplot(1, 2, 2)
    plt.scatter(data_scaled[:, 0], data_scaled[:, 1], c=xmeans_labels, cmap='viridis', s=50)
    plt.title('X-Means Clustering')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    iris()
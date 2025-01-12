import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, KernelPCA
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def load_datasets():
    pima_columns = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"]
    pima_data = pd.read_csv("https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv", header=None, names=pima_columns)
    pima_features = pima_data.iloc[:, :-1]
    pima_target = pima_data.iloc[:, -1]

    digits_data = load_digits()
    digits_features = digits_data.data
    digits_target = digits_data.target

    scaler = StandardScaler()
    pima_features_scaled = scaler.fit_transform(pima_features)
    digits_features_scaled = scaler.fit_transform(digits_features)

    return pima_target, pima_features_scaled,digits_target, digits_features_scaled


def principal_component_analysis(pima_features_scaled, digits_features_scaled, nr_components=3):
    pca = PCA(n_components=nr_components)
    pima_pca = pca.fit_transform(pima_features_scaled)
    digits_pca = pca.fit_transform(digits_features_scaled)
    return pima_pca, digits_pca

def kernel_principal_component_analysis(pima_features_scaled, digits_features_scaled, nr_components=3):
    kpca = KernelPCA(n_components=nr_components, kernel='rbf', gamma=15)
    pima_kpca = kpca.fit_transform(pima_features_scaled)
    digits_kpca = kpca.fit_transform(digits_features_scaled)
    return pima_kpca, digits_kpca

def plotting(pima_pca, digits_pca, pima_kpca, digits_kpca, pima_target, digits_target):

    print("PCA Transformed Data (Pima Indians Diabetes):\n", pima_pca[:5])
    print("PCA Transformed Data (Digits):\n", digits_pca[:5])
    print("KPCA Transformed Data (Pima Indians Diabetes):\n", pima_kpca[:5])
    print("KPCA Transformed Data (Digits):\n", digits_kpca[:5])

    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    # Pima PCA
    axs[0, 0].scatter(pima_pca[:, 0], pima_pca[:, 1], c=pima_target, cmap='viridis', edgecolor='k', s=40)
    axs[0, 0].set_title('Pima Indians Diabetes: PCA (2 Components)')
    axs[0, 0].set_xlabel('PC1')
    axs[0, 0].set_ylabel('PC2')

    # Digits PCA
    scatter = axs[0, 1].scatter(digits_pca[:, 0], digits_pca[:, 1], c=digits_target, cmap='tab10', edgecolor='k', s=40)
    axs[0, 1].set_title('Digits Dataset: PCA (2 Components)')
    axs[0, 1].set_xlabel('PC1')
    axs[0, 1].set_ylabel('PC2')
    fig.colorbar(scatter, ax=axs[0, 1], label='Target Label')

    # Pima KPCA
    axs[1, 0].scatter(pima_kpca[:, 0], pima_kpca[:, 1], c=pima_target, cmap='viridis', edgecolor='k', s=40)
    axs[1, 0].set_title('Pima Indians Diabetes: KPCA (2 Components)')
    axs[1, 0].set_xlabel('PC1')
    axs[1, 0].set_ylabel('PC2')

    # Digits KPCA
    scatter = axs[1, 1].scatter(digits_kpca[:, 0], digits_kpca[:, 1], c=digits_target, cmap='tab10', edgecolor='k', s=40)
    axs[1, 1].set_title('Digits Dataset: KPCA (2 Components)')
    axs[1, 1].set_xlabel('PC1')
    axs[1, 1].set_ylabel('PC2')
    fig.colorbar(scatter, ax=axs[1, 1], label='Target Label')

    plt.tight_layout()
    plt.show()

    # First three principal components (Digits Dataset)
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(digits_pca[:, 0], digits_pca[:, 1], digits_pca[:, 2], c=digits_target, cmap='tab10', edgecolor='k', s=40)
    ax.set_title('Digits Dataset: PCA (3 Components)')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    plt.show()

    # First three kernel principal components (Digits Dataset)
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(digits_kpca[:, 0], digits_kpca[:, 1], digits_kpca[:, 2], c=digits_target, cmap='tab10', edgecolor='k', s=40)
    ax.set_title('Digits Dataset: KPCA (3 Components)')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    plt.show()

if __name__ == "__main__":
    # [0] - scaled
    # [1] - target
    pima_target, pima_features_scaled,digits_target, digits_features_scaled = load_datasets()
    pima_pca, digits_pca = principal_component_analysis(nr_components=3, pima_features_scaled=pima_features_scaled, digits_features_scaled = digits_features_scaled)
    pima_kpca, digits_kpca = kernel_principal_component_analysis(nr_components=3, pima_features_scaled=pima_features_scaled, digits_features_scaled = digits_features_scaled)
    plotting(pima_pca=pima_pca, digits_pca=digits_pca, pima_kpca=pima_kpca, digits_kpca=digits_kpca, pima_target=pima_target, digits_target=digits_target)

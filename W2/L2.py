import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import KNNImputer
from sklearn.feature_selection import SelectKBest, f_classif, chi2
from sklearn.model_selection import train_test_split


def read_csv_to_dataframe(file_path):
    try:
        df = pd.read_csv(file_path)
        return df
    except FileNotFoundError:
        print(f"Error: The file at {file_path} was not found.")


def save_matrix_as_heatmap(matrix, title, filename):
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        square=True,
        cbar_kws={"shrink": 0.8},
    )
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def save_plots(df, name):
    cov_matrix = df.cov()
    corr_matrix = df.corr()
    save_matrix_as_heatmap(
        cov_matrix, f"{name} covariance matrix", f"{name}_covariance_matrix.png"
    )
    save_matrix_as_heatmap(
        corr_matrix,
        f"{name} correlation matrix",
        f"{name}_correlation_matrix.png",
    )


def preprocess_diabetes(df):
    df.replace(0.0, np.nan, inplace=True)
    return df


def preprocess_iris(df):
    df.replace(0.0, np.nan, inplace=True)
    to_replace = {"Iris-setosa": 1.0, "Iris-versicolor": 2.0, "Iris-virginica": 3.0}
    for key in to_replace:
        df.replace(key, to_replace[key], inplace=True)
    return df


def impute(df):
    column_names = df.columns.tolist()
    imputer = KNNImputer(n_neighbors=5)
    imputed_df = imputer.fit_transform(df)
    df_imputed = pd.DataFrame(imputed_df, columns=column_names)
    print(df_imputed.head())
    return df_imputed




def Select_K_Best_iris(k_values_iris, df):
    X_iris = df.drop('class', axis=1)
    y_iris = df['class']
    for k in k_values_iris:
        selector = SelectKBest(score_func=f_classif, k=k)
        X_selected_iris = selector.fit_transform(X_iris, y_iris)
        print(f"Iris Dataset: Selected {k} features: {X_selected_iris.shape[1]} features selected")


def Select_K_Best_diabetes(k_values_pima, df):
    X_pima = df.drop('Outcome', axis=1)
    y_pima = df['Outcome']

    for k in k_values_pima:
        selector = SelectKBest(score_func=chi2, k=k)
        X_selected_pima = selector.fit_transform(X_pima, y_pima)
        print(f"Pima Dataset: Selected {k} features: {X_selected_pima.shape[1]} features selected")


if __name__ == "__main__":
    df_diabetes = read_csv_to_dataframe(
        file_path=r"W2\data\diabetes.csv"
    )
    preprocess_diabetes = preprocess_diabetes(df_diabetes)
    imputed_diabetes = impute(preprocess_diabetes)
    #save_plots(imputed_diabetes, "diabetes")
    Select_K_Best_diabetes([1, 3], imputed_diabetes)

    df_iris = read_csv_to_dataframe(
        file_path=r"W2\data\iris.csv"
    )
    preprocess_iris = preprocess_iris(df_iris)
    imputed_iris = impute(preprocess_iris)
    #save_plots(imputed_iris, "iris")
    Select_K_Best_iris([1,2], imputed_iris)

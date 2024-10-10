import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.impute import KNNImputer


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


if __name__ == "__main__":
    df_diabetes = read_csv_to_dataframe(
        file_path="/home/istvan/Masters/Year1/Sem1/IAADPI/W2/data/diabetes.csv"
    )
    preprocess_diabetes = preprocess_diabetes(df_diabetes)
    imputed_diabetes = impute(preprocess_diabetes)
    save_plots(imputed_diabetes, "diabetes")

    df_iris = read_csv_to_dataframe(
        file_path="/home/istvan/Masters/Year1/Sem1/IAADPI/W2/data/iris.csv"
    )
    preprocess_iris = preprocess_iris(df_iris)
    imputed_iris = impute(preprocess_iris)
    save_plots(imputed_iris, "iris")

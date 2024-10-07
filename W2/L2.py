import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def read_csv_to_dataframe(file_path):
    try:
        df = pd.read_csv(file_path)
        return df
    except FileNotFoundError:
        print(f"Error: The file at {file_path} was not found.")
    except pd.errors.EmptyDataError:
        print("Error: The file is empty.")
    except pd.errors.ParserError:
        print("Error: There was a problem parsing the file.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


def save_matrix_as_heatmap(matrix, title, filename):
    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix, annot=True, fmt='.2f', cmap='coolwarm', square=True, cbar_kws={"shrink": .8})
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()



if __name__ == "__main__":
    file_path = "/home/istvan/Masters/Year1/Sem1/IAADPI/W2/diabetes.csv"
    df = read_csv_to_dataframe(file_path=file_path)
    cov_matrix = df.cov()
    corr_matrix = df.corr()
    print(f"{cov_matrix}\n\n\n{corr_matrix}")
    print(cov_matrix["Insulin"]["Insulin"])
    save_matrix_as_heatmap(cov_matrix, 'Covariance Matrix', 'covariance_matrix.png')
    save_matrix_as_heatmap(corr_matrix, 'Correlation Matrix', 'correlation_matrix.png')

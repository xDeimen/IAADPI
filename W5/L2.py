import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes, load_iris

# Function to evaluate and visualize regression results
def evaluate_model(y_test, y_pred, title):
    r2 = r2_score(y_test, y_pred)
    print(f"{title} - R2 Score: {r2:.4f}")
    plt.scatter(y_test, y_pred, alpha=0.7, label='Predicted vs True')
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], '--r', label='Ideal Fit')
    plt.xlabel("True Values")
    plt.ylabel("Predicted Values")
    plt.title(f"{title} - True vs Predicted")
    plt.legend()
    plt.show()

def diab():
    diabetes = load_diabetes()
    X_dia = diabetes.data
    y_dia = diabetes.target
    X_train_dia, X_test_dia, y_train_dia, y_test_dia = train_test_split(X_dia, y_dia, test_size=0.3, random_state=42)
    return X_train_dia, X_test_dia, y_train_dia, y_test_dia

def iris():
    iris = load_iris()
    X_iris = iris.data[:, :2]
    y_iris = iris.data[:, 2] 
    X_train_iris, X_test_iris, y_train_iris, y_test_iris = train_test_split(X_iris, y_iris, test_size=0.3, random_state=42)
    return X_train_iris, X_test_iris, y_train_iris, y_test_iris

def cars():
    data_cars = pd.read_csv(r"C:\Masters\Y1\Sem1\IAADPI\W5\data\cars.csv")
    cars_df = pd.DataFrame(data_cars)
    X_cars = cars_df[['age', 'gender', 'miles','debt','income']]
    y_cars = cars_df['sales']
    X_train_cars, X_test_cars, y_train_cars, y_test_cars = train_test_split(X_cars, y_cars, test_size=0.3, random_state=42)
    return X_train_cars, X_test_cars, y_train_cars, y_test_cars

def define_regressors():
    regressors = {
        "k-NN": KNeighborsRegressor(n_neighbors=3),
        "SVM": SVR(kernel='rbf', C=10, gamma='scale'),
        "MLP": MLPRegressor(hidden_layer_sizes=(10, 10), max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeRegressor(max_depth=5, random_state=42)
    }
    return regressors

def predict(dataset_name, name, regressor, X_train, y_train, x_test, y_test):
    print(f"\n--- {name} Regressor ---")
    regressor.fit(X_train, y_train)
    y_pred_dia = regressor.predict(x_test)
    evaluate_model(y_test, y_pred_dia, f"{name} {dataset_name}")

if __name__ == "__main__":
    datasets = {"Diabetes":diab(), "Iris":iris(), "Cars":cars()}
    regressors = define_regressors()
    for name, regressor in regressors.items():
        for key, value in datasets.items():
            x_train, x_test, y_train, y_test = value
            predict(dataset_name=key, name=name, regressor=regressor, X_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)


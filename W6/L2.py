import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler


def load_datasets():
    iris = load_iris()
    X_iris, y_iris = iris.data, iris.target

    pima = pd.read_csv("https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv", header=None)
    X_pima, y_pima = pima.iloc[:, :-1].values, pima.iloc[:, -1].values
    return X_iris, y_iris, X_pima, y_pima

def evaluate_adaboost(X_train, X_test, y_train, y_test, base_learner):
    adaboost = AdaBoostClassifier(estimator=base_learner, n_estimators=50, random_state=42)
    adaboost.fit(X_train, y_train)
    y_pred = adaboost.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

def standardize_data(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled

def ada_boost_decision_tree(X_train, X_test, y_train, y_test):
    base_learner = DecisionTreeClassifier(max_depth=1)  # Weak decision tree
    accuracy = evaluate_adaboost(X_train, X_test, y_train, y_test, base_learner)
    return accuracy

def ada_boost_svm(X_train, X_test, y_train, y_test):
    base_learner = SVC(kernel='linear', probability=True)  # Linear SVM
    X_train_scaled, X_test_scaled = standardize_data(X_train, X_test)
    accuracy = evaluate_adaboost(X_train_scaled, X_test_scaled, y_train, y_test, base_learner)
    return accuracy


def evaluate(X, y, test_size, random_state):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    # Dt
    accuracy_dt = ada_boost_decision_tree(X_train, X_test, y_train, y_test)
    # SVM
    accuracy_svm = ada_boost_svm(X_train, X_test, y_train, y_test)
    return accuracy_dt, accuracy_svm

if __name__ == "__main__":
    X_iris, y_iris, X_pima, y_pima = load_datasets()
    accuracy_pima_dt, accuracy_pima_svm = evaluate(X_pima, y_pima, test_size=0.3, random_state=42)
    accuracy_iris_dt, accuracy_iris_svm = evaluate(X_iris, y_iris, test_size=0.3, random_state=42)

    print(f"AdaBoost with Decision Tree on Iris: {accuracy_iris_dt:.4f}")
    print(f"AdaBoost with SVM on Iris: {accuracy_iris_svm:.4f}")
    print(f"AdaBoost with Decision Tree on Pima: {accuracy_pima_dt:.4f}")
    print(f"AdaBoost with SVM on Pima: {accuracy_pima_svm:.4f}")

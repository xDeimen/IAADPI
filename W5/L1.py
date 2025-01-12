import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score


def preprocess_data():
    #Load
    iris = load_iris()
    X_iris, y_iris = iris.data, iris.target
    diabetes_df = pd.read_csv(r'./data/diabetes.csv')
    X_diabetes = diabetes_df.iloc[:, :-1].values
    y_diabetes = diabetes_df.iloc[:, -1].values

    #Scaling
    scaler_iris = StandardScaler().fit(X_iris)
    X_iris = scaler_iris.transform(X_iris)

    scaler_diabetes = StandardScaler().fit(X_diabetes)
    X_diabetes = scaler_diabetes.transform(X_diabetes)
    l = [X_iris, y_iris, X_diabetes, y_diabetes]
    return l


def _display_cv_results(model, X, y, name):
    #prettier print
    scores = cross_val_score(model, X, y, cv=5)
    print(f"{name} - Mean Accuracy: {scores.mean():.3f}, Std Dev: {scores.std():.3f}")


def _generate_random_iris_sample():
    #approximate ranges for each feature
    sepal_length = np.random.uniform(4.3, 7.9)
    sepal_width = np.random.uniform(2.0, 4.4)
    petal_length = np.random.uniform(1.0, 6.9)
    petal_width = np.random.uniform(0.1, 2.5)
    return np.array([[sepal_length, sepal_width, petal_length, petal_width]])

def _generate_random_diabetes_sample():
    #approximate ranges for each feature
    pregnancies = np.random.randint(0, 17)
    glucose = np.random.uniform(0, 200)
    blood_pressure = np.random.uniform(0, 122)
    skin_thickness = np.random.uniform(0, 99)
    insulin = np.random.uniform(0, 846)
    bmi = np.random.uniform(18, 67)
    diabetes_pedigree = np.random.uniform(0.078, 2.42)
    age = np.random.randint(21, 81)
    return np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age]])


def arbirtrary_input_eval(classifiers, scaled_data):

    X_iris, y_iris, X_diabetes, y_diabetes = scaled_data
    s1 =  _generate_random_iris_sample()
    s2= _generate_random_diabetes_sample()

    sample_input_iris = _generate_random_iris_sample()
    sample_input_diabetes= _generate_random_diabetes_sample()

    s3 = np.array([[5.1, 3.5, 1.4, 0.2]])
    s4 = np.array([[6, 148, 72, 35, 0, 33.6, 0.627, 50]])

    print(s1.shape, s2.shape, s3.shape, s4.shape)

    print(sample_input_diabetes)

    print("\n--- Predictions with Arbitrary Inputs ---")

    for name, model in classifiers:
        # Train the model on the full dataset to make predictions
        model.fit(X_iris, y_iris)
        prediction_iris = model.predict(sample_input_iris)
        print(f"{name} prediction on Iris sample input: {prediction_iris}")

        model.fit(X_diabetes, y_diabetes)
        prediction_diabetes = model.predict(sample_input_diabetes)
        print(f"{name} prediction on Diabetes sample input: {prediction_diabetes}")


def cross_val_evaluation(classifiers, scaled_data):
    X_iris, y_iris, X_diabetes, y_diabetes = scaled_data
    for name, model in classifiers:
        print(f"\n{name} on Iris Dataset:")
        _display_cv_results(model, X_iris, y_iris, name)
        
        print(f"\n{name} on Pima Indians Diabetes Dataset:")
        _display_cv_results(model, X_diabetes, y_diabetes, name)




if __name__ == "__main__":
    classifiers = [
        ("K-Nearest Neighbors (k=3)", KNeighborsClassifier(n_neighbors=3)),
        ("Multilayer Perceptron (hidden_layer_sizes=(10,))", MLPClassifier(hidden_layer_sizes=(10,), max_iter=1000)),
        ("Support Vector Machine (linear kernel)", SVC(kernel="linear")),
        ("Decision Tree (max_depth=5)", DecisionTreeClassifier(max_depth=5)),
    ]

    data = preprocess_data()
    cross_val_evaluation(classifiers, data)
    arbirtrary_input_eval(classifiers=classifiers, scaled_data=data)

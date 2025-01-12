import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
from sklearn.metrics import accuracy_score


def load_dataset():
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
    columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
    data = pd.read_csv(url, names=columns)

    X = data.drop('Outcome', axis=1)
    y = data['Outcome']
    return X, y

def normalize_and_split(X, y):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
    return  X_train, X_test, y_train, y_test

def define_classifiers():
    svc = SVC(probability=True, random_state=42)
    knn = KNeighborsClassifier()
    dt = DecisionTreeClassifier(random_state=42)
    mlp = MLPClassifier(random_state=42)

    estimators = [
        ('svc', svc),
        ('knn', knn),
        ('dt', dt),
        ('mlp', mlp)
    ]
    stacking = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())
    classifiers = {"SVC":svc, "KNN":knn, "DT":dt, "MLP":mlp, "STACKING":stacking}
    return classifiers

def predict(classifier, XT, Xt, yT, yt):
    classifier.fit(XT, yT)
    pred = classifier.predict(Xt)
    accuracy = accuracy_score(yt, pred)
    return accuracy

if __name__ == "__main__":
    X, y = load_dataset()
    classfieirs = define_classifiers()
    X_train, X_test, y_train, y_test = normalize_and_split(X, y)
    
    for name, classifier in classfieirs.items():
        accuracy = predict(classifier, X_train, X_test, y_train, y_test)
        print(f"Accuracy of {name}: {accuracy:.4f}")




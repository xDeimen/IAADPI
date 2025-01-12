import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier

def load_data():
    data = pd.read_csv('https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv',header=None)
    data.columns = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"]

    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test

def load_classifiers(rndm_state):
    svc = SVC(probability=True, random_state=rndm_state)
    knn = KNeighborsClassifier()
    dt = DecisionTreeClassifier(random_state=rndm_state)
    classifiers = {"SVC": svc, "k-NN": knn, "Decision Tree": dt}
    return classifiers

def predict_basic(classifiers, split_data):
    X_train, X_test, y_train, y_test = split_data
    for name, clf in classifiers.items():
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"Accuracy of {name}: {acc:.4f}")

def predict_bagging(classifiers, split_data):
    X_train, X_test, y_train, y_test = split_data
    for name, base_clf in classifiers.items():
        bagging_clf = BaggingClassifier(estimator=base_clf, n_estimators=50, random_state=42)
        bagging_clf.fit(X_train, y_train)
        y_pred = bagging_clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"Accuracy of Bagging with {name}: {acc:.4f}")

if __name__ == "__main__":
    split_data = load_data()
    classifiers = load_classifiers(rndm_state=42)
    predict_basic(classifiers, split_data)
    predict_bagging(classifiers, split_data)

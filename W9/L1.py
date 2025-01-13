import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, recall_score, roc_auc_score, roc_curve
)
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

def specificity_score(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    return tn / (tn + fp)

def plot_confusion_matrix(cm, class_names, title="Confusion Matrix"):
    plt.figure(figsize=(6, 6))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.show()

def evaluate_model(model, X_test, y_test, class_names):
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    cm = confusion_matrix(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted")
    rec = recall_score(y_test, y_pred, average="weighted")
    spec = specificity_score(y_test, y_pred) if len(cm.ravel()) == 4 else None
    auc = roc_auc_score(y_test, y_pred_proba, multi_class="ovr") if y_pred_proba is not None else None

    print("Confusion Matrix:\n", cm)
    print(f"Accuracy: {acc:.2f}, Precision: {prec:.2f}, Recall: {rec:.2f}, Specificity: {spec:.2f}, AuC: {auc:.2f}")

    plot_confusion_matrix(cm, class_names)

    if y_pred_proba is not None and auc is not None:
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba, pos_label=model.classes_[1] if len(model.classes_) == 2 else None)
        plt.figure()
        plt.plot(fpr, tpr, label=f"ROC curve (area = {auc:.2f})")
        plt.plot([0, 1], [0, 1], "k--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend(loc="lower right")
        plt.show()


def load_data():
    data = load_iris()
    X_iris = data.data
    y_iris = data.target
    class_names_iris = data.target_names

    diabetes_df = pd.read_csv("https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv", 
                            header=None)
    X_diabetes = diabetes_df.iloc[:, :-1].values
    y_diabetes = diabetes_df.iloc[:, -1].values

    y_iris_binary = LabelEncoder().fit_transform((y_iris == 2).astype(int))

    X_train_iris, X_test_iris, y_train_iris, y_test_iris = train_test_split(
        X_iris, y_iris_binary, test_size=0.2, random_state=42
    )

    X_train_pima, X_test_pima, y_train_pima, y_test_pima = train_test_split(
        X_diabetes, y_diabetes, test_size=0.2, random_state=42
    )

    return X_train_iris, X_test_iris, y_train_iris, y_test_iris, X_train_pima, X_test_pima, y_train_pima, y_test_pima, class_names_iris

def scale_data(X_train_iris, X_test_iris, X_train_pima, X_test_pima):
    scaler_iris = StandardScaler()
    X_train_iris = scaler_iris.fit_transform(X_train_iris)
    X_test_iris = scaler_iris.transform(X_test_iris)

    scaler_pima = StandardScaler()
    X_train_pima = scaler_pima.fit_transform(X_train_pima)
    X_test_pima = scaler_pima.transform(X_test_pima)

def svm(XT, Xt, yT, yt, kernel, probaility, random_state, print_string, class_names):
    print(print_string)
    svm = SVC(kernel=kernel, probability=probaility, random_state=random_state)
    svm.fit(XT, yT)
    evaluate_model(svm, Xt, yt, class_names)

def mlp(XT, Xt, yT, yt, hidden_later_sizes, max_iter, random_state, print_string, class_names):
    print(print_string)
    svm = MLPClassifier(hidden_layer_sizes=hidden_later_sizes, max_iter=max_iter, random_state=random_state)
    svm.fit(XT, yT)
    evaluate_model(svm, Xt, yt, class_names)

if __name__ == "__main__":
    X_train_iris, X_test_iris, y_train_iris, y_test_iris, X_train_pima, X_test_pima, y_train_pima, y_test_pima, class_names_iris = load_data()
    svm(XT=X_train_iris, 
        Xt=X_test_iris, 
        yT=y_train_iris, 
        yt=y_test_iris, 
        kernel="rbf", 
        probaility=True, 
        random_state=42, 
        print_string=f"\n=== SVM on Iris Dataset ===",
        class_names=class_names_iris
    )
    svm(XT=X_train_pima, 
        Xt=X_test_pima, 
        yT=y_train_pima, 
        yt=y_test_pima, 
        kernel="rbf", 
        probaility=True, 
        random_state=42, 
        print_string=f"\n=== SVM on Pima ===",
        class_names=["No Diabetes", "Diabetes"]
    )
    mlp(XT=X_train_pima, 
        Xt=X_test_pima, 
        yT=y_train_pima, 
        yt=y_test_pima, 
        hidden_later_sizes=(10,), 
        max_iter=1000, 
        random_state=42, 
        print_string=f"\n=== MLP on Pima Indians Diabetes Dataset ===",
        class_names=["No Diabetes", "Diabetes"]
    )
    mlp(XT=X_train_iris, 
        Xt=X_test_iris, 
        yT=y_train_iris, 
        yt=y_test_iris, 
        hidden_later_sizes=(10,), 
        max_iter=1000, 
        random_state=42, 
        print_string=f"\n=== MLP on Iris Dataset ===",
        class_names=class_names_iris
    )


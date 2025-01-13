from sklearn.svm import SVC
from sklearn.feature_selection import RFE

from L1 import load_data, svm, mlp

def perform_rfe(X_train, y_train, X_test, n_features):
    svm_rfe = SVC(kernel="linear", random_state=42)
    rfe = RFE(estimator=svm_rfe, n_features_to_select=n_features)
    rfe.fit(X_train, y_train)
    X_train_rfe = rfe.transform(X_train)
    X_test_rfe = rfe.transform(X_test)
    return X_train_rfe, X_test_rfe

if __name__ == "__main__":
    X_train_iris, X_test_iris, y_train_iris, y_test_iris, X_train_pima, X_test_pima, y_train_pima, y_test_pima, class_names_iris = load_data()
    X_train_rfe_iris, X_test_rfe_iris = perform_rfe(X_train_iris, y_train_iris, X_test_iris, 2)
    X_train_rfe_pima, X_test_rfe_pima = perform_rfe(X_train_pima, y_train_pima, X_test_pima, 4)
    svm(XT=X_train_rfe_iris, 
        Xt=X_test_rfe_iris, 
        yT=y_train_iris, 
        yt=y_test_iris,
        kernel="rbf", 
        probaility=True, 
        random_state=42, 
        print_string=f"\n=== SVM RFE on Iris Dataset ===",
        class_names=class_names_iris
    )
    svm(XT=X_train_rfe_pima, 
        Xt=X_test_rfe_pima, 
        yT=y_train_pima, 
        yt=y_test_pima,
        kernel="rbf", 
        probaility=True, 
        random_state=42, 
        print_string=f"\n=== SVM RFE on Pima Dataset ===",
        class_names=["No Diabetes", "Diabetes"]
    )
    mlp(XT=X_train_rfe_pima, 
        Xt=X_test_rfe_pima, 
        yT=y_train_pima, 
        yt=y_test_pima, 
        hidden_later_sizes=(10,), 
        max_iter=1000, 
        random_state=42, 
        print_string=f"\n=== MLP RFE on Pima Indians Diabetes Dataset ===",
        class_names=["No Diabetes", "Diabetes"]
    )
    mlp(XT=X_train_rfe_iris, 
        Xt=X_test_rfe_iris, 
        yT=y_train_iris, 
        yt=y_test_iris, 
        hidden_later_sizes=(10,), 
        max_iter=1000, 
        random_state=42, 
        print_string=f"\n=== MLP RFE on Iris Dataset ===",
        class_names=class_names_iris
    )


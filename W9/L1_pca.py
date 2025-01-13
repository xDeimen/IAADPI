from sklearn.decomposition import PCA
from L1 import load_data, svm, mlp

def perform_pca(X_train, X_test, n_components):
    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    return X_train_pca, X_test_pca

if __name__ == "__main__":
    X_train_iris, X_test_iris, y_train_iris, y_test_iris, X_train_pima, X_test_pima, y_train_pima, y_test_pima, class_names_iris = load_data()
    X_train_pca_iris, X_test_pca_iris = perform_pca(X_train_iris, X_test_iris, 2)
    X_train_pca_pima, X_test_pca_pima = perform_pca(X_train_pima, X_test_pima, 4)
    svm(XT=X_train_pca_iris, 
        Xt=X_test_pca_iris, 
        yT=y_train_iris, 
        yt=y_test_iris,
        kernel="rbf", 
        probaility=True, 
        random_state=42, 
        print_string=f"\n=== SVM PCA on Iris Dataset ===",
        class_names=class_names_iris
    )
    svm(XT=X_train_pca_pima, 
        Xt=X_test_pca_pima, 
        yT=y_train_pima, 
        yt=y_test_pima,
        kernel="rbf", 
        probaility=True, 
        random_state=42, 
        print_string=f"\n=== SVM PCA on Pima Dataset ===",
        class_names=["No Diabetes", "Diabetes"]
    )
    mlp(XT=X_train_pca_pima, 
        Xt=X_test_pca_pima, 
        yT=y_train_pima, 
        yt=y_test_pima, 
        hidden_later_sizes=(10,), 
        max_iter=1000, 
        random_state=42, 
        print_string=f"\n=== MLP PCA in Pima Indians Diabetes Dataset ===",
        class_names=["No Diabetes", "Diabetes"]
    )
    mlp(XT=X_train_pca_iris, 
        Xt=X_test_pca_iris, 
        yT=y_train_iris, 
        yt=y_test_iris, 
        hidden_later_sizes=(10,), 
        max_iter=1000, 
        random_state=42, 
        print_string=f"\n=== MLP PCA on Iris Dataset ===",
        class_names=class_names_iris
    )


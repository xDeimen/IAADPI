from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.datasets import make_classification



def load_data_iris():

    iris = load_iris()
    X = iris.data
    y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    return X_train, X_test, y_train, y_test

def load_random_data_set():
     #This is an artificial dataset to test, because for iris all accuracies were 1
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, 
                            n_redundant=5, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    return X_train, X_test, y_train, y_test


def clfs():
    svc_clf = SVC(probability=True, random_state=42)
    knn_clf = KNeighborsClassifier()
    tree_clf = DecisionTreeClassifier(random_state=42)
    mlp_clf = MLPClassifier(random_state=42)

    classifiers = [('SVC', svc_clf), ('k-NN', knn_clf), ('Decision Tree', tree_clf), ('MLP', mlp_clf)]
    return classifiers

def train_evaulate_clsf(clf, XT, Xt, yT, yt):
    clf.fit(XT, yT)
    y_pred = clf.predict(Xt)
    accuracy = accuracy_score(yt, y_pred)
    return accuracy


def voting(classifiers, XT, Xt, yT, yt, voting_type):
    voting = VotingClassifier(estimators=classifiers, voting='soft')
    voting.fit(XT, yT)
    y_pred = voting.predict(Xt)
    accuracy = accuracy_score(yt, y_pred)
    print(f'{voting_type}_voting accuracy: {accuracy:.4f}')
    return accuracy

def computation_based_on_dataset(classifiers, data, dataset_name):
    print(20*"-",dataset_name,20*"-")
    X_train, X_test, y_train, y_test = data
    for clf_name, clf in classifiers:
        accuracy= train_evaulate_clsf(clf, X_train, X_test, y_train, y_test)
        print(f'{clf_name} accuracy: {accuracy:.4f}')

    voting(classifiers, X_train, X_test, y_train, y_test, "soft")
    voting(classifiers, X_train, X_test, y_train, y_test, "hard")


if __name__ == "__main__":
    computation_based_on_dataset(classifiers=clfs(), data=load_data_iris(), dataset_name="Iris dataset")
    #I tried with a random dataset because for the iris dataset all acuracies were 1 and i thought it was odd
    computation_based_on_dataset(classifiers=clfs(), data=load_random_data_set(), dataset_name="Random dataset")

    

    

    

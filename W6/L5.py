from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

def preprocess_iris_classifier(data_set):
    X = data_set.data
    y = data_set.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
    return X_train, X_test, y_train, y_test

def preprocess_iris_regressor(data_set):
    X = data_set.data[:, [0, 2, 3]]
    y = data_set.data[:, 1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
    return X_train, X_test, y_train, y_test

def random_forest_classifier(data, n_estimators=100):
    X_train, X_test, y_train, y_test = data
    rf_classifier = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    rf_classifier.fit(X_train, y_train)
    y_pred = rf_classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Random Forest Classifier Accuracy: {accuracy:.2f}')
    return accuracy

def random_forest_regressor(data, n_estimators=100):
    X_train, X_test, y_train, y_test = data
    rf_regressor = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
    rf_regressor.fit(X_train, y_train)
    y_pred = rf_regressor.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f'Random Forest Regressor Mean Squared Error: {mse:.2f}') 
    return mse

if __name__ == "__main__":
    iris = load_iris()
    random_forest_classifier(data=preprocess_iris_classifier(data_set=iris),n_estimators=1000)
    random_forest_regressor(data=preprocess_iris_regressor(data_set=iris),n_estimators=1000)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


def multiple_linear_regression(data, independent_vars, dependent_var, arbitrary_input):

    X = data[independent_vars]
    print("X:", X)
    y = data[dependent_var]
    print(f"{X}\n\n\n{y}")

    model = LinearRegression()
    model.fit(X, y)

    X1 = data[independent_vars[0]]
    X2 = data[independent_vars[1]]
    #print("sadfdsafdsfasf",X1)
    x1_range = np.linspace(X1.min(), X1.max(), 10)
    x2_range = np.linspace(X2.min(), X2.max(), 10)
    X1_grid, X2_grid = np.meshgrid(x1_range, x2_range)
    X_grid = np.column_stack((np.ones(X1_grid.ravel().shape), X1_grid.ravel(), X2_grid.ravel()))
    X_grid = [sublist[1:] for sublist in X_grid]
    print(X_grid)
    Y_grid = model.predict(X_grid).reshape(X1_grid.shape)

    fig = plt.figure(figsize=(10, 7))

    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(data[independent_vars[0]], data[independent_vars[1]], data[dependent_var], color='b', label='Data Points')
    ax.plot_surface(X1_grid, X2_grid, Y_grid, color='r', alpha=0.5, label='Regression Plane')
    ax.set_xlabel(independent_vars[0])
    ax.set_ylabel(independent_vars[1])
    ax.set_zlabel(dependent_var)
    ax.set_title('3D Multiple Regression Model')
    plt.legend()
    plt.show()


    arbitrary_input_array = np.array([arbitrary_input])
    predicted_value = model.predict(arbitrary_input_array)
    print(f"Predicted {dependent_var} for {independent_vars} = {arbitrary_input}: {predicted_value[0]}")

    return predicted_value[0]


if __name__ == "__main__":
    data = pd.read_csv('./data/diabetes.csv')
    predicted = multiple_linear_regression(data, ['Glucose', 'BloodPressure'], 'DiabetesPedigreeFunction', [120, 70])
    print(f"Predicted value: {predicted}")
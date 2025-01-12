# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import seaborn as sns

iris = sns.load_dataset('iris')

X = iris[['petal_width']]
y = iris['petal_length']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

linear_regressor = LinearRegression()
linear_regressor.fit(X_train, y_train)


y_train_pred = linear_regressor.predict(X_train)

plt.scatter(X_train, y_train, color='blue', label='Training data')
plt.plot(X_train, y_train_pred, color='red', label='Regression line')
plt.title('Simple Linear Regression: Petal Width vs Petal Length')
plt.xlabel('Petal Width (cm)')
plt.ylabel('Petal Length (cm)')
plt.legend()
plt.grid(True)
plt.show()

arbitrary_value = np.array([[1.5]]) 
predicted_length_linear = linear_regressor.predict(arbitrary_value)
print(f"Predicted Petal Length (Linear Regression) for Petal Width = 1.5 cm: {predicted_length_linear[0]:.2f} cm")




poly = PolynomialFeatures(degree=2)
X_poly_train = poly.fit_transform(X_train)

poly_regressor = LinearRegression()
poly_regressor.fit(X_poly_train, y_train)

y_poly_train_pred = poly_regressor.predict(X_poly_train)

X_train_sorted, y_poly_train_pred_sorted = zip(*sorted(zip(X_train['petal_width'], y_poly_train_pred)))

plt.scatter(X_train, y_train, color='blue', label='Training data')
plt.plot(X_train_sorted, y_poly_train_pred_sorted, color='green', label='Polynomial regression curve')
plt.title('Polynomial Regression (degree=2): Petal Width vs Petal Length')
plt.xlabel('Petal Width (cm)')
plt.ylabel('Petal Length (cm)')
plt.legend()
plt.grid(True)
plt.show()

arbitrary_value_poly = poly.transform(np.array([[1.5]]))  
predicted_length_poly = poly_regressor.predict(arbitrary_value_poly)
print(f"Predicted Petal Length (Polynomial Regression) for Petal Width = 1.5 cm: {predicted_length_poly[0]:.2f} cm")

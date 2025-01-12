import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import statsmodels.api as sm

# Step 1: Generate sample data
np.random.seed(0)
# Generate independent variables
X1 = np.random.rand(100)
X2 = np.random.rand(100)
# Generate a dependent variable with some noise
Y = 3 + 2 * X1 + 1 * X2 + np.random.randn(100) * 0.1

# Combine into a DataFrame
data = pd.DataFrame({'X1': X1, 'X2': X2, 'Y': Y})

# Step 2: Fit the multiple regression model
X = data[['X1', 'X2']]
X = sm.add_constant(X)  # Adds a constant term to the predictor
model = sm.OLS(data['Y'], X).fit()

# Step 3: Create a grid for the regression plane
x1_range = np.linspace(X1.min(), X1.max(), 10)
x2_range = np.linspace(X2.min(), X2.max(), 10)
X1_grid, X2_grid = np.meshgrid(x1_range, x2_range)
X_grid = np.column_stack((np.ones(X1_grid.ravel().shape), X1_grid.ravel(), X2_grid.ravel()))
result = [sublist[1:] for sublist in X_grid]
# Predict Y values on the grid
Y_grid = model.predict(X_grid).reshape(X1_grid.shape)

# Step 4: Plotting
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Plot the original data points
ax.scatter(data['X1'], data['X2'], data['Y'], color='b', label='Data Points')

# Plot the regression plane
ax.plot_surface(X1_grid, X2_grid, Y_grid, color='r', alpha=0.5, label='Regression Plane')

# Labels and title
ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('Y')
ax.set_title('3D Multiple Regression Model')

plt.legend()
plt.show()

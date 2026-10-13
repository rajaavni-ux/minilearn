import numpy as np
from minilearn.linear_model.linear_regression import LinearRegression
from minilearn.metrics.regression import mean_squared_error

# 1. Create dummy data (y = 2x + 1)
X_train = np.array([[1], [2], [3], [4], [5]])
y_train = np.array([3, 5, 7, 9, 11])

# 2. Train the model
model = LinearRegression(learning_rate=0.01, epochs=1000)
print("Training model...")
model.fit(X_train, y_train)

# 3. Generate predictions for the training data
y_pred = model.predict(X_train)

# 4. Calculate the Mean Squared Error
mse = mean_squared_error(y_train, y_pred)

print(f"Actual y:     {y_train}")
print(f"Predicted y:  {np.round(y_pred, 4)}")
print(f"Mean Squared Error: {mse:.6f}")
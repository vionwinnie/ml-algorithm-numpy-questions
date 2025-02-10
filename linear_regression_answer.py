import numpy as np

def linear_regression(X, y):
    # ... (same docstring)

    X_b = np.c_[np.ones((X.shape[0], 1)), X]  # Add bias term
    coefficients = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y # Normal equation

    return coefficients

# Example usage:
X = np.random.rand(100, 1)
y = 2*X + 1 + np.random.randn(100, 1) * 0.1
coefficients = linear_regression(X, y)
print("Coefficients:", coefficients)

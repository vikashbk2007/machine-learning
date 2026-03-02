import numpy as np

# Training Data
X = np.array([800, 1000, 1200, 1500, 1800, 2000], dtype=float)
y = np.array([20, 25, 30, 38, 45, 50], dtype=float)

# Add intercept term (1 for bias)
X_mat = np.column_stack((np.ones(len(X)), X))

# Test value
x_test = 1400
tau = 300

# Compute Gaussian weights
weights = np.exp(-((X - x_test) ** 2) / (2 * tau ** 2))

# Create diagonal weight matrix
W = np.diag(weights)

# Compute theta using pseudo-inverse (more stable than inv)
XTWX = X_mat.T @ W @ X_mat
XTWy = X_mat.T @ W @ y
theta = np.linalg.pinv(XTWX) @ XTWy

# Prediction
x_test_mat = np.array([1, x_test])
prediction = x_test_mat @ theta

# Output
print("Weights:\n", np.round(weights, 3))
print("\nTheta (Intercept, Slope):\n", np.round(theta, 4))
print("\nPredicted Price for 1400 sq.ft: ₹", round(prediction, 2), "Lakhs")

import numpy as np

def preprocess_data(data):
    # ... (same docstring as in the question file)

    num_samples = data.shape[0]
    num_features = data.shape[1] - 1

    X = data[:, :num_features]
    y = data[:, num_features]

    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    X_scaled = (X - mean) / std

    split_index = int(0.8 * num_samples)
    X_train = X_scaled[:split_index]
    X_test = X_scaled[split_index:]
    y_train = y[:split_index]
    y_test = y[split_index:]

    return X_train, X_test, y_train, y_test


# Example usage (for testing):
data = np.random.rand(100, 3)
X_train, X_test, y_train, y_test = preprocess_data(data)

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

import numpy as np

def preprocess_data(data):
    """
    Preprocesses the data by standardizing features and splitting into training/testing sets.

    Args:
        data: A NumPy array where the first columns are features and the last column is the label.

    Returns:
        A tuple containing:
            - X_train: Training features (standardized).
            - X_test: Testing features (standardized).
            - y_train: Training labels.
            - y_test: Testing labels.
    """

    num_samples = data.shape[0]
    num_features = data.shape[1] - 1  # Last column is the label

    # 1. Separate features (X) and labels (y)
    X = data[:, :num_features]
    y = data[:, num_features]

    # 3. Split the data into training and testing sets (80/20 split)
    split_index = int(0.8 * num_samples)
    # TODO: Use NumPy array slicing to create the train/test split
    X_train = X[:split_index, :]
    X_test = X[split_index:, :]
    y_train = y[split_index:]
    y_test = y[:split_index]

    # 2. Standardize the features (X) using NumPy (z-score normalization)
    # TODO: Implement standardization using NumPy's mean and std
    #Z_Score = (x- mean) / std 

    mean = np.mean(X_train, axis = 0)
    std = np.std(X_train, axis = 0)
    print(mean)
    print(std)

    X_train = (X_train - mean) / std
    X_test = (X_test - mean) /std 

    return X_train, X_test, y_train, y_test


# Example usage (for testing your solution):
data = np.random.rand(100, 3)  # 100 samples, 2 features + 1 label
X_train, X_test, y_train, y_test = preprocess_data(data)  # Uncomment after implementing

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

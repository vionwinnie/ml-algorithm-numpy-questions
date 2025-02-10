import numpy as np

def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2)**2))


def knn(X_train, y_train, X_test, k):
    # ... (same docstring)
    num_test_samples = X_test.shape[0]
    predictions = np.zeros(num_test_samples, dtype=int)

    for i in range(num_test_samples):
        distances = np.array([euclidean_distance(X_test[i], x) for x in X_train])
        k_nearest_indices = np.argsort(distances)[:k]
        k_nearest_labels = y_train[k_nearest_indices]
        predictions[i] = np.argmax(np.bincount(k_nearest_labels))

    return predictions

# Example usage:
X_train = np.random.rand(80, 2)
y_train = np.random.randint(0, 2, 80)
X_test = np.random.rand(20, 2)
k = 5

predictions = knn(X_train, y_train, X_test, k)
print("Predictions:", predictions)

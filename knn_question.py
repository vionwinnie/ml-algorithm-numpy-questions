import numpy as np

def euclidean_distance(point1, point2):
    """Calculates the Euclidean distance between two points."""
    # point1 = np.array shape(num_data, dim)
    # point2 = np.array shape(num_centroids,dim)

    # TODO: Implement Euclidean distance using NumPy
    square_diff = (point1[:,np.newaxis,:] - point2[np.newaxis,:,:])**2
    # (num_data, num_centroid)
    return np.sqrt(np.sum(square_diff**2, axis=2))


def knn(X_train, y_train, X_test, k):
    """
    Predicts the labels for X_test using k-Nearest Neighbors.
    """
    num_test_samples = X_test.shape[0]
    predictions = np.zeros(num_test_samples, dtype=int)

    # X_train (num_data, dim)
    # X_test (num_new_data)
    dist_matrix = euclidean_distance(X_train, X_test)
    nearest_neighbor =  np.argsort(dist_matrix,axis=1)[:k,:]
    nearest_neighbor = nearest_neighbor.reshape((num_test_samples,k))

    labels = np.zeros((num_test_samples,))
    for i in range(num_test_samples):
        avg_label = np.mean(y_train[nearest_neighbor[i]])
        labels[i] = avg_label 
    cutoff = 0.5
    result = np.where(labels >= cutoff, 1, 0)

    # This might be better:
    # predictions[i] = np.argmax(np.bincount(k_nearest_labels))
    return result 

# Example usage:
X_train = np.random.rand(80, 2)
y_train = np.random.randint(0, 2, 80)
X_test = np.random.rand(20, 2)
k = 5

predictions = knn(X_train, y_train, X_test, k) # Uncomment after implementing
print("Predictions:", predictions)

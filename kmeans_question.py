import numpy as np

# Exercise 1: Euclidean Distance Calculation

def euclidean_distance(point1, point2):
    """
    Calculates the Euclidean distance between two points.

    Args:
        point1: A NumPy array representing the first point.
        point2: A NumPy array representing the second point.

    Returns:
        The Euclidean distance between the two points.
    """
    print(point1.shape)
    print(point2.shape)
    # TODO: Implement the Euclidean distance calculation using NumPy operations.
    # Hint: Use np.sqrt(), np.sum(), and element-wise subtraction.
    # L2 = sqrt( (x1-y1)**2 + (x2-y2)**2  )
    # point1 after reshape = (3,1,10) point2 = (1,100,10)
    square_diff = (point1[:,np.newaxis,:] - point2[np.newaxis,:,:])**2
    
    return np.sqrt(np.sum(square_diff**2, axis=2))


# Exercise 2: Centroid Update

def update_centroid(points):
    """
    Calculates the new centroid of a set of points.

    Args:
        points: A NumPy array where each row represents a point.

    Returns:
        A NumPy array representing the new centroid.
    """

    return np.mean(points,axis=1)

# Exercise 3: K-means Clustering (Partial Implementation)

def kmeans(data, k, max_iterations=100):
    """
    Performs K-means clustering on the given data.

    Args:
        data: A NumPy array where each row represents a data point.
        k: The number of clusters.
        max_iterations: The maximum number of iterations.

    Returns:
        A tuple containing:
            - A NumPy array of cluster assignments for each data point.
            - A NumPy array of the final centroids.
    """

    num_points = data.shape[0]
    # Initialize centroids randomly (for simplicity, you can improve this)
    centroids = data[np.random.choice(num_points, k, replace=False)] 
    cluster_assignments = np.random.randint(0, 3, size=num_points)

    vectorize = np.vectorize(euclidean_distance)

    for iteration in range(max_iterations):
        # Exercise 3a: Assign points to the nearest centroid

        centroid_dist = euclidean_distance(centroids,data)

        # Evaluating that the 
        avg_distance = np.zeros((k,))
        for m in range(k):
            indices = np.where(cluster_assignments==m)
            avg_distance[m] = np.mean(centroid_dist[m,indices])
        print(f"average intra-cluster distance for iteration {iteration}: {avg_distance}")

        # Shape (k, num_data)

        # E Step (Expectation): Assignment points to clusters 
        new_cluster_assignments = np.argmin(centroid_dist, axis=0)
        # Shape  (num_data,)

        if np.array_equal(new_cluster_assignments, cluster_assignments):  # Check for convergence
            break

        cluster_assignments = new_cluster_assignments

        # Exercise 3b: Update centroids
        for j in range(k):
            indices = np.where(cluster_assignments==j)
            # M step (Maximiation): Updating centroids
            new_centroid = update_centroid(data[indices,:])[0]
            centroids[j,:] = new_centroid

    return cluster_assignments, centroids



# Example usage:
data = np.random.rand(100, 10)  # 100 data points in 2 dimensions
k = 3  # Number of clusters

#print(euclidean_distance(data[1,:].reshape((1,4)), data[2,:].reshape((1,4))))

cluster_assignments, centroids = kmeans(data, k, max_iterations=10) # Uncomment after implementing


# Exercise 4 (Optional):  Implement a more robust centroid initialization method (e.g., k-means++)
# Exercise 5 (Optional): Implement a function to calculate the Sum of Squared Errors (SSE) for evaluating the clustering.

import numpy as np

# Exercise 1: Euclidean Distance Calculation

def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1- point2)**2))


# Exercise 2: Centroid Update

def update_centroid(points):
    return np.mean(points, axis=0)


# Exercise 3: K-means Clustering (Partial Implementation)

def kmeans(data, k, max_iterations=100):
    num_points = data.shape[0]
    centroids = data[np.random.choice(num_points, k, replace=False)]  
    cluster_assignments = np.zeros(num_points, dtype=int)

    for _ in range(max_iterations):
        new_cluster_assignments = np.zeros(num_points, dtype=int)
        # Exercise 3a: Assign points to the nearest centroid
        for i in range(num_points):
            distances = np.array([euclidean_distance(data[i], centroid) for centroid in centroids])
            new_cluster_assignments[i] = np.argmin(distances)

        if np.array_equal(new_cluster_assignments, cluster_assignments):  # Check for convergence
            break

        cluster_assignments = new_cluster_assignments

        # Exercise 3b: Update centroids
        for j in range(k):
            points_in_cluster = data[cluster_assignments == j]
            if len(points_in_cluster) > 0:
                centroids[j] = update_centroid(points_in_cluster)
            else:
                #Option 1: Re-initialize the centroid randomly.
                centroids[j] = data[np.random.choice(num_points, 1, replace=False)]
                #Option 2:  Keep the old centroid (less ideal).
                pass

    return cluster_assignments, centroids



# Example usage:
data = np.random.rand(100, 2)  # 100 data points in 2 dimensions
k = 3  # Number of clusters

cluster_assignments, centroids = kmeans(data, k)

print("Cluster Assignments:", cluster_assignments)
print("Centroids:", centroids)


# Exercise 4 (Optional):  Implement a more robust centroid initialization method (e.g., k-means++)
# Exercise 5 (Optional): Implement a function to calculate the Sum of Squared Errors (SSE) for evaluating the clustering.

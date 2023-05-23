import numpy as np
import random as rand

def initialize_centroids_forgy(data, k):
    # TODO implement random initialization
    return data[rand.sample(range(len(data)), k)]

def initialize_centroids_kmeans_pp(data, k):
    # TODO implement kmeans++ initizalization
    def calc_distance(centroids, c):
        return np.sqrt(sum([sum(abs(centroid - c))**2 for centroid in centroids]))
    
    
    centroids = [rand.choice(data)]
    for i in range(k):
        farthest = 0
        best_centroid = None
        
        for d in data:
            distance = calc_distance(centroids, d)
            if distance > farthest:
                farthest = distance
                best_centroid = d
                
        centroids.append(best_centroid)
    
    return np.array(centroids)



def assign_to_cluster(data, centroid):
    # TODO find the closest cluster for each data point
    assignments = []
    for d in data:
        closest = np.inf
        idx = 0
        for i in range(len(centroid)):
            dist = np.sqrt(np.sum((centroid[i] - d)**2))
            if dist < closest:
                closest = dist
                idx = i
        assignments.append(idx)
    return assignments

def update_centroids(data, assignments):
    # TODO find new centroids based on the assignments
    k = np.max(assignments) + 1
    centroids = np.zeros(k)
    for i in range(k):
        centroids[i] = np.average(data[assignments == i], axis=1)
        
    return centroids
    
def mean_intra_distance(data, assignments, centroids):
    return np.sqrt(np.sum((data - centroids[assignments, :])**2))

def k_means(data, num_centroids, kmeansplusplus= False):
    # centroids initizalization
    if kmeansplusplus:
        centroids = initialize_centroids_kmeans_pp(data, num_centroids)
    else: 
        centroids = initialize_centroids_forgy(data, num_centroids)

    
    assignments  = assign_to_cluster(data, centroids)
    for i in range(100): # max number of iteration = 100
        print(f"Intra distance after {i} iterations: {mean_intra_distance(data, assignments, centroids)}")
        centroids = update_centroids(data, assignments)
        new_assignments = assign_to_cluster(data, centroids)
        if np.all(new_assignments == assignments): # stop if nothing changed
            break
        else:
            assignments = new_assignments

    return new_assignments, centroids, mean_intra_distance(data, new_assignments, centroids)         


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


obs = [1,2,3,4,5,6]
X1 = [1,1,0,5,6,4]
X2 = [4,3,4,1,2,0]
points = [(X1[i], X2[i]) for i in range(len(X1))]

def PartA():
    plt.figure(figsize=(8, 6))
    plt.scatter(X1, X2, color='blue', s=100)
    plt.title('X1 vs X2', fontsize=16)
    plt.xlabel('X1', fontsize=14)
    plt.ylabel('X2', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.savefig('scatter_plot.png', dpi=300, bbox_inches='tight')
    plt.show()
    
def PartB():
    clusters = {'a': [], 'b': []}
    for i in obs:
        r = np.random.rand()
        if r < 0.5:
            clusters['a'].append((i))
        else:
            clusters['b'].append((i))

    print("Cluster assignments:")
    for cluster, pts in clusters.items():
        print(f"Cluster {cluster}: {pts}")
    
def PartC():
    clusters = {'a': [2, 5], 'b': [1, 3, 4, 6]}
    centroids = {}
    for cluster, points in clusters.items():
        meanX = [X1[i-1] for i in points]
        meanY = [X2[i-1] for i in points]
        centroids[cluster] = (np.mean(meanX), np.mean(meanY))
    
    for cluster, centroid in centroids.items():
        print(f"Centroid of cluster {cluster}: ({centroid[0]}, {centroid[1]})")
        
    return centroids

def distance_between_points(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def PartD():
    centroids = PartC()
    new_cluster = {'a': [], 'b': []}

    for i, p in enumerate(points):
        dists = dict()
        for cluster, cen in centroids.items():
            dist = distance_between_points(p, cen)
            dists[cluster] = dist

        closest_cluster = min(dists, key=dists.get)
        print(f"Point {p} is closest to cluster {closest_cluster} with distance {dists[closest_cluster]:.2f}")
        
        new_cluster[closest_cluster].append((i+1, p))

    print("New Clusters (obs):")
    for cluster, pts in new_cluster.items():
        print(f"Cluster {cluster}: {[x[0] for x in pts]}")

    print("New Clusters (points):")
    for cluster, pts in new_cluster.items():
        print(f"Cluster {cluster}: {[x[1] for x in pts]}")

def calculate_centroids(clusters: dict[str, list[tuple[int, tuple[int, int]]]]) -> dict[str, tuple[float, float]]:
    centroids = {}
    for cluster, cPoints in clusters.items():
        meanX = [X1[p[0]-1] for p in cPoints]
        meanY = [X2[p[0]-1] for p in cPoints]
        centroids[cluster] = (np.mean(meanX), np.mean(meanY))
    return centroids

def reassign_clusters(clusters: dict[str, list[tuple[int, tuple[int ,int]]]]) -> dict[str, list[tuple[int, tuple[int ,int]]]]:
    centroids = calculate_centroids(clusters)

    new_clusters: dict[str, list[tuple[int, tuple[int, int]]]] = {'a': [], 'b': []}

    for i, p in enumerate(points):
        dists = dict()
        for cluster, cen in centroids.items():
            dist = distance_between_points(p, cen)
            dists[cluster] = dist

        closest_cluster = min(dists, key=dists.get) 
        new_clusters[closest_cluster].append((i+1, p))

    return new_clusters

def PartE():
    clusters: dict[str, list[tuple[int, tuple[int, int]]]] = dict(a=[], b=[])
    
    cluster_a = [2,5]
    cluster_b = [1,3,4,6]
    for i in cluster_a:
        clusters['a'].append((i, points[i-1]))
    for i in cluster_b:
        clusters['b'].append((i, points[i-1]))

    print("Initial Clusters:")
    for cluster, pts in clusters.items():
        print(f"Cluster {cluster}: {pts}")
    
    for i in range(5):
        print(f"Iteration {i+1}:")
        clusters = reassign_clusters(clusters)
        
        print("New Clusters (obs):")
        for cluster, pts in clusters.items():
            print(f"Cluster {cluster}: {[x[0] for x in pts]}")
        
        print("New Clusters (points):")
        for cluster, pts in clusters.items():
            print(f"Cluster {cluster}: {[x[1] for x in pts]}")
        
        print()
        
    return clusters

def PartF():
    clusters = PartE()
    
    plt.figure(figsize=(8, 6))
    for cluster, pts in clusters.items():
        plt.scatter(*zip(*[p[1] for p in pts]), label=f"Cluster {cluster}", s=100)
    
    plt.title('X1 vs X2', fontsize=16)
    plt.xlabel('X1', fontsize=14)
    plt.ylabel('X2', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.savefig('scatter_plot_F.png', dpi=300, bbox_inches='tight')
    plt.show()
    
PartF()
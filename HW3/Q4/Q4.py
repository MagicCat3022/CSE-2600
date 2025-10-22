import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from ISLP import load_data
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import \
     (KMeans,
      AgglomerativeClustering)
from scipy.cluster.hierarchy import \
     (dendrogram,
      cut_tree)
from ISLP.cluster import compute_linkage
from sklearn.cluster import \
     (KMeans,
      AgglomerativeClustering)
    
    


data = pd.read_csv('CEV2021.csv')

def PartA():
    features = data.drop('State', axis=1)
    state_labels = data['State'].array

    HClust = AgglomerativeClustering
    hc_comp = HClust(distance_threshold=0,
                    n_clusters=None,
                    linkage='complete')
    hc_comp.fit(features)
    
    linkage_comp = compute_linkage(hc_comp)
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    dendrogram(linkage_comp,
            ax=ax,
            leaf_rotation=90,
            labels=state_labels,
            color_threshold=0.15, # this is the important part 
            above_threshold_color='black')
    
    plt.show()

def PartB():
    features = data.drop('State', axis=1)

    HClust = AgglomerativeClustering
    hc_comp = HClust(distance_threshold=0,
                    n_clusters=None,
                    linkage='complete')
    hc_comp.fit(features)
    
    linkage_comp = compute_linkage(hc_comp)
    
    data['clusters'] = cut_tree(linkage_comp, n_clusters=3)
    print(data[['State', 'clusters']].sort_values(by='clusters'))
    
def PartC():
    features = data.drop('State', axis=1)

    HClust = AgglomerativeClustering
    hc_comp = HClust(distance_threshold=0,
                    n_clusters=None,
                    linkage='complete')
    hc_comp.fit(features)
    
    linkage_comp = compute_linkage(hc_comp)
    
    data['clusters'] = cut_tree(linkage_comp, n_clusters=4)
    print(data[['State', 'clusters']].sort_values(by='clusters'))
    
def PartD():
    features = data.drop('State', axis=1)
    
    kmeans = KMeans(n_clusters=3).fit(features)
    data['clusters'] = kmeans.labels_
    print(data[['State', 'clusters']].sort_values(by='clusters'))

def PartE():
    features = data.drop('State', axis=1)
    
    for f in features.columns:
        features[f] = (features[f] - features[f].mean()) / features[f].std()
    
    state_labels = data['State'].array

    HClust = AgglomerativeClustering
    hc_comp = HClust(distance_threshold=0,
                    n_clusters=None,
                    linkage='complete')
    hc_comp.fit(features)
    
    linkage_comp = compute_linkage(hc_comp)
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    dendrogram(linkage_comp,
            ax=ax,
            leaf_rotation=90,
            labels=state_labels,
            color_threshold=3.4, # this is the important part 
            above_threshold_color='black')
    
    plt.show()
    
def PartF():
    features = data.drop('State', axis=1)
    
    for f in features.columns:
        features[f] = (features[f] - features[f].mean()) / features[f].std()
    
    state_labels = data['State'].array

    HClust = AgglomerativeClustering
    hc_comp = HClust(distance_threshold=0,
                    n_clusters=None,
                    linkage='complete')
    hc_comp.fit(features)
    
    linkage_comp = compute_linkage(hc_comp)
    data['clusters'] = cut_tree(linkage_comp, n_clusters=3)
    print(data[['State', 'clusters']].sort_values(by='clusters'))
    
def PartG():
    features = data.drop('State', axis=1)
    
    for f in features.columns:
        features[f] = (features[f] - features[f].mean()) / features[f].std()
    
    kmeans = KMeans(n_clusters=3).fit(features)
    data['clusters'] = kmeans.labels_
    print(data[['State', 'clusters']].sort_values(by='clusters'))
    
def PartH():
    features = data.drop('State', axis=1)
    
    for f in features.columns:
        features[f] = (features[f] - features[f].mean()) / features[f].std()
    
    kmeans = KMeans(n_clusters=3).fit(features)
    data['clusters'] = kmeans.labels_
    
    plt.scatter(x=features['Voting_Local'],
                y=features['Organizational_Membership'],
                c=data['clusters'])
    
    plt.xlabel('Voting_Local (Standardized)')
    plt.ylabel('Organizational_Membership (Standardized)')
    plt.title('K-Means Clustering of States')
    plt.show()
    
PartH()
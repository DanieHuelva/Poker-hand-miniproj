from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
import sys


filename = 'Boston.csv'
matrix_data = pd.read_csv("Boston.csv")
scalar = StandardScaler()
STD = scalar.fit_transform(matrix_data)



def getKmeans(data, k, eps):
    scalar = StandardScaler()
    STD = scalar.fit_transform(data)
    pca = PCA(n_components=13)
    RPCA = pca.fit_transform(STD)
    #Make random centroids
    ## thhis is a looop!!1
    #assign points to closest centroid 
    #reassign centroid
    #check if distance is less than eps
    return 0




matrix_data = pd.read_csv("Customers.csv")
getKmeans(matrix_data, 4, sys.float_info.epsilon)
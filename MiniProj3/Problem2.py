from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA


filename = 'Boston.csv'
matrix_data = pd.read_csv("Boston.csv")
scalar = StandardScaler()
STD = scalar.fit_transform(matrix_data)


#Problem 2.1
def getKmeans(data, k, eps):
    return 0

#Problem 2.2
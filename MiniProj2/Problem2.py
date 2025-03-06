import random
import networkx as nx
from itertools import combinations
import numpy as np


def load_edges_from_file(file_path):
    lis = []
    with open(file_path, 'r') as file:
        for line in file:
            # Split the line by whitespace and convert the nodes to integers (or keep as strings)
            node1, node2 = line.strip().split()
            # Add the edge between node1 and node2
            lis.append((int(node1), int(node2)))
    return lis


def makeIntoGraph(edges):
    G = nx.Graph()  # Create an empty graph
    for edge in edges:
        G.add_edge(*edge)
    return G


#THIS IS A TINY GRAPH TO TEST FUNCTIONS AND CHECK ACCURACY
listEd = [ (4,0),
        (0,1),
        (1,2),
        (2,3),
        (6,5),
        (5,1),
        (1,7),
        (7,8),
        (2,9),
        (9,10),
        (0,7),
        (2,7)]




##PROBLEM 1  (verified)
def getNodes(G):
    nodes = set()
    for edge in G:
        for node in edge:
            if node not in nodes:
                nodes.add(node)
    return len(nodes)


# G = load_edges_from_file("/Users/daniehuelva/Desktop/comp/Data Mining/Poker-hand-miniproj-1/MiniProj2/FBdata.txt")
newG = makeIntoGraph(listEd)


##PROBLEM 1
# print("Number of nodes: ", getNodes(listEd))







##PROBLEM 2 (veridied)
def getDeg(G, i):
    deg = {}
    for edge in G:
        for node in edge:
            if node not in deg:
                deg[node] = 1           #use node as key and value as how many degree a node has
            else:
                deg[node] +=1
    return deg[i]


##PROBLEM 2
# print("Getting the degree of 5: ", getDeg(listEd, 5))
# print(nx.degree(newG)[5])








##PROBLEM 3 (verified)
def getAdj(listed):
    nodesNum = getNodes(listed)
    adjMatrix = []
    rows, cols = nodesNum, nodesNum     #we want a 2d array 
    for _ in range(rows):
        row = []
        for _ in range(cols):
            row.append(0)
        adjMatrix.append(row)
    for edge in listed:
        u,v = edge
        adjMatrix[u][v] = 1             #make that position in the adjacency matrix since it forms an edge
        adjMatrix[v][u] = 1
    return adjMatrix



##PROBLEM 3
# adj = getAdj(listEd)
# for row in adj:
#     print(row)
# print()
# print(nx.adjacency_matrix(newG, nodelist=sorted(newG.nodes())).toarray())






##PROBLEM 4 (verified)
def degDesti(G):
    deg = {}
    for edge in G:                      #made a dictionary of nodes and its degress
        for node in edge:
            if node not in deg:
                deg[node] = 1
            else:
                deg[node] +=1
    nodeDeg = {}                    #dictionary for degree as key and value as how many nodes are with that degree
    maxVal = max(deg.values())      #only up until the maximum degree
    for i in range(maxVal+1):
        nodeDeg[i] = 0
    for node in deg.keys():
        val = deg[node]
        nodeDeg[val] += 1
    return list(nodeDeg.values())


##PROBLEM 4
# lis = degDesti(listEd)
# print(lis)
# degree_distribution = nx.degree_histogram(newG)
# print(degree_distribution) 







##PROBLEM 5 (verified)
def probabVertex(degreeList, degree):
    totalsum = 0
    total = sum(degreeList)
    for i in range(degree, len(degreeList)):
        totalsum+= degreeList[i]
    return (totalsum/total)



##PROBLEM 5
# lis = degDesti(listEd)
# print(lis)
# print("Probabiltiy of getting", 4, "or higher: ", probabVertex(lis, 4))






##PROBLEM 6  (verified)
def eccen(lis, vertex):
    G = makeIntoGraph(lis)
    nodeNum = getNodes(lis)
    listPath = []
    for i in range(nodeNum):
        length = nx.shortest_path_length(G, vertex, i)          #finding if which lentgth is the longest and uses that as the longest path of a vertec
        listPath.append(length)
    return max(listPath)




##PROBLEM 6
# print("eccentricity of vertex 10:" ,eccen(listEd, 10))
# print(nx.eccentricity(newG, v=[10]))






##PROBLEM 7  (verified)
def diameter(lis):
    nodeNUm = getNodes(lis)
    findingDiam = []
    for i in range(nodeNUm):
        diam = eccen(lis, i)
        findingDiam.append(diam)
    return max(findingDiam)         #maximum of all the eccentricity of all nodes


##PROBLEM 7
# print("Diameter of graph:", diameter(listEd))
# print(nx.diameter(newG))





##PROBLEM 8    (verified)
def radius(lis):
    nodeNUm = getNodes(lis)
    findingDiam = []
    for i in range(nodeNUm):
        diam = eccen(lis, i)
        findingDiam.append(diam)
    return min(findingDiam)         #minimum of all the eccentricity of all nodes





##PROBLEM 8
# print("Radius of a graph", radius(listEd))
# print(nx.radius(newG))



##PROBLEM 9   (verified)
def clusCoeff(lis, vertex):
    G = makeIntoGraph(lis)  
    neighbors = list(G.neighbors(vertex)) 
    m = 0
    num = len(neighbors)
    if num < 2:
        return 0 
    # Iterate through pairs of neighbors
    for i in range(num-1):
        for j in range(i+1, num):  
            if G.has_edge(neighbors[i], neighbors[j]):
                m += 1
    
    coeff = (2 * m) / (num * (num - 1))  # Clustering coefficient formula
    return coeff



##PROBLEM 9
# print("Clustering coeff of graph: ",clusCoeff(listEd, 7))
# print(nx.clustering(newG, 7))



##PROBELM 10
def between(edges, vertex):
    G = nx.Graph()
    G.add_edges_from(edges)
    centrality = 0
    for s, t in combinations(G.nodes, 2):  # Avoids duplicate pairs automatically
        if vertex in (s, t):  
            continue
        all_shortest_paths = list(nx.all_shortest_paths(G, source=s, target=t))
        total_paths = len(all_shortest_paths)  # Total shortest paths from s to t
        paths_through_vertex = sum(1 for path in all_shortest_paths if vertex in path)
        if total_paths > 0:  # Avoid division by zero
            centrality += paths_through_vertex / total_paths  # Normalize contribution
    # Normalize by the number of possible pairs (excluding vertex itself)
    total_pairs = (len(G.nodes) - 1) * (len(G.nodes) - 2) / 2
    return centrality / total_pairs if total_pairs > 0 else 0



##PROBLEM 10
# print(between(listEd, 7))
# betweenness_scores = nx.betweenness_centrality(newG) 
# print(betweenness_scores[7])




##PROBLEM 11 (verified)
def cloness(listed, vertex):
    G = nx.Graph()
    G.add_edges_from(listed)
    total = 0 
    numNodes = getNodes(listed)
    for i in range(numNodes):
        if i != vertex:
            length = nx.shortest_path_length(G,vertex,i)
            total+=length
    return ((numNodes-1)/total)



##PROBLEM 11
# print(cloness(listEd, 4))
# print(nx.closeness_centrality(newG)[4])





def power_iteration_eigenvector_centrality(adj_matrix, max_iter):
    n = len(adj_matrix[0]) # Number of nodes
    x = np.ones(n) / np.sqrt(n)  # Initial guess (normalized)
    for _ in range(max_iter):
        x_new = np.dot(adj_matrix, x)  # Multiply adjacency matrix with vector
        norm = np.linalg.norm(x_new, 2)  # Compute L2 norm (Euclidean norm)
        if norm == 0:
            return x_new  # Avoid division by zero
        x_new /= norm  # Normalize with L2 norm
        # Check for convergence
        x = x_new  # Update vector
    return x



##PROBLEM 12
adj = getAdj(listEd)
centrality = power_iteration_eigenvector_centrality(adj, 100)
print("Eigenvector Centrality:", centrality)
# print(nx.eigenvector_centrality(newG).values())
import random
import networkx as nx

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
# listEd = [ (1,2), 
#         (0, 4),
#         (3,1),
#         (2,3), (2,4), (2,5)
# ]

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


##PROBLEM 1
def getNodes(G):
    nodes = set()
    for edge in G:
        for node in edge:
            if node not in nodes:
                nodes.add(node)
    return len(nodes)


##PROBLEM 2
def getDeg(G, i):
    deg = {}
    for edge in G:
        for node in edge:
            if node not in deg:
                deg[node] = 1
            else:
                deg[node] +=1
    return deg[i]


##PROBLEM 3
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


##PROBLEM 4
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


##PROBLEM 5
def probabVertex(degreeList, degree):
    totalsum = 0
    total = sum(degreeList)
    for i in range(degree-1, len(degreeList)):
        totalsum+= degreeList[i]
    return (totalsum/total)


##PROBLEM 6
def eccen(lis, vertex):
    G = makeIntoGraph(lis)
    nodeNum = getNodes(lis)
    listPath = []
    for i in range(nodeNum):
        length = nx.shortest_path_length(G, vertex, i)
        listPath.append(length)
    return max(listPath)


##PROBLEM 7
def diameter(lis):
    nodeNUm = getNodes(lis)
    findingDiam = []
    for i in range(nodeNUm):
        diam = eccen(lis, i)
        findingDiam.append(diam)
    return max(findingDiam)


##PROBLEM 8
def radius(lis):
    nodeNUm = getNodes(lis)
    findingDiam = []
    for i in range(nodeNUm):
        diam = eccen(lis, i)
        findingDiam.append(diam)
    return min(findingDiam)


def clusCoeff(lis, vertex):
    G = makeIntoGraph(lis)  # Assuming makeIntoGraph is defined elsewhere
    neighbors = list(G.neighbors(vertex))  # Convert to list for easier indexing
    m = 0
    num = len(neighbors)
    if num < 2:
        return 0  # A vertex with fewer than 2 neighbors has no valid clustering coefficient
    # Iterate through pairs of neighbors
    for i in range(num-1):
        for j in range(i+1, num):  # Start from i+1 to avoid duplicate pairs
            if G.has_edge(neighbors[i], neighbors[j]):
                m += 1
    
    coeff = (2 * m) / (num * (num - 1))  # Clustering coefficient formula
    return coeff


G = load_edges_from_file("/Users/daniehuelva/Desktop/comp/Data Mining/Poker-hand-miniproj-1/MiniProj2/FBdata.txt")

# print(getNodes(G.edges()))
# print(getDeg(G.edges(), 2))
# print(getAdj(listEd))
# lis = degDesti(listEd)
# print(lis)
# print("Probabiltiy of getting", 3, "or higher: ", probabVertex(lis, 3)*100, "%")
# print(diameter(listEd))
# print(radius(listEd))
print(clusCoeff(listEd, 1))
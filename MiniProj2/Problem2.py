import random
import networkx as nx

def load_edges_from_file(file_path):
    G = nx.Graph()  # Create an empty graph
    with open(file_path, 'r') as file:
        for line in file:
            # Split the line by whitespace and convert the nodes to integers (or keep as strings)
            node1, node2 = line.strip().split()
            # Add the edge between node1 and node2
            G.add_edge(int(node1), int(node2))

    return G


#THIS IS A TINY GRAPH TO TEST FUNCTIONS AND CHECK ACCURACY
listEd = [ (1,2), 
        (0, 4),
        (3,1),
        (2,3), (2,4), (2,5)
]

def getNodes(G):
    nodes = set()
    for edge in G:
        for node in edge:
            if node not in nodes:
                nodes.add(node)
    return len(nodes)



def getDeg(G, i):
    deg = {}
    for edge in G:
        for node in edge:
            if node not in deg:
                deg[node] = 1
            else:
                deg[node] +=1
    return deg[i]


def getAdj(listed):
    nodesNum = getNodes(listed)
    adjMatrix = []
    rows, cols = nodesNum, nodesNum
    for _ in range(rows):
        row = []
        for _ in range(cols):
            row.append(0)
        adjMatrix.append(row)
    for edge in listed:
        u,v = edge
        adjMatrix[u][v] = 1
        adjMatrix[v][u] = 1
    return adjMatrix



G = load_edges_from_file("/Users/daniehuelva/Desktop/comp/Data Mining/Poker-hand-miniproj-1/MiniProj2/FBdata.txt")
# print(getNodes(G.edges()))
# print(getDeg(G.edges(), 2))
print(getAdj(listEd))


# Make sure you run this code cell everytime you open the notebook!
import networkx as nx            # Import NetworkX library as nx for its network functions
from networkx.algorithms.community import k_clique_communities
import numpy as np               # Import NumPy library as np for its mathematical functions
import matplotlib.pyplot as plt  # Import Matplotlib library as plt for its plotting functions
import matplotlib.patches as mpatches
import math
# import plotly.express as px
import pandas as pd



############# functions for plotting and network analyses #############

# A function defined for simple network plotting
def plot_graph(G):
    pos = nx.drawing.layout.circular_layout(G)
    plt.figure(figsize=(7,6))                                                             # Specify the figure size
    # in the next line, nx.draw(...) had an error, changing it to nx.draw_networkx(...) fixes it
    nx.draw_networkx(G, pos=pos, with_labels=True, font_weight='bold', node_size=500, \
    node_color='red', font_size=16, font_color='white',width=3, edge_color='green') # The figure properties
    plt.show()                                                                            # Show the network
    # uncomment the following line of code to save the simple network plot to a [.png]. change the path to wherever you want to output it.
    # if you want to output it in the same directory, write "./network.png" instead of what I have
    plt.savefig("../data/networkAnalysisData/network.png", format="PNG")


# A function that provides some network summary information.
def get_graph_info(graph):
    print("Number of nodes:", graph.number_of_nodes())
    print("Number of edges:", graph.number_of_edges())
    if type(graph) == nx.classes.digraph.DiGraph:
        print("Number of weakly connected components:",
              len(list(nx.weakly_connected_components(graph))))
    else:
        print("Number of connected components:", len(list(nx.connected_components(graph))))
    print("Node degree:", dict(graph.degree()))


# A function to print the highest centrality, and the corresponding nodes
def find_highest_centrality_nodes(centralities, centrality_metric):
  itemMaxValue = max(centralities.items(), key=lambda x: x[1])
  print('The maximum', centrality_metric, 'centrality score in the network : ', itemMaxValue[1])
  listOfKeys = list()
  # Iterate over all the items in dictionary to find keys with max value
  for key, value in centralities.items():
      if value == itemMaxValue[1]:
          listOfKeys.append(key)
  print('Nodes with the maximum', centrality_metric, 'centrality score: ', listOfKeys)




#loading data:
'''
    the code snippet below loads the data from 2 csv files, nodes and edges. I put both files in the same directory. 
    the edges file should have 2 SNP columns (at least for me) 
    each row has 2 snps and represents an edge between them: i.e.
    ##
    col1 col2
    snp1 snp2
    snp2 snp4
    ##
    above representation means snp1, snp2, and snp4 are our nodes, 
    and snp1 has an edge with snp2, snp2 has edge with snp4 etc.  

    nodes file thus lists all these nodes:
    col1 col2
    0    snp1
    1    snp2
    2    snp4
    where col1 is an ID column, not necessary
'''

#### loading a data from edges and nodes csv files. file is described above. ####
#### follow the line-by-line instructions if present ####

#change the paths to match your edges and nodes file paths, and change the column names to match your file's column names
edges = pd.read_csv('../data/networkAnalysisData/randomNetworkAnalysisData_edges.csv', skiprows = 1, sep=',', names=['SNP1', 'SNP2'])
nodes = pd.read_csv('../data/networkAnalysisData/randomNetworkAnalysisData_nodes.csv', skiprows = 1, sep =',', names = ['SNPid'])




#### plotting a graph using the data above, and using the plotting functions defined at the top ####
#### follow the line-by-line instructions if present ####
G = nx.from_pandas_edgelist(edges, 'SNP1', 'SNP2')
get_graph_info(G)

# print(edges.head()) # .head() shows the first few rows of your dataframe
# print(nodes.head())

plot_graph(G) # download and use gephi software to make a s*xier graph (maybe watch a youtube tutorial if unfamiliar)








#### Network Analyses Unweighted ####
# some of these may use the functions from the top of this file

# A. CENTRALITIES

# 1. DEGREE CENTRALITY, i.e. the number of links/edges to other nodes that a certain node has is its degree centrality

centralities=nx.degree_centrality(G)                               # Find the degree centrality of all nodes
# print("The degree centrality for all nodes: ", centralities)       # uncomment to see: All degree centrality of the nodes in the network G.
find_highest_centrality_nodes(centralities,'degree')               # The nodes with highest degree centrality

# following finds the degree centrality of node x. replace x with an index that is not out of bounds, then uncomment:
# print("The degree centrality for node x is ", centralities[x - 1])  # -1 cuz 
# can use this print command to show any of the other centralities for a specific node too, just copy/paste it under any other centrality metric below


# 2. HARMONIC CENTRALITY is a centrality measure related to closeness centrality, but better since it works for disconnected networks. 

# closeness centrality is how close a node is to all other nodes. it is measured using (reciprocal of ) the sum of shortest paths that all other nodes have to this node.
# thus if the shortest paths' sum is small, its reciprocal (and closeness centrality) is large

centralities=nx.harmonic_centrality(G)                               # Find the harmonic centrality of all nodes
# print("The harmonic centrality for all nodes: ", centralities)       # All harmonic centrality of the nodes in the network G.
find_highest_centrality_nodes(centralities,'harmonic')               # The nodes with highest harmonic centrality


#3. BETWEENNESS CENTRALITY: depends on how many shortest paths a node lies on (or shortest paths an edge exists in), higher the better

#a. node betweenness
centralities=nx.betweenness_centrality(G)                            # Find betweenness centrality of all nodes
# print("The betweenness centrality for all nodes: ", centralities)    # All betweenness centrality of the nodes in the network G.
find_highest_centrality_nodes(centralities,'betweenness')            # The nodes with highest betweenness centrality
#b. edge betweenness
centralities=nx.edge_betweenness_centrality(G)                              # Edge betweenness centrality for all nodes
# print("The betweenness centrality for all edges: ", centralities)
find_highest_centrality_nodes(centralities, 'betweenness')                  # The highest edge betweenness centrality and the corresponding edges


#4. PAGE RANK. PageRank computes a ranking of the nodes in the graph G based on the structure of the incoming links. 
# It was originally designed as an algorithm to rank web pages.
centralities=nx.pagerank(G)                                          # Find the pagerank centrality of all nodes
# print("The pagerank centrality for all nodes: ", centralities)       # All pagerank centrality of the nodes in the network G.
find_highest_centrality_nodes(centralities,'pagerank')               # The nodes with highest pagerank centrality



# B. COMMUNITIES

#6. CFINDER: finds k-clique overlapping communities using clique percolation. google clique percolation and cfinder for details, but:
# basically what it does is it divides the nodes up in communities, and allows one node to be a member of multiple communities
# which previous algorithms, like Girvan Newman algorithm, could not do

# Find 3-clique communities using k-clique communities
cliques = nx.algorithms.community.k_clique_communities(G, 3)

# Print the 3-clique communities
for i, clique in enumerate(cliques, start=1):
    print(f"3-Clique Community {i}: {clique}")















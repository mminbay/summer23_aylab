import networkx as nx
from networkx.algorithms.community import k_clique_communities
import pandas as pd
import matplotlib.pyplot as plt

############# Functions for Plotting and Network Analyses #############

# Function to plot the graph
def plot_graph(G):
    """
    Plot the graph using NetworkX library.

    Args:
        G (nx.Graph): The input graph.
    """
    pos = nx.drawing.layout.circular_layout(G)
    plt.figure(figsize=(7, 6))    # Specify the figure size
    # in the next line, nx.draw(...) had an error, changing it to nx.draw_networkx(...) fixes it
    nx.draw_networkx(
        G,
        pos=pos,
        with_labels=True,
        font_weight='bold',
        node_size=500,
        node_color='red',
        font_size=16,
        font_color='white',
        width=3,
        edge_color='green'
    ) # can change the figure properties from among the above as suitable
    
    plt.show()
    # Uncomment the following line to save the network plot as a PNG file
    # if you want to output it in the same directory, write "./network.png" instead of what I have
    # plt.savefig("../data/networkAnalysisData/network.png", format="PNG")


# Function to get summary information about the graph
def get_graph_info(graph):
    """
    Print summary information about the graph.

    Args:
        graph (nx.Graph): The input graph.
    """
    print("Number of nodes:", graph.number_of_nodes())
    print("Number of edges:", graph.number_of_edges())
    if type(graph) == nx.classes.digraph.DiGraph:
        print("Number of weakly connected components:", len(list(nx.weakly_connected_components(graph))))
    else:
        print("Number of connected components:", len(list(nx.connected_components(graph))))
    print("Node degree:", dict(graph.degree()))
    print('\n')

    
# Function to find the nodes with the highest centrality score
def find_highest_centrality_nodes(centralities, centrality_metric, is_node_metric):
    """
    Find the nodes with the highest centrality score.

    Args:
        centralities (dict): Dictionary of centrality scores for nodes.
        centrality_metric (str): The centrality metric used.

    """
    itemMaxValue = max(centralities.items(), key=lambda x: x[1])
    
    listOfKeys = list()
    for key, value in centralities.items():
        if value == itemMaxValue[1]:
            listOfKeys.append(key)
    if is_node_metric:
        print('Node(s) with the maximum', centrality_metric, 'centrality score:', listOfKeys)
        print('The maximum', centrality_metric, 'centrality score in the network: ', itemMaxValue[1])
    else:
        print('Edge(s) with the maximum', centrality_metric, 'centrality score:', listOfKeys)
        print('The maximum edge', centrality_metric, 'centrality score in the network: ', itemMaxValue[1])
    print('\n')


# function to print centrality info
def print_centrality_info(centralities, centrality_metric, highest_only=False, is_node_metric = True):
    """
    Print centrality information for nodes.

    Args:
        centralities (dict): Dictionary of centrality scores for nodes.
        centrality_metric (str): The centrality metric used.
        highest_only (bool): Whether to print only the highest centrality values(node/edge) or all centrality values.
    """
    if not highest_only:
        print("The", centrality_metric, "centrality for all nodes:", centralities)
    find_highest_centrality_nodes(centralities, centrality_metric, is_node_metric)

    
# Function to load data from CSV files
def load_data(edges_path, nodes_path):
    """
    Load data from edges and nodes CSV files.

    Args:
        edges_path (str): Path to the edges CSV file.
        nodes_path (str): Path to the nodes CSV file.

    Returns:
        pd.DataFrame: The loaded edges data.
        pd.DataFrame: The loaded nodes data.
    """
    edges = pd.read_csv(edges_path, skiprows=1, sep=',', names=['SNP1', 'SNP2'])
    nodes = pd.read_csv(nodes_path, skiprows=1, sep=',', names=['SNPid'])
    return edges, nodes

    
############# Main Code #############

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

# Define the paths to the edges and nodes CSV files
edges_path = '../data/networkAnalysisData/randomNetworkAnalysisData_edges.csv'
nodes_path = '../data/networkAnalysisData/randomNetworkAnalysisData_nodes.csv'

# Load data from edges and nodes CSV files
edges, nodes = load_data(edges_path, nodes_path)

# Create the graph
G = nx.from_pandas_edgelist(edges, 'SNP1', 'SNP2')

# Print graph information
get_graph_info(G)

# Plot the graph
plot_graph(G) # download and use gephi software to make a better (s*xier) looking graph 

############# Network Analyses Unweighted #############

# A. CENTRALITIES


# 1. Degree Centrality
''' i.e. the number of links/edges to other nodes that a certain node has is its degree centrality '''
centralities = nx.degree_centrality(G)                               # degree centrality for all nodes             
print_centrality_info(centralities, 'degree', highest_only=False)      # prints node(s) with the highest degree centrality values, as well as other info if highest_only=false
# each centrality measure follows similar 2 lines of code so they don't have comments explaining them
# following finds the degree centrality of node x. replace x with an index that is not out of bounds, then uncomment:
# print("The degree centrality for node 1 is ", centralities[1], '\n')  


# 2. Harmonic Centrality
''' this is a centrality measure related to closeness centrality, but better since it works for disconnected networks too '''
centralities = nx.harmonic_centrality(G)
print_centrality_info(centralities, 'harmonic', highest_only=True)

# 3. Betweenness Centrality
''' depends on how many shortest paths a node lies on (or shortest paths an edge exists in), higher the better '''
# a. node-betweenness
centralities = nx.betweenness_centrality(G)
print_centrality_info(centralities, 'betweenness', highest_only=True)
# b. edge-betweenness
centralities=nx.edge_betweenness_centrality(G)                             
print_centrality_info(centralities, 'betweenness', highest_only=True, is_node_metric=False)                 

# 4. PageRank Centrality
''' PageRank computes a ranking of the nodes in the graph G based on the structure of the incoming links. 
    It was originally designed as an algorithm to rank web pages. '''
centralities = nx.pagerank(G)
print_centrality_info(centralities, 'PageRank', highest_only=True)  


# B. COMMUNITIES
'''
 k-clique communities, or CFINDER: finds k-clique overlapping communities using clique percolation. google clique percolation and cfinder for details, but:
 in summary what it does is it divides the nodes up in communities, and allows one node to be a member of multiple communities
 which previous algorithms, like Girvan Newman algorithm, could not do
'''

# 5. k-Clique Communities

cliques = k_clique_communities(G, 3) # finds 3-clique communities, change the 3 to find bigger-clique communities
for i, clique in enumerate(cliques, start=1):
    print(f"3-Clique Community {i}: {clique}")
print('\n')

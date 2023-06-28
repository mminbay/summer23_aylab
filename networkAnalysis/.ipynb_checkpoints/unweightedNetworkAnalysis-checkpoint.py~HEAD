'''
Graph indexing is from 1, not 0.
Will add a step-by-step guide on how to use file below:
1.
'''



import networkx as nx
from networkx.algorithms.community import k_clique_communities
import networkit as nk           # Import networkit library for its katz similarity function, you need to install networkit  
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
import random

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
    plt.savefig("../data/networkAnalysisData/network.png", format="PNG")

    


    
# Function to plot communities with unique colors (has subfunctions below it)
def visualize_communities(graph, communities):
    pos = nx.spring_layout(graph)
    node_colors = assign_node_colors(graph, communities)
    color_legend = create_color_legend(node_colors)

    plt.figure(figsize=(8, 8))
    plt.subplots_adjust(top=5.5, bottom=0.1, right=2)  # modify the values here to move the plot up or down if legend overlaps it
    nx.draw(graph, pos, with_labels=True, node_color=node_colors, cmap=plt.cm.rainbow)
    legend = create_legend(color_legend)
    legend.get_frame().set_linewidth(1.5)
    plt.show()
     # Uncomment the following line to save the network plot as a PNG file
    # plt.savefig("../data/networkAnalysisData/k-cliqueCommunitiesNetwork.png", format="PNG")

def assign_node_colors(graph, communities):
    excluded_colors = ['pink', 'orange', 'purple']
    available_colors = get_available_colors(excluded_colors)
    random.shuffle(available_colors)
    node_colors = []

    for node in graph.nodes:
        num_communities = sum(1 for community in communities if node in community)
        if num_communities == 2:
            node_colors.append('orange')
        elif num_communities > 2:
            node_colors.append('pink')
        else:
            for i, community in enumerate(communities):
                if node in community:
                    node_colors.append(available_colors[i])
                    break
            else:
                node_colors.append('purple')

    return node_colors

def get_available_colors(excluded_colors):
    css4_colors = list(colors.CSS4_COLORS.values())
    available_colors = [color for color in css4_colors if color not in excluded_colors]
    return available_colors

def create_color_legend(node_colors):
    color_legend = {}
    unique_colors = list(set(node_colors))
    unique_colors.sort(key=lambda c: node_colors.count(c), reverse=True)

    for color in unique_colors:
        num_nodes = node_colors.count(color)
        if color == 'orange':
            color_legend[color] = f'2 communities ({num_nodes} nodes)'
        elif color == 'pink':
            color_legend[color] = f'More than 2 communities ({num_nodes} nodes)'
        elif color == 'purple':
            color_legend[color] = f'0 communities ({num_nodes} nodes)'
        else:
            color_name = get_color_name(color)
            color_legend[color] = f'Community {color_name} ({num_nodes} nodes)'

    return color_legend

def get_color_name(color):
    for name, rgb in colors.CSS4_COLORS.items():
        if color == rgb:
            return name

def create_legend(color_legend):
    patches = []
    labels = []
    for color, community in color_legend.items():
        patches.append(plt.plot([], [], marker='o', markersize=10, color=color, ls="")[0])
        labels.append(community)
    legend = plt.legend(patches, labels, loc='upper right', frameon=True, framealpha=1, borderaxespad=0.5)
    return legend


    

# Function to show  degree distribution histogram/curve
def degree_distribution(G, curve=False):
    if not curve: # make distribution histogram
        fig = plt.figure("Degree of a random network", figsize=(20, 12))
        axgrid = fig.add_gridspec(5, 4)
        degree_sequence = sorted((d for n, d in G.degree()), reverse=True)
        ax2 = fig.add_subplot(axgrid[3:, 2:])
        ax2.bar(*np.unique(degree_sequence, return_counts=True))
        ax2.set_title("Network Degree Distribution Histogram")
        ax2.set_xlabel("Degree")
        ax2.set_ylabel("# of Nodes")
        fig.tight_layout()
        plt.show()
        plt.savefig("../data/networkAnalysisData/degree_distribution_histogram.png", format="PNG", bbox_inches='tight')
    else: # make distribution curve
        plt.figure()
        # Calculate the degree of each node
        degrees = dict(G.degree())
    
        # Calculate the degree distribution
        degree_values = list(degrees.values())
        degree_distribution = [degree_values.count(d) for d in set(degree_values)]
    
        # Plot the degree distribution curve
        plt.scatter(list(set(degree_values)), degree_distribution)
        # plt.xscale('log') # uncomment this and the line below to make it log scale
        # plt.yscale('log')
        plt.xlabel('Degree')
        plt.ylabel('Number of Nodes')
        plt.title('Network Degree Distribution Curve')
        plt.show()
        plt.savefig("../data/networkAnalysisData/degree_distribution_curve.png", format="PNG", bbox_inches='tight')

    

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

###### Similarity analyses functions #######

# Define the Cosine Similarity function
def cosine_similarity(list1, list2):
    intersection = len(list(set(list1).intersection(list2)))
    degree1=len(list1)
    degree2=len(list2)
    return float(intersection) / np.sqrt(degree1*degree2)
    
# Define the Jaccard Similarity function
def jaccard(list1, list2):
    intersection = len(list(set(list1).intersection(list2)))
    union = (len(list1) + len(list2)) - intersection
    return float(intersection) / union
    
# We are using networkit library to find Katz similarity score
def calculate_katz_similarity(graph, beta, alpha):
    nk_graph = nk.nxadapter.nx2nk(graph, weightAttr=None)          # Convert the NetworkX graph to a networkit graph
    katz_index = nk.linkprediction.KatzIndex(nk_graph, beta, alpha)   # Calculate Katz similarity score for the network
    return nk_graph, katz_index


# Function to get the similar pairs from the similarity analyses
def get_similar_pairs(similarity_matrix, threshold):
    similar_pairs = []
    num_nodes = similarity_matrix.shape[0]

    for i in range(1, num_nodes): # graph indexing is from 1 and not 0
        for j in range(1, num_nodes):
            if similarity_matrix[i][j] >= threshold:
                similar_pairs.append((i, j))

    return similar_pairs

    
# prints the nodes with similarity scores higher than the threshold given. have to pass metric and threshold 
def print_high_similarity(G, metric_name, similarity_matrix, similarity_threshold, KatzIndex=None):   
    print(metric_name, similarity_matrix)
    for i in range(1,nx.number_of_nodes(G)): # graph indexing is from 1 and not 0
        for j in range(1,nx.number_of_nodes(G)):
            if metric_name == 'cosine':
                A_Cosine[i][j] = cosine_similarity(list(G.neighbors(i)), list(G.neighbors(j)))
            elif metric_name == 'jaccard':
                A_Jaccard[i][j] = jaccard(list(G.neighbors(i)), list(G.neighbors(j)))
            elif metric_name == 'katz':
                A_Katz[i][j] = KatzIndex.run(i, j)
    print(metric_name, similarity_matrix)
    similar_pairs = get_similar_pairs(similarity_matrix, similarity_threshold)
    if len(similar_pairs) == 0:
        print(f"No pairs of nodes have a {metric_name} similarity score higher than {similarity_threshold}.")
    else:
        print(f"Pairs of nodes with a {metric_name} similarity score higher than {similarity_threshold}:")
        # uncomment the lines below to print all the nodes with scores higher than threshold
        # for pair in similar_pairs:
        #     print(f"Nodes {pair[0]} and {pair[1]}, score is: {similarity_matrix[pair[0]][pair[1]]} \n")
    
############# Main Code ###############

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
G = nx.from_pandas_edgelist(edges, 'SNP1', 'SNP2') ###THIS GRAPH's indexing starts at 1, if your's starts at 0, change it to 1, or you will have to change a lot of code for it to work properly
# e.g. get_similar_pairs() and print_high_similarity() functions assume indexing starts at 1
# also, it skips the top row (the header row) so you can have one row of headers for the columns. 
# do confirm if the number of nodes in the graph match the number of nodes in your file. no. of nodes in the graph are given by the get_graph_info(G)

# Print graph information
get_graph_info(G)

# Plot the graph
plot_graph(G) # download and use gephi software to make a better (s*xier) looking graph 

############# Network Analyses Unweighted #############

# A. CENTRALITIES ##############


# 1. Degree Centrality
''' i.e. the number of links/edges to other nodes that a certain node has is its degree centrality '''
centralities = nx.degree_centrality(G)                               # degree centrality for all nodes             
print_centrality_info(centralities, 'degree', highest_only=False)      # prints node(s) with the highest degree centrality values, as well as other info if highest_only=false
# each centrality measure follows similar 2 lines of code so they don't have comments explaining them
# following finds the degree centrality of node x. replace x with an index that is not out of bounds, then uncomment:
# x = 1
# print(f"The degree centrality for node {x} is ", centralities[x], '\n')  


# 2. Harmonic Centrality
''' this is a centrality measure related to closeness centrality, but better since it works for disconnected networks too '''
centralities = nx.harmonic_centrality(G)
print_centrality_info(centralities, 'harmonic', highest_only=True)

# 3. Node Betweenness Centrality
''' depends on how many shortest paths a node lies on (or shortest paths an edge exists in), higher the better '''

centralities = nx.betweenness_centrality(G)
print_centrality_info(centralities, 'betweenness', highest_only=True)

# 4. Edge Betweenness Centrality
centralities=nx.edge_betweenness_centrality(G)                             
print_centrality_info(centralities, 'betweenness', highest_only=True, is_node_metric=False)                 

# 5. PageRank Centrality
''' PageRank computes a ranking of the nodes in the graph G based on the structure of the incoming links. 
    It was originally designed as an algorithm to rank web pages. '''
centralities = nx.pagerank(G)
print_centrality_info(centralities, 'PageRank', highest_only=True)  


# B. COMMUNITIES ##############
'''
 k-clique communities, or CFINDER: finds k-clique overlapping communities using clique percolation. google clique percolation and cfinder for details, but:
 in summary what it does is it divides the nodes up in communities, and allows one node to be a member of multiple communities
 which previous algorithms, like Girvan Newman algorithm, could not do
'''

# 6. k-Clique Communities
k = 3
cliques = list(k_clique_communities(G, k)) # finds 3-clique communities, change the 3 to find bigger-clique communities
for i, clique in enumerate(cliques, start=1):
    print(f"{k}-Clique Community {i}: {clique}")
print('\n')
# visualize_communities(G, cliques) # Plots the k-clique communities, the plt.savefig() line may be commented, check the visualize_communities() function above


# C. NETWORK SIMILARITY ##############


similarity_threshold = 0.5   # temporary, can change as needed
#7. Cosine Similarity 
''' How many common neigbors do 2 nodes have '''
A_Cosine = np.zeros((nx.number_of_nodes(G), nx.number_of_nodes(G)))
print_high_similarity(G, 'cosine', A_Cosine, similarity_threshold)
''' the print statement below prints the similarity scores for all pairs, similar print statements are commented out for jaccard and katz below
 it abridges the result in the terminal though '''
print("Cosine similarity matrix for each pair of nodes: ", A_Cosine, '\n')

#8. Jaccard Similarity 
''' similar to cosine, different formula '''
A_Jaccard = np.zeros((nx.number_of_nodes(G), nx.number_of_nodes(G)))
print_high_similarity(G, 'jaccard', A_Jaccard, similarity_threshold)
print("Jaccard similarity matrix for each pair of nodes: ", A_Jaccard, '\n')

#9 Katz similarity
''' 2 nodes have high katz similarity if they have neighbors which are similar '''
beta = 5       # Beta value
alpha = 0.005  # Alpha value
GNK, KatzIndex = calculate_katz_similarity(G, beta, alpha) #GNK is the networkit graph, katzindex is used to calculate katz similarity scores

A_Katz = np.zeros((nx.number_of_nodes(G), nx.number_of_nodes(G)))
print_high_similarity(G, 'katz', A_Katz, similarity_threshold, KatzIndex)
print("Katz similarity matrix for each pair of nodes: ", A_Katz, '\n')   

#  example print statement for checking katz score b/w 2 nodes: x and y, uncomment to run: 
x , y = 1 , 4
print(f"The similarity of nodes {x} and {y}, measured by Katz index is: ", KatzIndex.run(x, y), '\n') # Print the Katz similarity score between nodes 1 and 4.
   

# D. DEGREE DISTRIBUTION


# DEGREE DISTRIBUTION, MIN DEGREE, MAX DEGREE, AVERAGE DEGREE, HISTOGRAM/CURVE
degree_sequence = sorted((d for n, d in G.degree()), reverse=True)     # Sort the degrees into decreasing order
print("The list of degrees of network G sorted in decreasing order: ", degree_sequence)
print("The maximum degree of network G is: ", max(degree_sequence) )   # Maximum degree
print("The minimum degree of network G is: ", min(degree_sequence) )   # Minimum degree
print("The average degree of network G is: ",  2*G.number_of_edges()/G.number_of_nodes())                   # Average degree
# following plots the degree distribution histogram, do curve=Frue to plot curve
degree_distribution(G, curve=True)
# if curve is gaussian/normal dist, network is random, if the curve is like power-law dist (left side high), network is scale free




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
def plot_graph(G, path=None):
    """
    Plot the graph using the NetworkX library.

    Args:
        G (nx.Graph): The input graph.

    Note:
        - The graph will be plotted using NetworkX's `draw_networkx` function.
        - The layout algorithm used is circular_layout, which positions the nodes in a circular arrangement.
        - Graph will be saved only if path is provided
        - Various visual properties of the graph can be customized by modifying the arguments of `draw_networkx` inside the function (in the unweigtedNetworkAnalysis.py file).

    Returns:
        None
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
    if path is not None:
        plt.savefig(path, format="PNG")

    
# Function to get summary information about the graph
def print_graph_info(graph, bool_print=True):
    """
    Print summary information about the graph.

    Args:
        graph (nx.Graph): The input graph.
        bool_print (bool): Flag to indicate whether to print the information. Defaults to True.

    Returns:
        Tuple: A tuple containing the following information:
            - num_nodes (int): Number of nodes in the graph.
            - num_edges (int): Number of edges in the graph.
            - node_degree_dict (dict): Dictionary mapping nodes to their degree.
            - num_connected_components (int): Number of connected components in the graph (for undirected graphs).
            - num_weakly_connected_components (int): Number of weakly connected components in the graph (for directed graphs).

    """
    bool_type_graph = False
    num_nodes = graph.number_of_nodes()
    num_edges = graph.number_of_edges()
    node_degree_dict = dict(graph.degree())
    
    if type(graph) == nx.classes.digraph.DiGraph:
        num_weakly_connected_components = len(list(nx.weakly_connected_components(graph)))
        bool_type_graph = True
        if bool_print:
            print("Number of weakly connected components:", num_weakly_connected_components)
    else:
        num_connected_components = len(list(nx.connected_components(graph)))
        if bool_print:
            print("Number of connected components:", num_connected_components)
            
    if bool_print:
        print("Number of nodes:", num_nodes)
        print("Number of edges:", num_edges)
        print("Node degree:", node_degree_dict)
        print('\n')
    if bool_type_graph:
        return num_nodes, num_edges, node_degree_dict, num_weakly_connected_components
    else:
        return num_nodes, num_edges, node_degree_dict, num_connected_components

    
# Function to plot communities with unique colors (has subfunctions below it)
def visualize_communities(graph, communities, path=None):
    """
    Visualize the communities in a graph using NetworkX library.

    Args:
        graph (nx.Graph): The input graph.
        communities (list): List of community assignments for each node.
        path (str, optional): The file path to save the plot as a PNG image. Default is None.

    Note:
        - Graph is saved only if `path` is provided.
        - The graph will be plotted using NetworkX's `draw` function if path parameter is provided.
        - The layout algorithm used is spring_layout, which attempts to position the nodes using a force-directed algorithm.
        - Each community will be assigned a distinct color for visualization.
        - The color legend will be created to indicate the mapping between colors and communities.

    Returns:
        None
    """
    pos = nx.spring_layout(graph)
    node_colors = assign_node_colors(graph, communities)
    color_legend = create_color_legend(node_colors)

    plt.figure(figsize=(8, 8))
    plt.subplots_adjust(top=5.5, bottom=0.1, right=2)  # modify the values here to move the plot up or down if legend overlaps it
    nx.draw(graph, pos, with_labels=True, node_color=node_colors, cmap=plt.cm.rainbow)
    legend = create_legend(color_legend)
    legend.get_frame().set_linewidth(1.5)
    plt.show()
    if path is not None:
        plt.savefig(path, format="PNG")

def assign_node_colors(graph, communities):
    excluded_colors = ['pink', 'orange', 'purple', 'darkmagenta']
    available_colors = get_available_colors(excluded_colors)
    random.shuffle(available_colors)
    node_colors = []

    for node in graph.nodes:
        num_communities = sum(1 for community in communities if node in community)
        if num_communities == 2:
            node_colors.append(0.0)  # Assign a numerical value (0.0) for 'orange'
        elif num_communities > 2:
            node_colors.append(1.0)  # Assign a numerical value (1.0) for 'pink'
        else:
            for i, community in enumerate(communities):
                if node in community:
                    node_colors.append(i + 1)  # Assign numerical values (starting from 1) for other colors
                    break
            else:
                node_colors.append(len(communities) + 1)  # Assign a numerical value greater than the number of communities for 'purple'

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
        if color == 0.0:
            color = 'orange'
        elif color == 1.0:
            color = 'pink'
        elif color == len(color_legend) + 1:
            color = 'purple'
        else:
            color = get_color_name(color)
        patches.append(plt.plot([], [], marker='o', markersize=10, color=color, ls="")[0])
        labels.append(community)
    legend = plt.legend(patches, labels, loc='upper right', frameon=True, framealpha=1, borderaxespad=0.5)
    return legend


    

# Function to show  degree distribution histogram/curve
def degree_distribution(G, path=None,curve=False, log_scale=False):
    """
    Visualize the degree distribution of a network.

    This function plots the degree distribution of a network as either a histogram or a curve,
    depending on the value of the `curve` parameter. The degree distribution represents the
    number of nodes in the network with a given degree.

    Args:
        G (networkx.Graph): The input graph.
        path (str, optional): The file path to save the plot. If provided, the plot will be saved as an image.
            Defaults to None.
        curve (bool, optional): Determines whether to plot the degree distribution as a curve (scatter plot) or
            a histogram. If False, a histogram will be plotted. If True, a curve will be plotted. Defaults to False.
        log_scale (bool, optional): Determines whether to use a logarithmic scale for both x and y axes in the curve plot.
            Only applicable when `curve` is True. Defaults to False.

    Returns:
        None

    Note:
        - If `curve` is False, a histogram of the degree distribution will be plotted.
        - If `curve` is True, a curve (scatter plot) of the degree distribution will be plotted.
        - If `path` is provided, the plot will be saved as an image in the specified file path.
    """
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
        plt.show
        if path is not None:
            plt.savefig(path, format="PNG", bbox_inches='tight')
    else: # make distribution curve
        plt.figure()
        # Calculate the degree of each node
        degrees = dict(G.degree())
    
        # Calculate the degree distribution
        degree_values = list(degrees.values())
        degree_distribution = [degree_values.count(d) for d in set(degree_values)]
    
        # Plot the degree distribution curve
        plt.scatter(list(set(degree_values)), degree_distribution)
        if log_scale:
            plt.xscale('log') # uncomment this and the line below to make it log scale
            plt.yscale('log')
        plt.xlabel('Degree')
        plt.ylabel('Number of Nodes')
        plt.title('Network Degree Distribution Curve')
        plt.show()
        if path is not None:
            plt.savefig(path, format="PNG", bbox_inches='tight')

    


    
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
def load_data(edges_path, source_column_name, target_column_name, nodes_path=None, node_column_name=None):
    """
    Load data from edges and nodes CSV files.

    Args:
        edges_path (str): Path to the edges CSV file.
        nodes_path (str): Path to the nodes CSV file.

    Returns:
        pd.DataFrame: The loaded edges data.
        pd.DataFrame: The loaded nodes data.
    """
    edges = pd.read_csv(edges_path, skiprows=1, sep=',', names=[source_column_name, target_column_name])
    if (nodes_path is not None) and (node_column_name is not None):                                     #if both node file and column name provided
        nodes = pd.read_csv(nodes_path, skiprows=1, sep=',', names=[node_column_name])
        return edges, nodes
    else:
        return edges

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
def print_high_similarity(G, metric_name, similarity_matrix, similarity_threshold, KatzIndex=None, print_all=True):   
    for i in range(1,nx.number_of_nodes(G)): # graph indexing is from 1 and not 0
        for j in range(1,nx.number_of_nodes(G)):
            if metric_name == 'cosine':
                similarity_matrix[i][j] = cosine_similarity(list(G.neighbors(i)), list(G.neighbors(j)))
            elif metric_name == 'jaccard':
                similarity_matrix[i][j] = jaccard(list(G.neighbors(i)), list(G.neighbors(j)))
            elif metric_name == 'katz':
                similarity_matrix[i][j] = KatzIndex.run(i, j)
    similar_pairs = get_similar_pairs(similarity_matrix, similarity_threshold)
    num_sim_pairs = sum(1 for pair in similar_pairs if pair[0] != pair[1])
    if num_sim_pairs == 0:
        print(f"No pairs of nodes have a {metric_name} similarity score higher than {similarity_threshold}.")
    else:
        print(f"{num_sim_pairs} Pairs of nodes, excluding same nodes, have a {metric_name} similarity score higher than {similarity_threshold}:")
        # uncomment the lines below to print all the nodes with scores higher than threshold
        if print_all:
            for pair in similar_pairs:
                if (pair[0] == pair[1]):
                    pass
                else:
                    print(f"Nodes {pair[0]} and {pair[1]}, score is: {similarity_matrix[pair[0]][pair[1]]} \n")


# GRAPH MAKING AND NETWORK ANALYSIS FUNCTIONS

def make_graph_from_edges(edges_file_path, source_column_name, target_column_name, nodes_file_path=None, node_column_name=None):
    """
    Create a graph from an edges file.

    This function loads data from an edges file and optionally a nodes file, and creates a graph using the
    NetworkX library based on the provided edge and node information.

    Args:
        edges_file_path (str): The file path of the edges file.
        source_column_name (str): The name of the column in the edges file representing the source nodes.
        target_column_name (str): The name of the column in the edges file representing the target nodes.
        nodes_file_path (str, optional): The file path of the nodes file. Defaults to None.
        node_column_name (str, optional): The name of the column in the nodes file representing the nodes.
            Required if nodes_file_path is provided. Defaults to None.

    Returns:
        networkx.Graph: The created graph based on the provided edge and node information.

    Raises:
        ValueError: If nodes_file_path is provided but node_column_name is not.

    Note:
        - If only the edges file is provided, the function will create a graph using only the edge information.
        - If both the edges file and nodes file are provided, the function will create a graph using both the edge
          and node information.
    """
    if nodes_file_path is not None:
        edges, nodes = load_data(edges_file_path, source_column_name, target_column_name, nodes_file_path, node_column_name)
    else:
        edges = load_data(edges_file_path, source_column_name, target_column_name)
    return nx.from_pandas_edgelist(edges, source_column_name, target_column_name)


def print_centrality(G, centrality_metric, highest_only=True):
    '''
        """
    Print centrality information based on the specified centrality metric.
    
    Args:
        G (networkx.Graph): The input graph.
        centrality_metric (str): The centrality metric to compute.
            Valid options: 'degree', 'harmonic', 'node_betweenness', 'edge_betweenness', 'pagerank'
        highest_only (bool, optional): Whether to print only the nodes with the highest centrality values.
            Defaults to True.
    """
    '''
    centralities = None
    
    if centrality_metric == 'degree':
        # 1. Degree Centrality
        centralities = nx.degree_centrality(G)
        print_centrality_info(centralities, 'degree', highest_only=highest_only)
        
    elif centrality_metric == 'harmonic':
        # 2. Harmonic Centrality
        centralities = nx.harmonic_centrality(G)
        print_centrality_info(centralities, 'harmonic', highest_only=highest_only)
        
    elif centrality_metric == 'node_betweenness':
        # 3. Node Betweenness Centrality
        centralities = nx.betweenness_centrality(G)
        print_centrality_info(centralities, 'betweenness', highest_only=highest_only)
        
    elif centrality_metric == 'edge_betweenness':
        # 4. Edge Betweenness Centrality
        centralities = nx.edge_betweenness_centrality(G)
        print_centrality_info(centralities, 'betweenness', highest_only=highest_only, is_node_metric=False)
        
    elif centrality_metric == 'pagerank':
        # 5. PageRank Centrality
        centralities = nx.pagerank(G)
        print_centrality_info(centralities, 'PageRank', highest_only=highest_only)
        
    else:
        print(f"Invalid centrality metric: {centrality_metric}")
    
    return centralities

    
def CFINDER_communities(G, k=3, path=None):
    """
    Find k-clique communities using CFinder algorithm and visualize them.

    Args:
        G (nx.Graph): The input graph.
        k (int, optional): The size of the cliques to search for. Default is 3.
        path (str, optional): The path to save the visualization plot as a PNG file. If not provided, the plot will not be saved.

    Note:
        - CFinder algorithm identifies k-clique communities in the graph.
        - The `k_clique_communities` function from NetworkX is used to find the cliques.
        - The communities will be visualized using the `visualize_communities` function.
        - If `path` is provided, the visualization plot will be saved as a PNG file at the specified location.

    Returns:
        None
    """
    cliques = list(k_clique_communities(G, k))
    for i, clique in enumerate(cliques, start=1):
        print(f"{k}-Clique Community {i}: {clique}")
    print('\n')
    if path is not None:
        visualize_communities(G, cliques, path=path)
    else:
        visualize_communities(G, cliques)

# k = 3
#     cliques = list(k_clique_communities(G, k)) # finds 3-clique communities, change the 3 to find bigger-clique communities
#     for i, clique in enumerate(cliques, start=1):
#         print(f"{k}-Clique Community {i}: {clique}")
#     print('\n')
#     path = '../data/networkAnalysisData/k-cliqueCommunities.png'
#     visualize_communities(G, cliques, path=path) # Plots the k-clique communities, the plt.savefig() line may be commented out, check the visualize_communities() function above


def print_similarity(G, similarity_metric, similarity_threshold=0.5, print_all_pairs=False):
    """
    Print the similarity scores between nodes in the graph based on the specified similarity metric.

    Args:
        G (nx.Graph): The input graph.
        similarity_metric (str): The similarity metric to use. Possible values are 'cosine', 'jaccard', and 'katz'.
        similarity_threshold (float, optional): The threshold value for considering a similarity score as high. Default is 0.5.
        print_all_pairs (bool, optional): Whether to print the similarity scores for all pairs of nodes. Default is False.

    Note:
        - This function calculates and prints the similarity scores between nodes in the graph based on the specified similarity metric.
        - The similarity metric options are:
            - 'cosine': Cosine similarity between node attribute vectors.
            - 'jaccard': Jaccard similarity between node attribute sets.
            - 'katz': Katz similarity measure using the Katz Index algorithm.
        - The function uses the `print_high_similarity` function to print the high similarity scores.
        - If `print_all_pairs` is True, it will print the similarity matrix for each pair of nodes.

    Returns:
        None
    """
    if similarity_metric == 'cosine':
        A_Cosine = np.zeros((nx.number_of_nodes(G), nx.number_of_nodes(G)))
        print_high_similarity(G, 'cosine', A_Cosine, similarity_threshold, print_all=print_all_pairs)
        ''' the print statement below prints the similarity scores for all pairs, similar print statements are commented out for jaccard and katz below
            it abridges the result in the terminal though '''
        if print_all_pairs:
            print("Cosine similarity matrix for each pair of nodes: ", A_Cosine, '\n')
    elif similarity_metric == 'jaccard':
        A_Jaccard = np.zeros((nx.number_of_nodes(G), nx.number_of_nodes(G)))
        print_high_similarity(G, 'jaccard', A_Jaccard, similarity_threshold, print_all=print_all_pairs)
        if print_all_pairs:
            print("Jaccard similarity matrix for each pair of nodes: ", A_Jaccard, '\n')
    elif similarity_metric == 'katz':
        beta = 5       # Beta value
        alpha = 0.005  # Alpha value
        GNK, KatzIndex = calculate_katz_similarity(G, beta, alpha) #GNK is the networkit graph, katzindex is used to calculate katz similarity scores
        
        A_Katz = np.zeros((nx.number_of_nodes(G), nx.number_of_nodes(G)))
        print_high_similarity(G, 'katz', A_Katz, similarity_threshold, KatzIndex, print_all=print_all_pairs)
        if print_all_pairs:
            print("Katz similarity matrix for each pair of nodes: ", A_Katz, '\n') 
    else:
        raise ValueError("Invalid similarity metric. Please choose 'cosine', 'jaccard', or 'katz'.")


def plot_degree_distribution(G, curve=True, print_degree_info=False, path=None, log_scale=False):

    """
    Plot the degree distribution of the graph.

    Args:
        G (nx.Graph): The input graph.
        curve (bool, optional): Whether to plot the degree distribution as a curve. If False, a histogram is plotted. Default is True.
        print_degree_info (bool, optional): Whether to print degree-related information. Default is False.
        path (str, optional): The file path to save the plot. If not provided, the plot will be displayed interactively (plot.show()).
        log_scale (bool, optional): Whether to use a logarithmic scale for the x-axis (degree). Only applicable when `curve` is True. Default is False.

    Note:
        - This function calculates and plots the degree distribution of the graph.
        - The degree distribution represents the frequency of each degree value in the graph.
        - If `curve` is True, the degree distribution is plotted as a curve.
        - If `curve` is False, the degree distribution is plotted as a histogram.
        - If `print_degree_info` is True, it will print: the list of degrees in decreasing order, maximum degree, minimum degree, and average degree of the graph.
        - If `path` is provided, the plot will be saved as a PNG file.

    Returns:
        None
    """    
    degree_sequence = sorted((d for n, d in G.degree()), reverse=True)     # Sort the degrees into decreasing order
    if print_degree_info: 
        print("The list of degrees of network G sorted in decreasing order: ", degree_sequence)
        print("The maximum degree of network G is: ", max(degree_sequence) )   # Maximum degree
        print("The minimum degree of network G is: ", min(degree_sequence) )   # Minimum degree
        print("The average degree of network G is: ",  2*G.number_of_edges()/G.number_of_nodes())        # Average degree

    # following plots the degree_distribution. makes a histogram if curve=false. if curve true, and log_scale true, puts curve on log_scale
    degree_distribution(G, curve=curve, path=path, log_scale=log_scale)  
    
##############################################################################
##############################################################################

############# Main Code ###############
def main():
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
    source_col = 'SNP1'
    target_col = 'SNP2'
    
    nodes_path = '../data/networkAnalysisData/randomNetworkAnalysisData_nodes.csv'
    node_col = 'SNPid'
    
    # Load data from edges and nodes CSV files
    edges, nodes = load_data(edges_path, source_col, target_col, nodes_path, node_col)
    
    
    # Create the graph
    G = nx.from_pandas_edgelist(edges, source_col, target_col) 
    '''
    ###THIS GRAPH's indexing starts at 1, if your's starts at 0, change it to 1, or you will have to change a lot of code for it to work properly
    # e.g. get_similar_pairs() and print_high_similarity() functions assume indexing starts at 1
    # also, it skips the top row (the header row) so you can have one row of headers for the columns. 
    # do confirm if the number of nodes in the graph match the number of nodes in your file. no. of nodes in the graph are given by the print_graph_info(G)
    '''
    
    
    
    # Print graph information
    print_graph_info(G) # have to provide path
    
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
    path = '../data/networkAnalysisData/k-cliqueCommunities.png'
    visualize_communities(G, cliques, path=path) # Plots the k-clique communities, the plt.savefig() line may be commented out, check the visualize_communities() function above
    
        
    
    
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


############################ END OF MAIN CODE ##########################

# Check if the file is executed directly, if so, then runs main code
if __name__ == "__main__":
    main()

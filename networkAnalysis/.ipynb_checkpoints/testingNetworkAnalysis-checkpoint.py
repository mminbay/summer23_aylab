'''
HOW TO DO NETWORK ANALYSIS (unweighted)
STEP 1.
    import the file. if in the same directory: 
    import unweightedNetworkAnalysis as unet
    if in a different directory:
    import sys
    sys.path.append('/path/to/networkAnalysis/')  # Replace with the actual path to the 'networkAnalysis' directory
    import unweightedNetworkAnalysis as unet
STEP 2.
    make a graph. Get info and plot it if  needed 
    follow the steps in the Step 2 section below.
STEP 3.
    use the different network analysis functions.
    follow the steps in the Step 3 section below.
    

EXTRA NOTES:
    # all functions have docstring documentation. you can use the following two ways to access them
    # for example for the print_centrality function, the following two will give the args and details
    # Access the function documentation using help()
    help(print_centrality)
    
    # Access the function documentation using __doc__
    print(print_centrality.__doc__)

'''


'''
Step 2 refers to below
The following function is needed to make the Graph:

 1.
    make_graph_from_edges(edges_file_path, source_column_name, target_column_name, nodes_file_path=None, node_column_name=None)
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
 2.
    plot_graph(G, path=None):
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
 3.
    print_graph_info(graph, bool_print=True):
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
        CAN RETRIEVE ALL values from the tuple like: nodes, edges, degree_dict, cc, wcc = print_graph_info(G, bool_print=True)
    """

################### 
Step 3 refers to below

You may need the following functions for network analysis:
 1.
    print_centrality(G, centrality_metric, highest_only=True) 
    Args:
        G is graph, 
        centrality metric can be one of the following:    ['degree', 'harmonic', 'node_betweenness', 'edge_betweenness', 'pagerank'], 
        highest_only prints node(s) with highest centrality of the chosen metric. default is true. if false, centralities for all nodes will be printed
 2.
    CFINDER_communities(G, k=3, path=None)
    Args:
        G (nx.Graph): The input graph.
        k (int, optional): The size of the cliques to search for. Default is 3.
        path (str, optional): The path to save the visualization plot as a PNG file. If not provided, the plot will not be saved.
    NOTE: the graph and colors are generated somewhat randomly, so if you don't like colors, or if there is a weird overlap of the legend and graph, run the code again 
 3.
    print_similarity(G, similarity_metric, similarity_threshold=0.5, print_all_pairs=False)
    Args: 
        G (nx.Graph): The input graph.
        similarity_metric (str): The similarity metric to use. Possible values are 'cosine', 'jaccard', and 'katz'.
        similarity_threshold (float, optional): The threshold value for considering a similarity score as high. Default is 0.5.
        print_all_pairs (bool, optional): Whether to print the similarity scores for all pairs of nodes. Default is False.
 4. 
    plot_degree_distribution(G, curve=True, print_degree_info=False, path=None, log_scale=False)
    Args:
        G (nx.Graph): The input graph.
        curve (bool, optional): Whether to plot the degree distribution as a curve. If False, a histogram is plotted. Default is True.
        print_degree_info (bool, optional): Whether to print degree-related information. Default is False.
        path (str, optional): The file path to save the plot. If not provided, the plot will be displayed interactively (plot.show()).
        log_scale (bool, optional): Whether to use a logarithmic scale for the x-axis (degree). Only applicable when `curve` is True. Default is False.


'''
############ YOUR CODE GOES HERE #############
# example code provided
import unweightedNetworkAnalysis as unet

# making Graph
edges_file_path = '../data/networkAnalysisData/randomNetworkAnalysisData_edges.csv'
source_col = 'SNP1'
target_col = 'SNP2'

G = unet.make_graph_from_edges(edges_file_path, source_col, target_col) # makes the graph

plot_path = '../data/networkAnalysisData/circularNetwork.png'
unet.plot_graph(G, plot_path)

unet.print_graph_info(G, bool_print=True)

# running network analyses
centrality_metrics = ['degree', 'harmonic', 'node_betweenness', 'edge_betweenness', 'pagerank']

for metric in centrality_metrics:
    unet.print_centrality(G, metric, highest_only=True)

communities_path = '../data/networkAnalysisData/k-cliqueCommunities.png'
unet.CFINDER_communities(G, k=4, path=communities_path)







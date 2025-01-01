import random
from matplotlib import pyplot as plt
import networkx as nx
from networkx.algorithms.community import greedy_modularity_communities
import numpy as np
import requests
import random



# Step 1: Load the Network
def load_snap_email_network():
    """
    Loads the email-Eu-core network dataset from the downloaded text file.
    """
    local_file = "email-Eu-core.txt"  # Ensure the filename matches exactly
    print("Loading graph from local file...")
    G = nx.read_edgelist(local_file, nodetype=int, create_using=nx.Graph())
    return G

# Step 2: Perform Modularity-Based Clustering
def modularity_clustering(G):
    """
    Finds clusters (communities) in the graph using greedy modularity optimization.
    Returns the detected communities and their modularity score.
    """
    # Detect communities
    communities = list(greedy_modularity_communities(G))
    
    # Calculate modularity score
    from networkx.algorithms.community import modularity
    mod_score = modularity(G, communities)
    print(f"Modularity Score: {mod_score}")
    
    return communities, mod_score

# Step 3: Save Clusters for Gephi Visualization
def export_to_gephi(G, communities, output_file):
    """
    Exports the graph and community data to a .gexf file for Gephi visualization.
    Each community is assigned a unique attribute 'community'.
    """
    # Assign community labels to nodes
    community_map = {}
    for i, community in enumerate(communities):
        for node in community:
            community_map[node] = i
    
    nx.set_node_attributes(G, community_map, "community")
    nx.write_gexf(G, output_file)
    print(f"Graph exported to {output_file} for Gephi visualization.")

def calculate_distance_quality(G, partition, num_random_graphs=1):
    """
    Calculate the Distance Quality Function (Q_d) for a given graph and partition.

    Parameters:
    - G (networkx.Graph): The observed graph.
    - partition (list of sets): A partition of the nodes into clusters.
    - num_random_graphs (int): Number of random graphs to generate for expected distance calculation.

    Returns:
    - float: The distance quality score (Q_d).
    """

    def compute_observed_distance(cluster, G):
        """
        Compute the observed total pairwise distances within a cluster.
        """
        nodes = list(cluster)
        total_distance = 0
        # Check if the random graph is connected
        if not nx.is_connected(G):
            # Use the diameter of the largest connected component
            largest_cc = max(nx.connected_components(G), key=len)
            subgraph = G.subgraph(largest_cc)
            graph_diameter = nx.diameter(subgraph)
        else:
            # Use the diameter of the entire graph
            graph_diameter = nx.diameter(G)

            graph_diameter = nx.diameter(G)  # Diameter of the random graph

        for i in range(len(nodes)):
            for j in range(len(nodes)):
                
                try:
                    path_length = nx.shortest_path_length(G, source=nodes[i], target=nodes[j])
                except nx.NetworkXNoPath:
                    path_length = 2 * graph_diameter  # Finite penalty for disconnected pairs
                total_distance += path_length
                

        return total_distance / 2    # Divide by 2 for symmetry

    def compute_expected_distance(cluster, G, num_random_graphs):
        """
        Estimate the expected total pairwise distance for a random graph
        with the same degree distribution as G.
        """
        nodes = list(cluster)
        n = len(nodes)
        total_distance = 0
        count = 0
        
        for _ in range(num_random_graphs):
            # Generate a random graph with the same degree distribution
            random_graph = nx.expected_degree_graph([d for _, d in G.degree()], selfloops=False)
            
            # visualize_graph(random_graph)

            # Check if the random graph is connected
            if not nx.is_connected(random_graph):
                # Use the diameter of the largest connected component
                largest_cc = max(nx.connected_components(random_graph), key=len)
                subgraph = random_graph.subgraph(largest_cc)
                random_diameter = nx.diameter(subgraph)
            else:
                # Use the diameter of the entire graph
                random_diameter = nx.diameter(random_graph)

            for i in range(len(nodes)):
                for j in range(len(nodes)):
                    try:
                        path_length = nx.shortest_path_length(random_graph, source=nodes[i], target=nodes[j])
                    except nx.NetworkXNoPath:
                        path_length = 2 * random_diameter  # Finite penalty for disconnected pairs
                    total_distance += path_length

        return total_distance / 2   # Divide by 2 for symmetry

    # Calculate Q_d
    Q_d = 0
    observed_distance = 0
    expected_distance = 0
    for cluster in partition:
        observed_distance = compute_observed_distance(cluster, G)
        expected_distance = compute_expected_distance(cluster, G, num_random_graphs)
        Q_d += expected_distance - observed_distance

    # Normalize Q_d by the total number of nodes for scale invariance
    Q_d /= G.number_of_edges()
    return Q_d, observed_distance, expected_distance, len(G.nodes())

def maximize_distance_quality(
    G, initial_partition, calculate_distance_quality, max_iterations=100, max_no_improvement=10
):
    """
    Maximize the distance quality function (Q_d) and track Q_d values once per iteration.
    Simplified version without checkpoints.
    """
    partition = initial_partition
    max_qd = calculate_distance_quality(G, partition)
    qd_values = [max_qd]  # Track Q_d values for all iterations
    no_improvement_count = 0
    iterations = 0

    while iterations < max_iterations:
        # print(f"Iteration {iterations}: Current Q_d = {qd_values[-1]}")
        improved = False
        best_qd = max_qd

        # Node movement logic
        for node in G.nodes():
            # Remove the node from its current cluster
            for cluster in partition:
                if node in cluster:
                    current_cluster = cluster
                    cluster.remove(node)
                    break

            # Track the best move for this node
            best_cluster = None

            # Test moving the node to each cluster
            for cluster in partition:
                cluster.add(node)
                new_qd = calculate_distance_quality(G, partition)
                print(f'Tried moving node {node} to cluster {cluster} with distance quality of {new_qd}')
                if new_qd > best_qd:
                    best_qd = new_qd
                    best_cluster = cluster
                cluster.remove(node)  # Undo the move

            # Test creating a new cluster for the node
            new_cluster = {node}
            partition.append(new_cluster)
            new_qd = calculate_distance_quality(G, partition)
            print(f'Tried moving node {node} to cluster {cluster} with distance quality of {new_qd}')
            if new_qd > best_qd:
                best_qd = new_qd
                best_cluster = new_cluster
            partition.pop()  # Undo the move

            # Assign the node to the best cluster (if any improvement was found)
            if best_cluster:
                if best_cluster is not new_cluster:
                    best_cluster.add(node)
                else:
                    partition.append(new_cluster)
            else:
                # If no improvement, restore the node to its original cluster
                current_cluster.add(node)

        # Update the global partition and Q_d if improvement occurs
        if best_qd > max_qd:
            max_qd = best_qd
            improved = True
            no_improvement_count = 0
        else:
            no_improvement_count += 1

        # Record Q_d only once per iteration
        qd_values.append(max_qd)

        # Check stopping condition
        if no_improvement_count >= max_no_improvement:
            print("Stopping: No improvement for consecutive iterations.")
            break
        # visualize_graph(G, partition, iterations)
        iterations += 1

    return partition, max_qd, qd_values

def random_partition(G, num_clusters):
    nodes = list(G.nodes())
    random.shuffle(nodes)
    partition = [set() for _ in range(num_clusters)]
    for i, node in enumerate(nodes):
        partition[i % num_clusters].add(node)
    return partition

def visualize_graph(G, partition, iteration=1):
    """
    Visualize the graph with nodes colored by their cluster.
    """
    # Assign a unique color to each cluster
    cluster_colors = {}
    for i, cluster in enumerate(partition):
        for node in cluster:
            cluster_colors[node] = i

    # Create a color list for nodes
    node_colors = [cluster_colors[node] for node in G.nodes()]

    # Plot the graph
    plt.figure(figsize=(8, 6))
    pos = nx.spring_layout(G)  # Spring layout for better visualization
    nx.draw(
        G, pos, with_labels=True, node_color=node_colors, cmap=plt.cm.Set3, node_size=300
    )
    plt.title(f"Graph Clustering at Iteration {iteration}")
    plt.show()

def generate_graph(graph_type, num_nodes):
    """
    Generate a graph of the specified type.

    Parameters:
    - graph_type (str): The type of graph to generate ('connected', 'star', 'cycle', 'grid', etc.).
    - num_nodes (int): The number of nodes in the graph.

    Returns:
    - networkx.Graph: The generated graph.
    """
    if graph_type == 'connected':
        # Generate a complete graph first, then create a random spanning tree
        complete_graph = nx.complete_graph(num_nodes)
        spanning_tree = nx.random_spanning_tree(complete_graph)
        return nx.Graph(spanning_tree)
    elif graph_type == 'star':
        return nx.star_graph(num_nodes - 1)  # Generates a star graph
    elif graph_type == 'cycle':
        return nx.cycle_graph(num_nodes)  # Generates a cycle graph
    elif graph_type == 'complete':
        return nx.complete_graph(num_nodes)  # Generates a fully connected graph
    elif graph_type == 'grid':
        side_length = int(num_nodes**0.5)
        return nx.grid_2d_graph(side_length, side_length)  # Generates a 2D grid
    else:
        raise ValueError(f"Unknown graph type: {graph_type}")

def generate_random_partition(graph, num_clusters):
    """
    Generate a random partition of the nodes in the graph.

    Parameters:
    - graph (networkx.Graph): The graph whose nodes will be partitioned.
    - num_clusters (int): The number of clusters.

    Returns:
    - list of sets: A random partition of the graph's nodes into clusters.
    """
    nodes = list(graph.nodes())
    random.shuffle(nodes)  # Shuffle the nodes randomly
    partition = []
    cluster_size = len(nodes) // num_clusters
    for i in range(num_clusters - 1):
        partition.append(set(nodes[i * cluster_size:(i + 1) * cluster_size]))
    partition.append(set(nodes[(num_clusters - 1) * cluster_size:]))  # Remaining nodes in the last cluster
    return partition

def display_partition(partition):
    """
    Print the generated partition.

    Parameters:
    - partition (list of sets): The partition of nodes to display.
    """
    print("Generated Partition:")
    for i, cluster in enumerate(partition):
        print(f"Cluster {i + 1}: {sorted(cluster)}")

def visualize_graph(G, partition=None):
    """
    Visualize the graph and optionally color nodes by partition.

    Parameters:
    - G (networkx.Graph): The graph to visualize.
    - partition (list of sets, optional): Partition of nodes to color clusters.
    """
    pos = nx.spring_layout(G)
    if partition:
        colors = []
        for node in G.nodes():
            for i, cluster in enumerate(partition):
                if node in cluster:
                    colors.append(i)
                    break
        nx.draw(G, pos, with_labels=True, node_color=colors, cmap=plt.cm.tab20)
    else:
        nx.draw(G, pos, with_labels=True)
    plt.show()

def generate_connected_graph(num_nodes, density=0.5):
    """
    Generates a random connected graph with the specified density.

    Parameters:
    - num_nodes (int): Number of nodes in the graph.
    - density (float): Desired density of the graph (0 < density ≤ 1).

    Returns:
    - networkx.Graph: A connected random graph.
    """
    if density <= 0 or density > 1:
        raise ValueError("Density must be between 0 and 1.")
    if num_nodes < 2:
        raise ValueError("Number of nodes must be at least 2.")

    # Step 1: Create a fully connected graph
    full_graph = nx.complete_graph(num_nodes)

    # Step 2: Generate a random spanning tree (ensures connectivity)
    spanning_tree_edges = list(nx.minimum_spanning_edges(full_graph, algorithm="kruskal", data=False))
    G = nx.Graph()
    G.add_edges_from(spanning_tree_edges)

    # Step 3: Add extra edges to meet the density requirement
    max_edges = num_nodes * (num_nodes - 1) // 2  # Maximum possible edges in a complete graph
    target_edges = int(max_edges * density)

    # Get all possible edges and exclude existing ones
    possible_edges = set(full_graph.edges()) - set(G.edges())
    remaining_edges = list(possible_edges)

    random.shuffle(remaining_edges)
    extra_edges = target_edges - len(G.edges())

    for edge in remaining_edges[:extra_edges]:
        G.add_edge(*edge)

    return G

# Main Script
if __name__ == "__main__":

    # --------------------------------------------------  Baseline Graphs Evaluation  ------------------------------------------------------------ #

    # # Parameters for graph generation
    # num_nodes = 200  # Number of nodes in the graph
    # num_clusters = 10  # Number of random clusters
    # graph_types = [ 'star', 'cycle', 'complete']

    # # Generate and partition graphs
    # for graph_type in graph_types:
    #     print(f"\nGenerating a {graph_type} graph with {num_nodes} nodes...")
    #     G = generate_graph(graph_type, num_nodes)
        
    #     # Random partition
    #     partition = generate_random_partition(G, num_clusters)
        
    #     # Display graph and partition
    #     print(f"Graph type: {graph_type}")
    #     display_partition(partition)


    #     dq, observed_distance, expected_distance, n = calculate_distance_quality(G, partition, 1)
    #     print(f"Distance quality: {dq}, observed_distance: {observed_distance}, expected_distance: {expected_distance}, number of nodes: {n}")
        
        # Visualize the graph
        # visualize_graph(G, partition)


    # ---------------------------------------- Maximize Distance Quality ---------------------------------------- #
    G = generate_connected_graph(600)
    
    # Initial partition, Random Clusters
    partition = generate_random_partition(G, 10)
    dq, observed_distance, expected_distance, n = calculate_distance_quality(G, partition, 1)
    print(f"Distance quality: {dq}, observed_distance: {observed_distance}, expected_distance: {expected_distance}, number of nodes: {n}")


    # Initial partition, Modularity partition
    modularity_clusters = greedy_modularity_communities(G)
    partition = [set(cluster) for cluster in modularity_clusters]


    dq, observed_distance, expected_distance, n = calculate_distance_quality(G, partition, 1)
    print(f"Distance quality: {dq}, observed_distance: {observed_distance}, expected_distance: {expected_distance}, number of nodes: {n}")

    # Perform the search
    # best_partition, max_qd, qd_values = maximize_distance_quality(
    #     G, initial_partition, calculate_distance_quality
    # )

    # Visualize the results
    # print("Best Partition:", best_partition)
    # print("Maximum Q_d:", max_qd)
    # print(f"qd_values.len = {len(qd_values)}")
    # # Plot the progress of Q_d
    # plt.plot(qd_values, marker='o')
    # plt.title("Distance Quality (Q_d) Progress Over Iterations")
    # plt.xlabel("Iteration")
    # plt.ylabel("Q_d")
    # plt.grid()
    # plt.show()

    # # Load the graph
    # print("Loading the email-Eu-core network...")
    # G = load_snap_email_network()
    # print(f"Graph loaded with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    
    # # Perform modularity-based clustering
    # print("Performing modularity-based clustering...")
    # communities, mod_score = modularity_clustering(G)
    # print(f"Detected {len(communities)} communities.")
    # print(f'Detected {mod_score} for modularity scores')
    
    # # Calculate Distance Quality
    # distance_quality = calculate_distance_quality(G, communities)
    # print(f"Distance Quality (Q_d): {distance_quality}")

    # Export the graph with community labels for Gephi
    # output_file = "email_eu_core_modularity_clusters.gexf"
    # export_to_gephi(G, communities, output_file)
    # print("All done!")
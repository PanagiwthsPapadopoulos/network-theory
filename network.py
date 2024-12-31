import random
from matplotlib import pyplot as plt
import networkx as nx
from networkx.algorithms.community import greedy_modularity_communities
import numpy as np
import requests



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

def calculate_distance_quality(G, partition):
    """
    Calculate the Distance Quality Function (Q_d) for a given graph and partition.

    Parameters:
    - G (networkx.Graph): The input graph.
    - partition (list of sets): A partition of the nodes into clusters. 
      Each cluster is a set of nodes.

    Returns:
    - float: The distance quality score (Q_d).
    """
    def compute_average_distance(cluster, G):
        """
        Compute the average pairwise distance within a cluster.
        """
        subgraph = G.subgraph(cluster)
        distances = []
        for u, v in nx.all_pairs_shortest_path_length(subgraph):
            distances.extend(v.values())  # Collect all distances
        num_pairs = len(cluster) * (len(cluster) - 1)  # Total node pairs
        return np.sum(distances) / num_pairs if num_pairs > 0 else 0

    def compute_random_expected_distance(cluster, G):
        """
        Estimate the expected average distance for a random graph with the same node count.
        """
        n = len(cluster)
        if n <= 1:
            return 0  # Single node or empty cluster
        random_graph = nx.gnm_random_graph(n, len(G.edges()))
        while not nx.is_connected(random_graph):
            random_graph = nx.gnm_random_graph(n, len(G.edges()))
        try:
            distances = [nx.shortest_path_length(random_graph, source=u, target=v)
                         for u in random_graph for v in random_graph if u != v]
            return np.mean(distances)
        except nx.NetworkXError:
            return np.inf  # Disconnected random graph

    # Calculate Q_d
    Q_d = 0
    for cluster in partition:
        observed_avg_distance = compute_average_distance(cluster, G)
        expected_avg_distance = compute_random_expected_distance(cluster, G)
        # print(f"Observed Avg: {observed_avg_distance}, Expected Avg: {expected_avg_distance}")
        Q_d += (observed_avg_distance - expected_avg_distance)

    return Q_d


def compute_observed_distance(cluster, G):
    """
    Compute the observed average pairwise distance within a cluster.
    Penalize disconnected pairs with a specified penalty.
    """
    if len(cluster) <= 1:
        return 0  # Single node or empty cluster

    # Subgraph for the cluster
    subgraph = G.subgraph(cluster)
    distances = []
    
    try:
        graph_diameter = nx.diameter(G)
    except nx.NetworkXError:
        graph_diameter = float('inf')  # Infinite for disconnected graphs

    penalty = 3*graph_diameter

    for u in cluster:
        for v in cluster:
            if u != v:
                try:
                    distances.append(nx.shortest_path_length(subgraph, source=u, target=v))
                except nx.NetworkXNoPath:
                    distances.append(penalty)  # Apply penalty for disconnected pairs

    return distances/2 if distances else penalty/2


def compute_expected_distance(cluster, G, num_random_graphs=10):
    """
    Compute the expected average pairwise distance for the same cluster in a random graph model.
    Penalize disconnected pairs with a specified penalty.
    """
    if len(cluster) <= 1:
        return 0  # Single node or empty cluster

    try:
        graph_diameter = nx.diameter(G)
    except nx.NetworkXError:
        graph_diameter = float('inf')  # Infinite for disconnected graphs

    penalty = 3*graph_diameter

    total_distance = 0
    valid_graphs = 0

    for _ in range(num_random_graphs):
        random_graph = nx.configuration_model([d for _, d in G.degree()])
        random_graph = nx.Graph(random_graph)  # Convert to a simple graph
        random_graph.remove_edges_from(nx.selfloop_edges(random_graph))  # Remove self-loops

        # Subgraph for the cluster in the random graph
        subgraph = random_graph.subgraph(cluster)

        distances = []
        for u in cluster:
            for v in cluster:
                if u != v:
                    try:
                        distances.append(nx.shortest_path_length(subgraph, source=u, target=v))
                    except nx.NetworkXNoPath:
                        distances.append(penalty)  # Apply penalty for disconnected pairs

        if distances:
            total_distance += np.mean(distances)
            valid_graphs += 1

    return total_distance / valid_graphs if valid_graphs > 0 else penalty


def compute_inter_cluster_distance(partition, G):
    """
    Compute the average inter-cluster distance for all pairs of clusters.
    """
    inter_cluster_distances = []
    for i, cluster1 in enumerate(partition):
        for j, cluster2 in enumerate(partition):
            if i < j:  # Avoid double-counting pairs
                for u in cluster1:
                    for v in cluster2:
                        try:
                            inter_cluster_distances.append(nx.shortest_path_length(G, source=u, target=v))
                        except nx.NetworkXNoPath:
                            inter_cluster_distances.append(penalty)  # Apply penalty for disconnected pairs

    return np.mean(inter_cluster_distances) if inter_cluster_distances else penalty


def calculate_distance_quality(G, partition, num_random_graphs=10):
    """
    Calculate the Distance Quality Function (Q_d) for a given graph and partition.
    - Includes penalties for disconnected clusters.
    - Does not penalize based on the number of clusters.
    - Normalizes by graph size for scale invariance.
    """


    Q_d_intra = 0
    for cluster in partition:
        observed_distance = compute_observed_distance(cluster, G)
        expected_distance = compute_expected_distance(cluster, G, num_random_graphs)
        Q_d_intra += (expected_distance - observed_distance)

    # Complementarity: Add inter-cluster distances to the evaluation
    # inter_cluster_distance = compute_inter_cluster_distance(partition, G)
    # Q_d = Q_d_intra - inter_cluster_distance
    Q_d = Q_d_intra
    # Normalize by the graph size (scale invariance)
    total_node_pairs = G.number_of_nodes() * (G.number_of_nodes() - 1)
    Q_d /= total_node_pairs

    return Q_d

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

def visualize_graph(G, partition, iteration):
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

# Main Script
if __name__ == "__main__":

    G = nx.karate_club_graph()
    n=5

    G = nx.complete_graph(n)
    partition = [set(G.nodes())] 
    score = calculate_distance_quality(G, partition)
    print(f"Score for connected graph: {score}")

    G = nx.star_graph(n - 1)  # One central node (0), n-1 peripheral nodes
    partition = [set(G.nodes())]  # One cluster with all nodes
    score = calculate_distance_quality(G, partition)
    print(f"Score for star graph: {score}")

    # Initial partition, Single Cluster
    initial_partition = [set(G.nodes())]

    # Initial partition, Random Clusters
    # initial_partition = random_partition(G, 3)  # Randomly split into 3 clusters

    # Initial partition, Modularity partition
    # modularity_clusters = greedy_modularity_communities(G)
    # initial_partition = [set(cluster) for cluster in modularity_clusters]

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
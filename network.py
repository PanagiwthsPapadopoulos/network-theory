import itertools
import random
import time
from joblib import Parallel, delayed
from matplotlib import pyplot as plt
import networkx as nx
from networkx.algorithms.community import greedy_modularity_communities
import numpy as np
import requests
import random
import matplotlib.colors as mcolors
from networkx.algorithms import community
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering



def generate_true_random_connected_graph(num_nodes, edge_probability=0.1):
    """
    Generate a random connected graph with the specified number of nodes and edge probability.

    Parameters:
    - num_nodes (int): Number of nodes in the graph.
    - edge_probability (float): Probability of an edge between two nodes (0 < edge_probability ≤ 1).

    Returns:
    - networkx.Graph: A connected random graph.
    """
    if num_nodes < 2:
        raise ValueError("The graph must have at least 2 nodes.")

    while True:
        # Generate a random graph with the given edge probability
        G = nx.erdos_renyi_graph(num_nodes, edge_probability)
        
        # Check if the graph is connected
        if nx.is_connected(G):
            return G  # Return the graph if it's connected

def load_snap_email_network():
    """
    Loads the email-Eu-core network dataset from the downloaded text file.
    """
    local_file = "email-Eu-core.txt"  
    print("Loading graph from local file...")
    G = nx.read_edgelist(local_file, nodetype=int, create_using=nx.Graph())
    return G

def load_twitter_graph():
    """
    Loads the Twitter Interaction Network for the US Congress from the downloaded file.
    """
    local_file = "congress.edgelist"  
    print("Loading graph from local file...")
    G = nx.read_edgelist(local_file, nodetype=int, create_using=nx.Graph())
    return G

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
            mapping = {i: original_label for i, original_label in enumerate(G.nodes())}
            random_graph = nx.relabel_nodes(random_graph, mapping)
            missing_nodes = [node for node in nodes if node not in random_graph]
            if missing_nodes:
                print(f"Missing nodes from random graph: {missing_nodes}")
                        
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
    cluster_contributions = []
    number_of_edges = G.number_of_edges()
    # print(f'Number of edges: {number_of_edges}')
    # print(f'Graph with nodes: {list(G.nodes())}')
    for cluster in partition:
        
        if len(cluster) > 1:
            observed_distance = compute_observed_distance(cluster, G)
            expected_distance = compute_expected_distance(cluster, G, num_random_graphs)
            cluster_contributions.append( (expected_distance - observed_distance)  / number_of_edges)
            Q_d += (expected_distance - observed_distance) / number_of_edges
        else:
            cluster_contributions.append(0)


    # Normalize Q_d by the total number of nodes for scale invariance
    # Q_d /= number_of_edges
    return Q_d, observed_distance, expected_distance, cluster_contributions

def random_partition(G, num_clusters):
    nodes = list(G.nodes())
    random.shuffle(nodes)
    partition = [set() for _ in range(num_clusters)]
    for i, node in enumerate(nodes):
        partition[i % num_clusters].add(node)
    return partition

def visualize_graph(G, partition, title = ''):
    """
    Visualize the graph with nodes colored by their cluster.
    """
    assigned_nodes = {node for cluster in partition for node in cluster}
    unassigned_nodes = set(G.nodes()) - assigned_nodes

    # Work on a copy of the partition to avoid modifying the original
    partition_copy = partition.copy()

    if unassigned_nodes:
        print(f"Adding {len(unassigned_nodes)} unassigned nodes to a separate cluster.")
        partition_copy.append(unassigned_nodes)


    unique_colors = list(mcolors.TABLEAU_COLORS.values())  # Use Tableau colors for distinctness
    num_clusters = len(partition_copy)
    cluster_colors = {}

    # Assign colors to clusters, ensuring unassigned nodes are gray
    for i, cluster in enumerate(partition):
        color = "gray" if cluster == unassigned_nodes else unique_colors[i % len(unique_colors)]
        for node in cluster:
            cluster_colors[node] = color

    # Assign colors to nodes
    node_colors = [cluster_colors[node] for node in G.nodes()]

    # Plot the graph
    plt.figure()
    pos = nx.spring_layout(G)  # Spring layout for better visualization
    nx.draw(
        G, pos, with_labels=True, node_color=node_colors, cmap=plt.cm.Set3, node_size=300
    )
    # Add title with adjusted padding for visibility
    plt.suptitle(title, fontsize=20)  # Increase padding to avoid overlap
    plt.subplots_adjust(top=0.85)  # Make space at the top for the title
    plt.legend()
    plt.show()

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

def group_high_degree_nodes(G, mean_degree):
    """
    Group nodes with high degrees into proto-clusters based on connectivity.
    """
    high_degree_nodes = [node for node, degree in G.degree() if degree >= mean_degree]
    return [{node} for node in high_degree_nodes]

def proto_clusters_betweenness(G, threshold=0.05):
    """
    Form proto-clusters based on high betweenness centrality.
    Includes a failsafe to ensure some nodes are selected.
    """
    centrality = nx.betweenness_centrality(G)
    sorted_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)
    num_protos = max(1, int(len(G.nodes) * threshold))  # Ensure at least one proto-cluster
    proto_centers = [node for node, _ in sorted_nodes[:num_protos]]
    
    # Failsafe: If no proto-centers, select one random node
    if not proto_centers:
        proto_centers = [random.choice(list(G.nodes))]
    
    return [{center} for center in proto_centers]

def proto_clusters_closeness(G, threshold=0.05):
    """
    Form proto-clusters based on high closeness centrality.
    Includes a failsafe to ensure some nodes are selected.
    """
    centrality = nx.closeness_centrality(G)
    sorted_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)
    num_protos = max(1, int(len(G.nodes) * threshold))  # Ensure at least one proto-cluster
    proto_centers = [node for node, _ in sorted_nodes[:num_protos]]
    
    # Failsafe: If no proto-centers, select one random node
    if not proto_centers:
        proto_centers = [random.choice(list(G.nodes))]
    
    return [{center} for center in proto_centers]

def proto_clusters_spectral(G, num_clusters=5):
    """
    Form proto-clusters using spectral clustering.
    Includes a failsafe to ensure a reasonable number of clusters.
    """
    adjacency_matrix = nx.to_numpy_array(G)
    num_clusters = min(num_clusters, len(G.nodes))  # Ensure clusters do not exceed nodes
    clustering = SpectralClustering(n_clusters=num_clusters, affinity='precomputed', random_state=0)
    labels = clustering.fit_predict(adjacency_matrix)
    
    proto_clusters = [set() for _ in range(num_clusters)]
    for node, label in enumerate(labels):
        proto_clusters[label].add(node)
    
    # Failsafe: If any cluster is empty, assign random nodes
    for cluster in proto_clusters:
        if not cluster:
            cluster.add(random.choice(list(G.nodes)))
    
    return proto_clusters

def proto_clusters_k_core(G, k=3):
    """
    Form proto-clusters using k-core decomposition.
    Includes a failsafe to ensure non-empty clusters.
    """
    k_core = nx.k_core(G, k)
    proto_clusters = [set(k_core.nodes)]
    
    # Failsafe: If no nodes in k-core, fall back to a random node
    if not proto_clusters[0]:
        proto_clusters = [{random.choice(list(G.nodes))}]
    
    return proto_clusters

def proto_clusters_clustering_coeff(G, threshold=0.05):
    """
    Form proto-clusters based on high local clustering coefficient.
    Includes a failsafe to ensure some nodes are selected.
    """
    clustering_coeff = nx.clustering(G)
    sorted_nodes = sorted(clustering_coeff.items(), key=lambda x: x[1], reverse=True)
    num_protos = max(1, int(len(G.nodes) * threshold))  # Ensure at least one proto-cluster
    proto_centers = [node for node, _ in sorted_nodes[:num_protos]]
    
    # Failsafe: If no proto-centers, select one random node
    if not proto_centers:
        proto_centers = [random.choice(list(G.nodes))]
    
    return [{center} for center in proto_centers]

def spectral_clustering(G, num_clusters):
    """
    Perform spectral clustering on a graph.
    
    Parameters:
    - G: networkx.Graph
    - num_clusters: Number of clusters
    
    Returns:
    - clusters: List of sets, each containing nodes in a cluster
    """
    # Compute the normalized Laplacian matrix
    L = nx.normalized_laplacian_matrix(G).toarray()

    # Perform eigendecomposition
    eigvals, eigvecs = np.linalg.eigh(L)

    # Take the first `num_clusters` eigenvectors
    eigvecs_subset = eigvecs[:, :num_clusters]

    # Use k-means to cluster rows of the eigenvector matrix
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    labels = kmeans.fit_predict(eigvecs_subset)

    # Group nodes by cluster
    clusters = [set() for _ in range(num_clusters)]
    for node, label in zip(G.nodes(), labels):
        clusters[label].add(node)

    return clusters

def label_propagation_clustering(G):
    """
    Perform label propagation clustering on a graph.

    Parameters:
    - G: networkx.Graph

    Returns:
    - clusters: List of sets, each containing nodes in a cluster
    """
    # Use NetworkX's built-in label propagation
    labels = nx.algorithms.community.asyn_lpa_communities(G, weight=None)
    clusters = [set(community) for community in labels]
    return clusters

def cluster_initialization(G):
    import networkx.algorithms.community as nx_comm

    # Perform modularity-based clustering (Louvain method as an example)
    communities = nx_comm.greedy_modularity_communities(G)

    # Convert the result into a list of sets
    clusters = [set(community) for community in communities]

    # clusters = [{node} for node in G.nodes()]

    return clusters   

def distance_based_initialization(G, radius=1):
    """
    Initialize clusters by grouping nodes with small pairwise shortest-path distances.

    Parameters:
    - G: networkx.Graph
    - radius: Maximum distance within a cluster

    Returns:
    - clusters: List of sets, each containing nodes in a cluster
    """
    unvisited = set(G.nodes())
    clusters = []

    while unvisited:
        # Start with an arbitrary node
        node = unvisited.pop()
        cluster = {node}

        # Expand the cluster to include nodes within the radius
        for neighbor in nx.single_source_shortest_path_length(G, source=node, cutoff=radius):
            if neighbor in unvisited:
                cluster.add(neighbor)
                unvisited.remove(neighbor)

        clusters.append(cluster)

    return clusters

def expand_clusters(G, proto_clusters):
    """
    Expand proto-clusters outward by adding nodes iteratively.
    """
    unassigned_nodes = set(G.nodes()) - set(node for cluster in proto_clusters for node in cluster)
    clusters = proto_clusters
    while True:
        # Create a set of all nodes currently assigned to clusters
        assigned_nodes = set().union(*clusters)

        # Identify unassigned nodes
        unassigned_nodes = set(G.nodes()) - assigned_nodes
        changes = {}
        for cluster in clusters:
            changes[frozenset(cluster)] = []
        # Check if the set is empty
        if not unassigned_nodes: 
            break

        # Map unassigned nodes to their candidate clusters
        node_to_candidates = {}

        for node in unassigned_nodes:
            # Find clusters containing the neighbors of this node
            candidate_clusters = set()
            neighbors = set(G.neighbors(node))

            for cluster in clusters:
                if neighbors & cluster:  # Check if any neighbor belongs to the cluster
                    candidate_clusters.add(frozenset(cluster))  # Store as frozenset for immutability

            # Only add to the dictionary if candidate_clusters is not empty
            if candidate_clusters:
                node_to_candidates[node] = candidate_clusters

            # Map the node to its candidate clusters
            node_to_candidates[node] = candidate_clusters

        for key, value in list(node_to_candidates.items()):
            if not value: node_to_candidates.pop(key)
       
        centrality = nx.betweenness_centrality(G)
        sorted_nodes = sorted(G.nodes, key=lambda node: centrality[node], reverse=True)

        


        # Iterate over each unassigned node
        for node in sorted_nodes:
            if(node in node_to_candidates.keys()):
                candidate_clusters = node_to_candidates[node]
                
                best_cluster = None
                min_distance = float('inf')

                # Test adding the node to each candidate cluster
                for candidate_cluster in candidate_clusters:
                    candidate_cluster = set(candidate_cluster)  # Convert frozenset to set for manipulation
                    new_distance, _ = calculate_distance_quality_ercq(G, [candidate_cluster | {node}])

                    # Find the cluster with the minimal distance
                    if new_distance < min_distance:
                        min_distance = new_distance
                        best_cluster = candidate_cluster
                # Assign the node to the best cluster found
                if best_cluster is not None:
                    for original_cluster in clusters:
                        if original_cluster == best_cluster:
                            # original_cluster.add(node)
                            changes[frozenset(original_cluster)].append(node)
                            

        new_clusters = []
        # Output the transformed clusters
        for cluster, nodes in changes.items():
            temp_set = set()
            for item in cluster:
                temp_set.add(item)
            temp_set.update(nodes)
            new_clusters.append(temp_set)
        clusters = new_clusters
        
        

    return clusters

def expand_clusters_in_parallel(G, proto_clusters):
    """
    Expand proto-clusters outward by adding nodes iteratively.
    """
    unassigned_nodes = set(G.nodes()) - set(node for cluster in proto_clusters for node in cluster)
    clusters = proto_clusters
    while True:
        # Create a set of all nodes currently assigned to clusters
        assigned_nodes = set().union(*clusters)
        unassigned_nodes = set(G.nodes()) - assigned_nodes
        changes = {}
        for cluster in clusters:
            changes[frozenset(cluster)] = []

        # Check if the set is empty
        if not unassigned_nodes: 
            break

        # Map unassigned nodes to their candidate clusters
        node_to_candidates = {}

        # Parallelized: Find candidate clusters for each unassigned node
        def find_candidates(node, clusters, G):
            neighbors = set(G.neighbors(node))
            candidate_clusters = {frozenset(cluster) for cluster in clusters if neighbors & cluster}
            return node, candidate_clusters

        node_to_candidates = dict(Parallel(n_jobs=-1)(delayed(find_candidates)(node, clusters, G) for node in unassigned_nodes))

        # Filter out nodes with no candidate clusters
        node_to_candidates = {node: candidates for node, candidates in node_to_candidates.items() if candidates}

        for key, value in list(node_to_candidates.items()):
            if not value: node_to_candidates.pop(key)
       
        # Calculate betweenness centrality
        centrality = nx.betweenness_centrality(G)
        sorted_nodes = sorted(node_to_candidates.keys(), key=lambda node: centrality[node], reverse=True)


        # Parallelized: Evaluate clusters for each node
        def evaluate_cluster(node, candidate_cluster, G):
            new_distance, _ = calculate_distance_quality_ercq(G, [candidate_cluster | {node}])
            return node, candidate_cluster, new_distance

        results = Parallel(n_jobs=-1)(
            delayed(evaluate_cluster)(node, candidate_cluster, G)
            for node in sorted_nodes
            for candidate_cluster in node_to_candidates[node]
        )

        # Assign nodes to the best clusters based on evaluation
        changes = {frozenset(cluster): [] for cluster in clusters}
        for node, candidate_cluster, new_distance in results:
            best_cluster = min(
                (candidate_cluster for candidate_cluster in node_to_candidates[node]),
                key=lambda cluster: calculate_distance_quality_ercq(G, [cluster | {node}])[0]
            )
            changes[frozenset(best_cluster)].append(node)

        # Update clusters with changes
        def update_cluster(cluster, nodes):
            updated_cluster = set(cluster)
            updated_cluster.update(nodes)
            return updated_cluster

        clusters = Parallel(n_jobs=-1)(
            delayed(update_cluster)(cluster, changes[frozenset(cluster)])
            for cluster in clusters
        )

    return clusters

def compute_cluster_distance(cluster, G):
    """
    Compute the total pairwise distances within a cluster.
    """
    nodes = list(cluster)
    total_distance = 0
    for i in range(len(nodes)):
        for j in range(len(nodes)):
            try:
                total_distance += nx.shortest_path_length(G, source=nodes[i], target=nodes[j])
            except nx.NetworkXNoPath:
                total_distance += 2 * nx.diameter(G)  # Penalty for disconnected pairs
    return total_distance / 2

def refine_clusters_with_merging(G, initial_clusters, num_random_graphs=1):
    """
    Refine clusters by attempting to merge them if it improves Q_d.
    """
    clusters = initial_clusters
    improved = True

    while improved:
        improved = False
        best_merge = None
        current_qd, contributions = calculate_distance_quality_ercq(G, clusters)
        cluster_contributions = list(zip(clusters, contributions))
        sorted_contributions = sorted(cluster_contributions, key=lambda x: x[1])


        # Evaluate all pairs of clusters
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                cluster_i = sorted_contributions[i][0]
                cluster_j = sorted_contributions[j][0]
                merged_cluster = cluster_i | cluster_j

                # New partition
                new_partition = [c for c in clusters if c != cluster_i and c != cluster_j] + [merged_cluster]
                new_qd, _ = calculate_distance_quality_ercq(G, new_partition)

                if new_qd > current_qd:
                    clusters = new_partition
                    current_qd = new_qd
                    improved = True
                    break  # Restart the iteration after a successful merge

            if improved:
                break

    return clusters

def refine_clusters_with_merging_parallel(G, initial_clusters, num_random_graphs=1):
    """
    Refine clusters by attempting to merge them if it improves Q_d in parallel.
    """
    clusters = initial_clusters
    improved = True

    def evaluate_merge(clusters, i, j, G):
        """
        Evaluate the distance quality after merging clusters i and j.
        """
        # Create a new partition by merging clusters[i] and clusters[j]
        merged_clusters = clusters[:i] + clusters[i+1:j] + clusters[j+1:] + [clusters[i] | clusters[j]]
        
        # Calculate the new distance quality for the merged clusters
        new_qd, _,  = calculate_distance_quality_ercq(G, merged_clusters)
        
        return (i, j, new_qd)
    
    def find_best_merge(clusters, G, current_qd):
        """
        Find the best pair of clusters to merge in parallel.
        """
        results = Parallel(n_jobs=-1)(  # Use all available CPU cores
            delayed(evaluate_merge)(clusters, i, j, G)
            for i in range(len(clusters)) for j in range(i + 1, len(clusters))
        )
        
        # Filter only the merges that improve Q_d
        valid_results = [res for res in results if res[2] > current_qd]
        
        # If no valid merges, return None
        if not valid_results:
            return None
        
        # Return the best merge (highest Q_d)
        return max(valid_results, key=lambda x: x[2])

    while improved:
        current_qd, _ = calculate_distance_quality_ercq(G, clusters)
        best_merge = find_best_merge(clusters, G, current_qd)
        
        if not best_merge:
            break  # No more valid merges

        i, j, new_qd = best_merge
        
        # Perform the merge
        merged_cluster = clusters[i] | clusters[j]
        clusters = [clusters[k] for k in range(len(clusters)) if k != i and k != j] + [merged_cluster]
        
        # Update current Q_d
        current_qd = new_qd


        

    return clusters

def maximize_distance_quality(G, mode='proto_clusters_betweenness', parameter=0.05):

    # Split the graph into its connected components
    components = [G.subgraph(c).copy() for c in nx.connected_components(G)]
    
    global_clusters = []
    
    for i, component in enumerate(components):
        # Perform initial clustering
        mean_degree = sum(dict(component.degree()).values()) / len(component.nodes())
        
        if mode == 'proto_clusters_betweenness':
            proto_clusters = proto_clusters_betweenness(component, parameter)
        if mode == 'proto_clusters_closeness':
            proto_clusters = proto_clusters_closeness(component, parameter)
        if mode == 'proto_clusters_spectral':
            proto_clusters = proto_clusters_spectral(component, parameter)
        if mode == 'proto_clusters_k_core':
            proto_clusters = proto_clusters_k_core(component, parameter)
        if mode == 'proto_clusters_clustering_coeff':
            proto_clusters = proto_clusters_clustering_coeff(component, parameter)

        # Expand clusters
        initial_clusters = expand_clusters_in_parallel(component, proto_clusters)

        # Refine clusters with merging
        final_clusters = refine_clusters_with_merging_parallel(component, initial_clusters, 1)

        # Add the final clusters of this component to the global clusters
        global_clusters.extend(final_clusters)

    
    return final_clusters

def evaluate_clusters(G, clusters):
    """
    Evaluate modularity and distance quality of given clusters.
    """
    modularity = community.modularity(G, clusters)
    dq, observed_distance, expected_distance, _ = calculate_distance_quality(G, clusters, 3)
    return modularity, dq

def compute_expected_distance_ercq(cluster, G):
    """
    Compute the expected pairwise distance for a random graph using ERCQ.
    This approximates the expected distance without generating random graphs.
    """
    n = len(cluster)
    if n <= 1:
        return 0  # Single node or empty cluster

    # Approximate expected distance based on degree distribution
    total_degree = sum(dict(G.degree(cluster)).values())
    expected_distance = (total_degree / (n * (n - 1)))  # Simplified approximation
    return expected_distance / 2

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

def calculate_distance_quality_ercq(G, partition):
    """
    Calculate the Distance Quality Function (Q_d) using ERCQ for expected distances.
    """
    Q_d = 0
    cluster_contributions = []
    number_of_edges = G.number_of_edges()

    for cluster in partition:
        observed_distance = compute_observed_distance(cluster, G)
        expected_distance = compute_expected_distance_ercq(cluster, G)
        cluster_contribution = (expected_distance - observed_distance) / number_of_edges
        cluster_contributions.append(cluster_contribution)
        Q_d += expected_distance - observed_distance

    Q_d /= number_of_edges  # Normalize by number of edges
    return Q_d, cluster_contributions


# Main Script
if __name__ == "__main__":

    # ---------------------------------------- Maximize Distance Quality ---------------------------------------- #

    NUMBER_OF_NODES = 100
    PROBABILITY = 0.15
    
    G = generate_true_random_connected_graph(NUMBER_OF_NODES, PROBABILITY)
    # # G = load_snap_email_network()
    # print(f"Graph loaded with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.\n")


    # # Split graph using modularity
    # modularity_clusters = list(community.greedy_modularity_communities(G))
    # modularity_score, dq_modularity = evaluate_clusters(G, modularity_clusters)
    # print(f"Modularity-based clusters: Modularity = {modularity_score}, Distance Quality = {dq_modularity}\n")

    # # Split graph using distance quality maximization
    # dq_clusters = maximize_distance_quality(G)
    # modularity_score_dq, dq_dq = evaluate_clusters(G, dq_clusters)
    # print(f"Distance-quality-based clusters: Modularity = {modularity_score_dq}, Distance Quality = {dq_dq}\n")



    # ---------------------------------------- Graph Clustering using different Methods ---------------------------------------- #

    # G = load_twitter_graph()
    print(f"Graph loaded with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.\n")

    # Modularity based Clustering
    start_time = time.time()
    modularity_clusters = list(community.greedy_modularity_communities(G))
    modularity_score, dq_modularity = evaluate_clusters(G, modularity_clusters)
    end_time = time.time()
    export_to_gephi(G, modularity_clusters, 'modularity_clusters.gexf')
    print(f"Modularity-based clusters: Modularity = {modularity_score}, Distance Quality = {dq_modularity}, Duration: {end_time-start_time}\n")


    # Distance Quality based Clustering with proto-clusters betweeness 0.05
    # start_time = time.time()
    # dq_clusters = maximize_distance_quality(G, 'proto_clusters_betweenness', 0.01)
    # modularity_score, dq_modularity = evaluate_clusters(G, dq_clusters)
    # end_time = time.time()
    # export_to_gephi(G, dq_clusters, 'dq_clusters_betweenness_0.01.gexf')
    # print(f"Distance Quality based Clustering with proto-clusters betweeness 0.01: Modularity = {modularity_score}, Distance Quality = {dq_modularity}, Duration: {end_time-start_time}\n")


    # Distance Quality based Clustering with proto-clusters betweeness 0.1
    # start_time = time.time()
    # dq_clusters = maximize_distance_quality(G, 'proto_clusters_betweenness', 0.03)
    # modularity_score, dq_modularity = evaluate_clusters(G, dq_clusters)
    # end_time = time.time()
    # export_to_gephi(G, dq_clusters, 'dq_clusters_betweenness_0.03.gexf')
    # print(f"Distance Quality based Clustering with proto-clusters betweeness 0.03: Modularity = {modularity_score}, Distance Quality = {dq_modularity}, Duration: {end_time-start_time}\n")


    # Distance Quality based Clustering with proto-clusters closeness 0.05
    # start_time = time.time()
    # dq_clusters = maximize_distance_quality(G, 'proto_clusters_closeness', 0.01)
    # modularity_score, dq_modularity = evaluate_clusters(G, dq_clusters)
    # end_time = time.time()
    # export_to_gephi(G, dq_clusters, 'proto_clusters_closeness_0.01.gexf')
    # print(f"Distance Quality based Clustering with proto-clusters closeness 0.01: Modularity = {modularity_score}, Distance Quality = {dq_modularity}, Duration: {end_time-start_time}\n")

    # Distance Quality based Clustering with proto-clusters closeness 0.1
    # start_time = time.time()
    # dq_clusters = maximize_distance_quality(G, 'proto_clusters_closeness', 0.03)
    # modularity_score, dq_modularity = evaluate_clusters(G, dq_clusters)
    # end_time = time.time()
    # export_to_gephi(G, dq_clusters, 'proto_clusters_closeness_0.03.gexf')
    # print(f"Distance Quality based Clustering with proto-clusters closeness 0.03: Modularity = {modularity_score}, Distance Quality = {dq_modularity}, Duration: {end_time-start_time}\n")

    # # Distance Quality based Clustering with proto-clusters spectral 5
    # start_time = time.time()
    # dq_clusters = maximize_distance_quality(G, 'proto_clusters_spectral', 5)
    # modularity_score, dq_modularity = evaluate_clusters(G, dq_clusters)
    # end_time = time.time()
    # export_to_gephi(G, dq_clusters, 'proto_clusters_spectral_5.gexf')
    # print(f"Distance Quality based Clustering with proto-clusters spectral 5 eigenvectors: Modularity = {modularity_score}, Distance Quality = {dq_modularity}, Duration: {end_time-start_time}\n")

    # # Distance Quality based Clustering with proto-clusters spectral 10
    # start_time = time.time()
    # dq_clusters = maximize_distance_quality(G, 'proto_clusters_spectral', 10)
    # modularity_score, dq_modularity = evaluate_clusters(G, dq_clusters)
    # end_time = time.time()
    # export_to_gephi(G, dq_clusters, 'proto_clusters_spectral_10.gexf')
    # print(f"Distance Quality based Clustering with proto-clusters spectral 10 eigenvectors: Modularity = {modularity_score}, Distance Quality = {dq_modularity}, Duration: {end_time-start_time}\n")

    # # Distance Quality based Clustering with proto-clusters spectral 15
    # start_time = time.time()
    # dq_clusters = maximize_distance_quality(G, 'proto_clusters_spectral', 15)
    # modularity_score, dq_modularity = evaluate_clusters(G, dq_clusters)
    # end_time = time.time()
    # export_to_gephi(G, dq_clusters, 'proto_clusters_spectral_15.gexf')
    # print(f"Distance Quality based Clustering with proto-clusters spectral 15 eigenvectors: Modularity = {modularity_score}, Distance Quality = {dq_modularity}, Duration: {end_time-start_time}\n")

    # # Distance Quality based Clustering with proto-clusters k-core Decomposition 3
    # start_time = time.time()
    # dq_clusters = maximize_distance_quality(G, 'proto_clusters_k_core', 3)
    # modularity_score, dq_modularity = evaluate_clusters(G, dq_clusters)
    # end_time = time.time()
    # export_to_gephi(G, dq_clusters, 'proto_clusters_k_core_3.gexf')
    # print(f"Distance Quality based Clustering with proto-clusters k-core Decomposition 3 nodes: Modularity = {modularity_score}, Distance Quality = {dq_modularity}, Duration: {end_time-start_time}\n")

    # # Distance Quality based Clustering with proto-clusters k-core Decomposition 6
    # start_time = time.time()
    # dq_clusters = maximize_distance_quality(G, 'proto_clusters_k_core', 6)
    # modularity_score, dq_modularity = evaluate_clusters(G, dq_clusters)
    # end_time = time.time()
    # export_to_gephi(G, dq_clusters, 'proto_clusters_k_core_6.gexf')
    # print(f"Distance Quality based Clustering with proto-clusters k-core Decomposition 6 nodes: Modularity = {modularity_score}, Distance Quality = {dq_modularity}, Duration: {end_time-start_time}\n")

    # # Distance Quality based Clustering with proto-clusters k-core Decomposition 10
    # start_time = time.time()
    # dq_clusters = maximize_distance_quality(G, 'proto_clusters_k_core', 10)
    # modularity_score, dq_modularity = evaluate_clusters(G, dq_clusters)
    # end_time = time.time()
    # export_to_gephi(G, dq_clusters, 'proto_clusters_k_core_10.gexf')
    # print(f"Distance Quality based Clustering with proto-clusters k-core Decomposition 10 nodes: Modularity = {modularity_score}, Distance Quality = {dq_modularity}, Duration: {end_time-start_time}\n")
    
    # Distance Quality based Clustering with proto-clusters clustering_coeff 0.05
    start_time = time.time()
    dq_clusters = maximize_distance_quality(G, 'proto_clusters_clustering_coeff', 0.05)
    modularity_score, dq_modularity = evaluate_clusters(G, dq_clusters)
    end_time = time.time()
    export_to_gephi(G, dq_clusters, 'proto_clusters_clustering_coeff_0.05.gexf')
    print(f"Distance Quality based Clustering with proto-clusters clustering_coeff 0.01: Modularity = {modularity_score}, Distance Quality = {dq_modularity}, Duration: {end_time-start_time}\n")

    # Distance Quality based Clustering with proto-clusters clustering_coeff 0.1
    # start_time = time.time()
    # dq_clusters = maximize_distance_quality(G, 'proto_clusters_clustering_coeff', 0.03)
    # modularity_score, dq_modularity = evaluate_clusters(G, dq_clusters)
    # end_time = time.time()
    # export_to_gephi(G, dq_clusters, 'proto_clusters_clustering_coeff_0.03.gexf')
    # print(f"Distance Quality based Clustering with proto-clusters clustering_coeff 0.03: Modularity = {modularity_score}, Distance Quality = {dq_modularity}, Duration: {end_time-start_time}\n")
    
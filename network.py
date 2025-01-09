import itertools
import random
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

    # Precompute shortest paths for the observed graph
    # shortest_paths = dict(nx.all_pairs_shortest_path_length(G))

    # visualize_graph(G, partition)

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
                    # path_length = shortest_paths[nodes[i]][nodes[j]]
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
            # print(f"List of nodes: {list(G.nodes())}")
            # print(f"List of random nodes: {list(random_graph.nodes())}")
            missing_nodes = [node for node in nodes if node not in random_graph]
            if missing_nodes:
                print(f"Missing nodes from random graph: {missing_nodes}")
            
            # Precompute shortest paths for the random graph
            random_shortest_paths = dict(nx.all_pairs_shortest_path_length(random_graph))
            
            # Check if the random graph is connected
            if not nx.is_connected(random_graph):
                # Use the diameter of the largest connected component
                largest_cc = max(nx.connected_components(random_graph), key=len)
                subgraph = random_graph.subgraph(largest_cc)
                random_diameter = nx.diameter(subgraph)
            else:
                # Use the diameter of the entire graph
                random_diameter = nx.diameter(random_graph)
            # print(f'From compute_expected_distance: Nodes: {nodes}')
            for i in range(len(nodes)):
                for j in range(len(nodes)):
                    try:
                        # print(f'Trying from source node {nodes[i]} to target node {nodes[j]}')
                        path_length = nx.shortest_path_length(random_graph, source=nodes[i], target=nodes[j])
                        # path_length = random_shortest_paths[nodes[i]][nodes[j]]
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

    # partition.pop(unassigned_nodes)
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

    # Step 1: Generate a random graph
    p = density  # Probability of edge creation
    G = nx.gnp_random_graph(num_nodes, p)

    # Step 2: Ensure the graph is connected
    while not nx.is_connected(G):
        # Find all connected components
        components = list(nx.connected_components(G))

        # Randomly choose two components and connect them
        c1, c2 = random.sample(components, 2)
        u = random.choice(list(c1))
        v = random.choice(list(c2))
        G.add_edge(u, v)

    return G

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
        # print('Clusters:')
        # print(clusters)
        # Visualize graph
        # visualize_graph(G, clusters, f'Proto Clusters with mean degree of ')

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

        # print(node_to_candidates.keys())
        
        # sorted_nodes = [node for node in sorted_nodes if node not in node_to_candidates.keys()]


        # Iterate over each unassigned node
        for node in sorted_nodes:
            if(node in node_to_candidates.keys()):
                candidate_clusters = node_to_candidates[node]
                
                best_cluster = None
                min_distance = float('inf')

                # Test adding the node to each candidate cluster
                for candidate_cluster in candidate_clusters:
                    candidate_cluster = set(candidate_cluster)  # Convert frozenset to set for manipulation
                    # current_distance = compute_cluster_distance(candidate_cluster, G)
                    # print(candidate_cluster | {node})
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
            # print(f'Cluster {cluster} with nodes {nodes}')
            temp_set = set()
            for item in cluster:
                temp_set.add(item)
            temp_set.update(nodes)
            new_clusters.append(temp_set)
        # new_clusters.pop()
        clusters = new_clusters
        
        

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
    # max_qd, _, _, _ = calculate_distance_quality(G, clusters, num_random_graphs)

    while improved:
        improved = False
        best_merge = None
        # best_qd = max_qd
        # print(f'From refine_clusters_with_merging: Sending clusters: {clusters}')
        current_qd, contributions = calculate_distance_quality_ercq(G, clusters)
        # current_qd, _, _, contributions = calculate_distance_quality(G, clusters, 1)
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
                # new_qd, _ , _, _= calculate_distance_quality(G, new_partition, 1)
                new_qd, _ = calculate_distance_quality_ercq(G, new_partition)

                if new_qd > current_qd:
                    # print(f"Merging clusters {cluster_i} and {cluster_j} improved Q_d from {current_qd} to {new_qd}")
                    clusters = new_partition
                    current_qd = new_qd
                    improved = True
                    break  # Restart the iteration after a successful merge

            if improved:
                # print('Improvement implemented, starting while loop again...')
                break


                # merged_clusters = clusters[:i] + clusters[i+1:j] + clusters[j+1:] + [clusters[i] | clusters[j]]
                # # new_clusters = [c for k, c in enumerate(clusters) if k != i and k != j] + [merged_cluster]
                # new_qd, _, _, _ = calculate_distance_quality(G, merged_clusters, num_random_graphs)
                # print(f'Checking clusters {i, j} with {clusters[i]} and {clusters[j]} and distance quality of {new_qd} and best distance quality of {best_qd}')
                # if new_qd > best_qd:
                #     best_qd = new_qd
                #     best_merge = (i, j)

        # # Perform the best merge
        # if best_merge:
        #     print(f'Better merge with {best_merge} with qd {new_qd}')
        #     i, j = best_merge
        #     merged_cluster = clusters[i].union(clusters[j])
        #     clusters = [c for k, c in enumerate(clusters) if k != i and k != j] + [merged_cluster]
        #     improved = True

    return clusters

def maximize_distance_quality(G):

    # Split the graph into its connected components
    components = [G.subgraph(c).copy() for c in nx.connected_components(G)]
    
    global_clusters = []
    
    for i, component in enumerate(components):
        # print(f"Processing connected component {i + 1} with {len(component.nodes())} nodes...")

        # Step 2: Perform initial clustering
        mean_degree = sum(dict(component.degree()).values()) / len(component.nodes())
        # proto_clusters = group_high_degree_nodes(component, mean_degree)
        # proto_clusters = cluster_initialization(component)
        # proto_clusters = label_propagation_clustering(component)
        # proto_clusters = distance_based_initialization(G, 2)
        proto_clusters = proto_clusters_betweenness(component, 0.1)


        # print(f'Finished Protoclusters')

        # Expand clusters
        initial_clusters = expand_clusters(component, proto_clusters)
        modularity_score, dq_modularity = evaluate_clusters(G, initial_clusters)
        
        qd, _, _, scores = calculate_distance_quality(component, initial_clusters)
        # print(f'\nScores: {scores}\n')
        print(f"Premature max distance quality clusters: Modularity = {modularity_score}, Distance Quality = {dq_modularity}\n")
        # visualize_graph(G, initial_clusters)
        # Step 3: Refine clusters with merging
        final_clusters = refine_clusters_with_merging(component, initial_clusters, 1)
        # final_clusters = initial_clusters        

        # Add the final clusters of this component to the global clusters
        global_clusters.extend(final_clusters)

    
    return final_clusters


def evaluate_clusters(G, clusters):
    """
    Evaluate modularity and distance quality of given clusters.
    """
    modularity = community.modularity(G, clusters)
    # print('Calculated modularity')
    dq, observed_distance, expected_distance, _ = calculate_distance_quality(G, clusters, 10)
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
                    # path_length = shortest_paths[nodes[i]][nodes[j]]
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

    NUMBER_OF_NODES = 130
    PROBABILITY = 0.8
    
    G = generate_true_random_connected_graph(NUMBER_OF_NODES, PROBABILITY)
    # G = load_snap_email_network()
    print(f"Graph loaded with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.\n")


    # Step 1: Split graph using modularity
    modularity_clusters = list(community.greedy_modularity_communities(G))
    modularity_score, dq_modularity = evaluate_clusters(G, modularity_clusters)
    print(f"Modularity-based clusters: Modularity = {modularity_score}, Distance Quality = {dq_modularity}\n")

    # Step 2: Split graph using distance quality maximization
    dq_clusters = maximize_distance_quality(G)
    modularity_score_dq, dq_dq = evaluate_clusters(G, dq_clusters)
    print(f"Distance-quality-based clusters: Modularity = {modularity_score_dq}, Distance Quality = {dq_dq}\n")



    # visualize_graph(G, modularity_clusters, 'Modularity Clusters')
    # visualize_graph(G, dq_clusters, 'Distance Quality Clusters')

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

    # export_to_gephi(G, modularity_clusters, 'modularity_clusters')
    # export_to_gephi(G, dq_clusters, 'distance_quality_clusters')
    # print("All done!")
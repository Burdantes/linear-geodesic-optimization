import networkx as nx
import numpy as np
import sklearn.cluster

import utility

def get_cluster_center(cluster):
    """Find the point in the cluster nearest to its centroid."""
    sphere_points = [
        utility.get_sphere_point((data['lat'], data['long']))
        for _, data in cluster
    ]
    centroid_direction = sum(sphere_points)
    center_index = np.argmax([
        centroid_direction @ sphere_point
        for sphere_point in sphere_points
    ])

    return cluster[center_index]

def cluster_graph(graph, eps, min_samples):
    """
    Simplify a graph using the DBSCAN algorithm.

    As input, take a graph whose vertices have `lat` (latitude) and
    `long` (longitude) attributes. The edges must also have associated
    `rtt` (round trip time, or RTT) attributes.

    For the given graph, cluster the vertices according to the input
    parameters. Then, for each cluster, pick a representative node.
    Generate a new graph on these representative nodes, where an edge
    exists between two representative nodes if there is an edge between
    their corresponding clusters. If there are multiple such edges
    between clusters, the associated RTT is the minimal one.
    """
    nodes = list(graph.nodes(data=True))

    if 'cluster' in nodes[0][1]:
        cluster_labels = [data['cluster'] for _, data in nodes]
        cluster_count = max(cluster_labels) + 1
    else:
        # Compute clusters. Ensure that each cluster has a unique label.
        # In particular, nodes that are labeled as "noise" by the
        # algorithm are instead treated as clusters of size 1
        clustering = sklearn.cluster.DBSCAN(
            eps=eps, min_samples=min_samples,
            metric=utility.get_spherical_distance
        )
        clustering.fit([
            utility.get_sphere_point((data['lat'], data['long']))
            for _, data in nodes
        ])
        cluster_labels = list(clustering.labels_)
        cluster_count = max(cluster_labels) + 1
        for i in range(len(cluster_labels)):
            if cluster_labels[i] == -1:
                cluster_labels[i] = cluster_count
                cluster_count += 1

    new_graph = nx.Graph()

    # Copy nodes to new graph
    clusters = [[] for _ in range(cluster_count)]
    cluster_centers = []
    for (node, data), label in zip(nodes, cluster_labels):
        clusters[label].append((node, data))
    for cluster in clusters:
        node, data = get_cluster_center(cluster)
        cluster_centers.append(node)
        new_graph.add_node(node, **data)

    # Copy edges to new graph
    for i, cluster_i in enumerate(clusters):
        for j, cluster_j in enumerate(clusters[i+1:], start=i+1):
            rtt = min([
                graph.edges[node_i,node_j]['rtt']
                for node_i, _ in cluster_i
                for node_j, _ in cluster_j
                if (node_i, node_j) in graph.edges
            ], default=np.inf)
            if rtt != np.inf:
                new_graph.add_edge(cluster_centers[i], cluster_centers[j],
                                   rtt=rtt)

    return new_graph

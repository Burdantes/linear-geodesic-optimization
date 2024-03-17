"""File containing utility go generate a graphml file."""

import argparse
import csv

from GraphRicciCurvature.OllivierRicci import OllivierRicci
import networkx as nx
import numpy as np

import cluster_probes
import utility

import pandas as pd
import gudhi as gd
from sklearn import manifold
import json
import sys
sys.path[0] = '../src/'
from utils import *
import matplotlib.pyplot as plt

# project_dir = '/Users/loqmansalamatian/Documents/GitHub/linear-geodesic-optimization/'
def minimize_id_removal(rtt_violation_list):
    id_counts = {}

    for id_source, id_target in rtt_violation_list:
        id_counts[id_source] = id_counts.get(id_source, 0) + 1
        id_counts[id_target] = id_counts.get(id_target, 0) + 1

    ids_to_remove = set()

    while rtt_violation_list:
        # Find id with maximum count
        max_id = max(id_counts, key=id_counts.get)
        ids_to_remove.add(max_id)

        # Remove all lines containing max_id
        rtt_violation_list = [line for line in rtt_violation_list if max_id not in line[:2]]

        # Reset id_counts for the next iteration
        id_counts = {}
        for line in rtt_violation_list:
            id_counts[line[0]] = id_counts.get(line[0], 0) + 1
            id_counts[line[1]] = id_counts.get(line[1], 0) + 1

    return ids_to_remove

def get_graph(
    probes_filename, latencies_filename, epsilon,
    clustering_distance=None, clustering_min_samples=None
):
    """
    Generate a NetworkX graph representing a delay space.

    As input, take two CSV files (for nodes and edges), and a special
    cutoff parameter `epsilon` that determines when an edge should be
    included in the graph.

    Additionally take the two parameters for the DBSCAN algorithm. The
    first parameter is treated in meters (think roughly how closely two
    cities should be in order to be in the same cluster).
    """
    # Create the graph
    graph = nx.Graph()

    # Create the RTT violation list
    rtt_violation_list = []

    # Get the vertecies
    with open(probes_filename) as probes_file:
        probes_reader = csv.DictReader(probes_file)
        for row in probes_reader:
            graph.add_node(
                row['id'],
                city=row['city'], country=row['country'],
                lat=float(row['latitude']), long=float(row['longitude'])
            )
    distance_matrix = {}
    # Get the edges
    final_results_csv = []
    with open(latencies_filename) as latencies_file:
        latencies_reader = csv.DictReader(latencies_file)
        for row in latencies_reader:
            id_source = row['source_id']
            id_target = row['target_id']
            lat_source = graph.nodes[id_source]['lat']
            long_source = graph.nodes[id_source]['long']
            lat_target = graph.nodes[id_target]['lat']
            long_target = graph.nodes[id_target]['long']
            rtt = row['rtt']
            final_results_csv.append([id_source, id_target, lat_source, long_source, lat_target, long_target, graph.nodes[id_target]['city'], graph.nodes[id_target]['city'], utility.get_GCD_latency(
                [lat_source, long_source],
                [lat_target, long_target]), rtt])
            if rtt is None or rtt == '':
                continue
            rtt = float(rtt)
            # if graph.nodes[id_source]['city'] == 'Oslo':
            #     print(graph.nodes[id_source], graph.nodes[id_target], rtt, utility.get_GCD_latency(
            #         [lat_source, long_source],
            #         [lat_target, long_target]))
            # Check how often the difference is larger than 0
            if rtt - utility.get_GCD_latency(
                [lat_source, long_source],
                [lat_target, long_target]) < 0:
                rtt_violation_list.append((id_source, id_target))
                print(id_source, id_target, graph.nodes[id_source], graph.nodes[id_target], rtt, utility.get_GCD_latency(
                    [lat_source, long_source],
                    [lat_target, long_target]))
            # Save the distance matrix for the persistence diagram
            distance_matrix[f'{id_source} - {id_target}'] = rtt - utility.get_GCD_latency(
                [lat_source, long_source],
                [lat_target, long_target])
            # Only add edges satisfying the cutoff requirement
            if (
                rtt - utility.get_GCD_latency(
                    [lat_source, long_source],
                    [lat_target, long_target]
                )
            ) < epsilon:
                # If there is multiple sets of RTT data for a single
                # edge, only pay attention to the minimal one
                if ((id_source, id_target) not in graph.edges
                        or graph.edges[id_source,id_target]['rtt'] > rtt):
                    graph.add_edge(id_source, id_target, weight=1., rtt=rtt)
    pd.DataFrame(final_results_csv, columns=['source_id', 'target_id', 'lat_source', 'long_source', 'lat_target', 'long_target', 'city_source', 'city_target', 'distance', 'rtt']).to_csv(f'{project_dir}/data/{ip_type}/{date}/final_results.csv', index=False)
    with open(f'{project_dir}/data/{ip_type}/{date}/distance_matrix.json', 'w') as f:
        json.dump(distance_matrix, f)
    # Delete nodes with inconsistent geolocation
    nodes_to_delete = minimize_id_removal(rtt_violation_list)

    for node in nodes_to_delete:
        graph.remove_node(node)

    # Simplify the graph by clustering
    if clustering_distance is not None and clustering_min_samples is not None:
        circumference_earth = 40075016.68557849
        graph = cluster_probes.cluster_graph(graph,
                              clustering_distance / circumference_earth,
                              clustering_min_samples)

    # Compute the curvatures. This adds attributes called
    # `ricciCurvature` to the vertices and edges of the graph
    orc = OllivierRicci(graph, weight='weight', alpha=0.)
    print(graph, latencies_filename)
    graph = orc.compute_ricci_curvature()

    # Delete extraneous edge data
    for _, _, d in graph.edges(data=True):
        del d['weight']
        del d['rtt']

    # Delete nodes with no edges
    nodes = list(graph.nodes)
    for node in nodes:
        if len(graph.edges(node)) == 0:
            graph.remove_node(node)

    return graph

def translating_distance_matrix(data):

    # Parse the keys to find unique points
    points = set()
    for key in data.keys():
        point1, point2 = key.split(' - ')
        points.add(point1)
        points.add(point2)

    # Create a list from the set of points
    points = sorted(list(points))

    # Initialize a DataFrame with zeros and points as indices and columns
    distance_matrix = pd.DataFrame(0, index=points, columns=points, dtype=float)

    # Populate the DataFrame
    for key, value in data.items():
        point1, point2 = key.split(' - ')
        distance_matrix.at[point1, point2] = value
        distance_matrix.at[point2, point1] = value
    return distance_matrix

def persistance_diagram(distance_matrix):
    skeleton_protein0 = gd.RipsComplex(
        distance_matrix=distance_matrix.values,
        max_edge_length=60,
    )

    Rips_simplex_tree_protein0 = skeleton_protein0.create_simplex_tree(max_dimension=3)

    BarCodes_Rips0 = Rips_simplex_tree_protein0.persistence()

    gd.plot_persistence_diagram(BarCodes_Rips0);
    plt.savefig(f'{project_dir}/plot/barcodes/rips_{ip_type}_{date}.png')

if __name__ == '__main__':
    # Parse arugments
    parser = argparse.ArgumentParser()
    parser.add_argument('--latencies-file', '-l', metavar='<filename>',
                        dest='latencies_filename', required=True)
    parser.add_argument('--ip-type', '-i', metavar='<ip-type>',required=True)
    # parser.add_argument('--probes-file', '-p', metavar='<filename>',
    #                     dest='probes_filename', required=True)
    parser.add_argument('--region', '-r', metavar='<region>', required=True)
    parser.add_argument('--epsilon', '-e', metavar='<epsilon>',
                        dest='epsilon', type=int, required=False)
    parser.add_argument('--output', '-o', metavar='<basename>',
                        dest='output_basename', required=True)
    args = parser.parse_args()
    latencies_filename = args.latencies_filename
    ip_type = args.ip_type
    region = args.region
    probes_filename = f'{ip_type}/probes_{ip_type}_{region}.csv'
    epsilons = [args.epsilon]
    if args.epsilon is None:
        epsilons = list(range(1, 21))
    output_basename = args.output_basename

    for epsilon in epsilons:
        graph = get_graph(probes_filename, latencies_filename, epsilon,
                          500000, 4)
        nx.write_graphml(graph, f'{output_basename}.graphml')
    distance_matrix = json.load(open(f'{project_dir}/data/{ip_type}/{date}/distance_matrix.json', 'r'))
    distance_matrix = translating_distance_matrix(distance_matrix)
    persistance_diagram(distance_matrix)
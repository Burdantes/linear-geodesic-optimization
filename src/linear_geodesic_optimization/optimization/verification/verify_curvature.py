import datetime
import json
import os

import numpy as np
import scipy

from linear_geodesic_optimization.mesh.rectangle import Mesh as RectangleMesh
from linear_geodesic_optimization.optimization import curvature_loss, laplacian, curvature

toy_directory = os.path.join('..', 'data', 'toy')

# Construct a mesh
width = 10
height = 10
mesh = RectangleMesh(width, height)
vertices = mesh.get_vertices()
V = vertices.shape[0]

coordinates = None
label_to_index = {}
with open(os.path.join(toy_directory, 'position.json')) as f:
    position_json = json.load(f)

    label_to_index = {label: index for index, label in enumerate(position_json)}

    coordinates = [None for _ in range(len(position_json))]
    for vertex, position in position_json.items():
        coordinates[label_to_index[vertex]] = position

network_vertices = mesh.map_coordinates_to_support(coordinates)

network_edges = []
ts = {i: [] for i in range(len(network_vertices))}
with open(os.path.join(toy_directory, 'latency.json')) as f:
    latency_json = json.load(f)

    for edge, latency in latency_json.items():
        u = label_to_index[edge[0]]
        v = label_to_index[edge[1]]

        network_edges.append((u, v))

        ts[u].append((v, latency))

network_curvatures = []
with open(os.path.join(toy_directory, 'curvature.json')) as f:
    network_curvatures = list(json.load(f).values())

laplacian_forward = laplacian.Forward(mesh)
curvature_forward = curvature.Forward(mesh, laplacian_forward)
curvature_loss_forward = curvature_loss.Forward(
    mesh, network_vertices, network_edges, network_curvatures, mesh.get_epsilon()
)
laplacian_reverse = laplacian.Reverse(mesh, laplacian_forward)
curvature_reverse = curvature.Reverse(mesh, laplacian_forward, curvature_forward,
                                      laplacian_reverse)
curvature_loss_reverse = curvature_loss.Reverse(
    mesh, network_vertices, network_edges, network_curvatures, mesh.get_epsilon()
)

l = 37
delta = 1e-5

rng = np.random.default_rng()
z_0 = rng.random(V)
mesh.set_parameters(z_0)
curvature_loss_forward.calc()
L_curvature_0 = curvature_loss_forward.L_curvature

curvature_loss_reverse.calc(mesh.get_partials()[l], l)
dif_L_curvature_0 = curvature_loss_reverse.dif_L_curvature

z_delta = np.copy(z_0)
z_delta[l] += delta
mesh.set_parameters(z_delta)
curvature_loss_forward.calc()
L_curvature_delta = curvature_loss_forward.L_curvature

print(dif_L_curvature_0 / ((L_curvature_delta - L_curvature_0) / delta))

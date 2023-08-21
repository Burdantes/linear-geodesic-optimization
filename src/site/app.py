#!/usr/bin/python3
import json
import flask
import numpy as np
import csv
from flask import request
from flask import Response
import networkx as nx
from networkx.readwrite import json_graph
from OllivierRicci import ricciCurvature
import potpourri3d as pp3d
import sys
import pickle
from python.geodesic import GeodesicDistanceComputation

# Assume we're running from src/
sys.path.append('.')
from linear_geodesic_optimization import data
from linear_geodesic_optimization.mesh.rectangle import Mesh as RectangleMesh

directory = '../out_two_islands/two_islands/1.0_0.004_0.0_20.0_50_50_1.0/'
max_iterations = 10000
vertical_scale = 0.15

sys.path.append(r'python/surface/src')

app = flask.Flask(__name__, static_folder='')
retval = None

@app.route('/calc-curvature', methods=['POST'])
def calc_curvature():
    data = request.json
    G = json_graph.node_link_graph(data)
    Gf = ricciCurvature(G,alpha=0,verbose=True)
    Gr = json_graph.node_link_data(Gf)

    return Gr

@app.route('/calc-distance', methods=['POST'])
def calc_distance():
    data = request.json
    verts = np.array(data['verts'])
    tris = np.array(data['faces'])
    nodes = np.array(data['nodes'])
    edges = np.array(data['edges'])
    # TODO: Change this to use the other geodesics algorithm
    compute_distance = GeodesicDistanceComputation(verts, tris)
    distances = []
    grads = []
    paths = []
    for node in nodes:
        dist = compute_distance(node)
        distances.append(dist.tolist())
    path_solver = pp3d.EdgeFlipGeodesicSolver(verts, tris)
    for edge in edges:
        if edge[0] != edge[1]:
            paths.append(path_solver.find_geodesic_path(v_start=edge[0], v_end=edge[1]).tolist())
        else:
            paths.append([[0, 0, 0]])
    ret = {}
    ret['distances'] = distances
    ret['paths'] = paths
    return json.dumps(ret)

@app.route('/calc-surface', methods=['POST'])
def calc_surface():
    json_data = request.json

    # TODO: Actually run the optimization algorithm with this data
    smooth_pen = int(json_data['smooth_pen'])
    niter = int(json_data['niter'])
    hmap = json_data['map']
    G = json_graph.node_link_graph(json_data['graph'])
    H = nx.Graph(G)

    mesh = data.get_mesh_output(directory, max_iterations, True)
    z = mesh.get_parameters()
    width = mesh.get_width()
    height = mesh.get_height()

    z = np.flip(z.reshape((width, height)), axis=1).T.reshape((-1))
    z = z - np.amin(z)
    z = z * vertical_scale / np.amax(z)
    z = z - np.amax(z)
    z = z.tolist()
    return Response(json.dumps(z), mimetype='text/plain')

@app.route('/')
def static_proxy():
    return app.send_static_file('index.html')

if __name__=='__main__':
    app.run()

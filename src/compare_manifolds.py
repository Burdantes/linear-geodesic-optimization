import scipy as sp
import numpy as np
import matplotlib.pylab as pl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa
import ot
import sys
from linear_geodesic_optimization import data
from utils import *

ip_type = 'ipv6'
directory = os.path.join(project_dir,'data', ip_type, date, 'out_Europe_hourly')


i = 0
current_directory = os.path.join(
    directory, f'graph_{i}_{threshold}', subdirectory_name)
initialization_path = os.path.join(directory, f'graph_0_{threshold}', subdirectory_name, '0')

mesh_ipv6 = data.get_mesh_output(
    current_directory, postprocessed=True,
    intialization_path=initialization_path
)
C1 = data.compute_geodesic_distance_matrix(mesh_ipv6)
C1_geodesic = data.read_geodesic_distances(current_directory)
ip_type = 'ipv4'
directory = os.path.join(project_dir,'data', ip_type, date, 'out_Europe_hourly')

i = 0
current_directory = os.path.join(
    directory, f'graph_{i}_{threshold}', subdirectory_name)
initialization_path = os.path.join(directory, f'graph_0_{threshold}', subdirectory_name, '0')
print(initialization_path)
mesh_ipv4 = data.get_mesh_output(
    current_directory, postprocessed=True,
    intialization_path=initialization_path
)

C2 = data.compute_geodesic_distance_matrix(mesh_ipv4)
C2_geodesic = data.read_geodesic_distances(current_directory)
# create a flash mesh with 50 points on each side
n = 50
x = np.linspace(0, 1, n)
y = np.linspace(0, 1, n)
X, Y = np.meshgrid(x, y)
mesh_flat = np.hstack((X.reshape(-1, 1), Y.reshape(-1, 1)))

### compute the distance matrix of the flat mesh
C3 = sp.spatial.distance.cdist(mesh_flat, mesh_flat)
# Compare the difference between the two matrices
print(C3)
print(np.linalg.norm(C1 - C2),)

n_samples = 50 * 50 # nb samples
# Normalizing C1 and C2
C1 /= C1.max()
C2 /= C2.max()
C3 /= C3.max()

p = ot.unif(n_samples)
q = ot.unif(n_samples)

# Create a figure and a set of subplots
fig, axs = plt.subplots(1, 2, figsize=(10, 5))  # You can adjust figsize to suit your needs

# Plotting C1
im1 = axs[0].imshow(C1)
axs[0].set_title('C1')

# Plotting C2
im2 = axs[1].imshow(C2)
axs[1].set_title('C2')

# Adding a colorbar
fig.colorbar(im1, ax=axs, orientation='vertical')

plt.show()
print(C1,C2,p,q)
# Look at the overlaping rows and columns between C1_geodesic and C2_geodesic
intersection = np.intersect1d(C1_geodesic.index, C2_geodesic.index)
C1_geodesic = C1_geodesic[intersection][C1_geodesic.index.isin(intersection)]
C2_geodesic = C2_geodesic[intersection][C2_geodesic.index.isin(intersection)]
print(C1_geodesic.shape,C2_geodesic.shape)
C1_geodesic /= C1_geodesic.max()
C2_geodesic /= C2_geodesic.max()
n_samples_1 = C1_geodesic.shape[0]  # nb samples
n_samples_2 = C2_geodesic.shape[0]  # nb samples
p = ot.unif(n_samples_1)
q = ot.unif(n_samples_2)
fig, axs = plt.subplots(1, 2, figsize=(10, 5))  # You can adjust figsize to suit your needs

print(C1_geodesic)
print(C2_geodesic)
# Plotting C1
im1 = axs[0].imshow(C1_geodesic)
axs[0].set_title('C1')

# Plotting C2
im2 = axs[1].imshow(C2_geodesic)
axs[1].set_title('C2')

# Adding a colorbar
fig.colorbar(im1, ax=axs, orientation='vertical')

plt.show()

# Conditional Gradient algorithm
gw0, log0 = ot.gromov.gromov_wasserstein(
    C1_geodesic.values, C2_geodesic.values, p, q, 'square_loss', verbose=True, log=True)

# Find the mapping
mapping = np.argmax(gw0, axis=1)  # or axis=0 depending on your specific setup
mapping_to_index = dict(C1_geodesic.index.to_series().items())
print(mapping_to_index)
for i, j in enumerate(mapping):
    if i != j:
        print(f"Source point {mapping_to_index[i]} is mapped to target point {mapping_to_index[j]}")
print('Gromov-Wasserstein distance estimated with Conditional Gradient solver: ' + str(log0['gw_dist']))
from mpl_toolkits.mplot3d import Axes3D

# Initialize the plot
fig = plt.figure(figsize=(12, 6))

coords1 = mesh_ipv4.get_coordinates()
coords2 = mesh_ipv6.get_coordinates()
# Plot the first mesh
from scipy.spatial import Delaunay

# Assuming coords1 is an Nx3 array of your mesh points
tri2 = mesh_ipv4.get_triangulation()
tri1 = mesh_ipv6.get_triangulation()


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

def plot_mesh(vertices, triangles, ax, color):
    """ Plot a mesh with given vertices and triangles on provided Axes. """
    for tri in triangles:
        triangle = vertices[tri]
        collection = Poly3DCollection([triangle], alpha=0.5, linewidths=1)
        collection.set_facecolor(color)
        collection.set_edgecolor('k')  # Black edge color
        ax.add_collection3d(collection)

def compare_and_plot(vertices, tri1, tri2):
    # Create sets of triangles for easy comparison
    set_tri1 = set(map(tuple, np.sort(tri1, axis=1)))
    set_tri2 = set(map(tuple, np.sort(tri2, axis=1)))

    # Find unique and common triangles
    common_triangles = np.array(list(set_tri1.intersection(set_tri2)))
    unique_tri1 = np.array(list(set_tri1 - set_tri2))
    unique_tri2 = np.array(list(set_tri2 - set_tri1))

    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot common triangles in one color (e.g., gray)
    plot_mesh(vertices, common_triangles, ax, color='gray')

    # Plot unique triangles in different colors
    plot_mesh(vertices, unique_tri1, ax, color='blue')  # Unique to tri1
    plot_mesh(vertices, unique_tri2, ax, color='red')   # Unique to tri2

    # Setting plot limits for better visualization
    ax.set_xlim([np.min(vertices[:, 0]), np.max(vertices[:, 0])])
    ax.set_ylim([np.min(vertices[:, 1]), np.max(vertices[:, 1])])
    ax.set_zlim([np.min(vertices[:, 2]), np.max(vertices[:, 2])])

    plt.show()


compare_and_plot(coords1, tri1, tri2)
ax1 = fig.add_subplot(121, projection='3d')
ax1.plot_trisurf(coords1[:, 0], coords1[:, 1], coords1[:, 2], triangles=tri1, color='skyblue', alpha=0.5)
ax1.set_title("IPv4")

# Plot the second mesh
ax2 = fig.add_subplot(122, projection='3d')
ax2.plot_trisurf(coords2[:, 0], coords2[:, 1], coords2[:, 2], triangles=tri2, color='salmon', alpha=0.5)
ax2.set_title("IPv6")

# Draw lines between matched points for a subset of points
subset_indices = np.random.choice(range(len(coords1)), size=100, replace=False) # Adjust size as needed
# for i in subset_indices:
#     j = mapping[i]
#     start_point = coords1[i]
#     end_point = coords2[j]
#     line = np.array([start_point, end_point])
#     ax2.plot(line[:, 0], line[:, 1], line[:, 2], color='green', linewidth=2, alpha=0.7)

# Adjust the viewing angle as needed
ax1.view_init(elev=20., azim=30)
ax2.view_init(elev=20., azim=30)

plt.show()
# compute OT sparsity level
gw0_sparsity = 100 * (gw0 == 0.).astype(np.float64).sum() / (n_samples_1 * n_samples_2)

err0 = np.linalg.norm(gw0.sum(1) - p) + np.linalg.norm(gw0.sum(0) - q)

pl.figure(figsize=(10, 5))
cmap = 'Blues'
fontsize = 12
pl.imshow(gw0, cmap=cmap)
pl.title('(CG algo) GW=%s \n  \n OT sparsity=%s \n feasibility error=%s' % (
    np.round(log0['gw_dist'], 4), str(np.round(gw0_sparsity, 2)) + ' %', np.round(np.round(err0, 4))),
    fontsize=fontsize)

pl.tight_layout()
print(date, i, threshold)
# pl.savefig(f'gw_ipv4_ipv6_{date}_{i}_{threshold}.png')
pl.show()

### IPv4 vs IPv6 plot

import matplotlib.pyplot as plt
import numpy as np

# Assuming mesh_ipv4 and mesh_ipv6 are defined and have the methods get_coordinates() and get_topology()

# IPv4 Plotting
vertices_ipv4 = mesh_ipv4.get_coordinates()
x_ipv4, y_ipv4, z_ipv4 = vertices_ipv4[:,0], vertices_ipv4[:,1], vertices_ipv4[:,2]

faces_ipv4 = []
for face in mesh_ipv4.get_topology().faces():
    faces_ipv4.append([v.index() for v in face.vertices()])

z_min_ipv4 = np.amin(z_ipv4)
z_max_ipv4 = np.amax(z_ipv4)
if z_min_ipv4 != z_max_ipv4:
    z_ipv4 = (z_ipv4 - z_min_ipv4) / (z_max_ipv4 - z_min_ipv4) / 4.

# IPv6 Plotting
vertices_ipv6 = mesh_ipv6.get_coordinates()
x_ipv6, y_ipv6, z_ipv6 = vertices_ipv6[:,0], vertices_ipv6[:,1], vertices_ipv6[:,2]

faces_ipv6 = []
for face in mesh_ipv6.get_topology().faces():
    faces_ipv6.append([v.index() for v in face.vertices()])

z_min_ipv6 = np.amin(z_ipv6)
z_max_ipv6 = np.amax(z_ipv6)
if z_min_ipv6 != z_max_ipv6:
    z_ipv6 = (z_ipv6 - z_min_ipv6) / (z_max_ipv6 - z_min_ipv6) / 4.

# Plot configuration
fig = plt.figure(figsize=(12, 6))  # You can adjust the figure size as needed

# IPv4 plot
ax_ipv4 = fig.add_subplot(1, 2, 1, projection='3d')  # 1 row, 2 cols, chart 1
p3dc_ipv4 = ax_ipv4.plot_trisurf(x_ipv4, y_ipv4, z_ipv4, triangles=faces_ipv4, color='tab:blue')
plt.title('IPv4')

# IPv6 plot
ax_ipv6 = fig.add_subplot(1, 2, 2, projection='3d')  # 1 row, 2 cols, chart 2
p3dc_ipv6 = ax_ipv6.plot_trisurf(x_ipv6, y_ipv6, z_ipv6, triangles=faces_ipv6, color='tab:orange')
plt.title('IPv6')

plt.show()


### read
#
# n_samples = 30  # nb samples
#
# mu_s = np.array([0, 0])
# cov_s = np.array([[1, 0], [0, 1]])
#
# mu_t = np.array([4, 4, 4])
# cov_t = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
#
# np.random.seed(0)
# xs = ot.datasets.make_2D_samples_gauss(n_samples, mu_s, cov_s)
# P = sp.linalg.sqrtm(cov_t)
# xt = np.random.randn(n_samples, 3).dot(P) + mu_t
#
# print(xs)
# print(xt)

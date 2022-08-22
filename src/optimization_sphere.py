import numpy as np
import scipy.optimize

from linear_geodesic_optimization.data import phony
from linear_geodesic_optimization.mesh.sphere import Mesh as SphereMesh
from linear_geodesic_optimization.optimization import optimization
from linear_geodesic_optimization.plot import get_scatter_fig, \
    combine_scatter_figs, Animation3D

if __name__ == '__main__':
    # Construct the mesh
    frequency = 4
    mesh = SphereMesh(frequency)
    partials = mesh.get_partials()
    V = partials.shape[0]
    rho = mesh.get_parameters()
    rho = mesh.set_parameters(np.random.rand(*rho.shape) + 0.5)

    dif_v = {l: partials[l] for l in range(V)}

    # Get some (phony) latency measurements
    ts = phony.sphere_random(mesh)
    s_indices = ts.keys()

    lam = 0.01
    hierarchy = optimization.DifferentiationHierarchy(mesh, ts, lam)

    f = hierarchy.get_loss_callback(s_indices)
    g = hierarchy.get_dif_loss_callback(s_indices)

    before = get_scatter_fig(hierarchy, True)

    hierarchy.print_diagnostics(None)
    scipy.optimize.minimize(f, rho, method='L-BFGS-B', jac=g,
                            callback=hierarchy.print_diagnostics)

    after = get_scatter_fig(hierarchy, False)

    combine_scatter_figs(before, after).show()

    animation_3D = Animation3D()
    for parameters in hierarchy.history:
        mesh.set_parameters(parameters)
        animation_3D.add_frame(mesh)
    animation_3D.get_fig(duration=50).show()

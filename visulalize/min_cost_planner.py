import numpy as np
import pyvista as pv

import cvxpy as cp
from pyvista import DataSet, PolyData
from pyvista.plotting import plotter


def closest_vertex(point, mesh: DataSet):
    """
    Find the closest vertex in the mesh to the given point.

    Parameters:
    - point: A 3D point as a list or a numpy array (e.g., [x, y, z]).
    - mesh: A pyvista DataSet containing the mesh.

    Returns:
    - closest_vertex_index: The index of the closest vertex in the mesh.
    - closest_vertex_coords: The coordinates of the closest vertex.
    """
    vertices = mesh.points

    distances = np.linalg.norm(vertices - point, axis=1)

    closest_vertex_index = np.argmin(distances)

    closest_vertex_coords = vertices[closest_vertex_index]

    return closest_vertex_index, closest_vertex_coords


def get_path(start_index: int, goal_index: int, mesh: pv.DataSet) -> PolyData:
    vertices = mesh.points

    edges_polydata = mesh.extract_all_edges()
    edge_cells = edges_polydata.lines.reshape(-1, 3)[:, 1:]

    edges = [(edge[0], edge[1]) for edge in edge_cells]
    edge_costs = [np.linalg.norm(vertices[edge[0]] - vertices[edge[1]]) for edge in edges]

    edge_count = len(edges)
    vertex_count = vertices.shape[0]

    edge_index = {edge: idx for idx, edge in enumerate(edges)}

    x = cp.Variable(edge_count, boolean=True)

    d = np.array(edge_costs)
    objective = cp.Minimize(x @ d)

    constraints = []

    for v in range(vertex_count):
        if v == start_index or v == goal_index:
            continue
        incoming = [x[edge_index[(u, v)]] for u in range(vertex_count) if (u, v) in edge_index]
        outgoing = [x[edge_index[(v, u)]] for u in range(vertex_count) if (v, u) in edge_index]
        constraints.append(-cp.sum(incoming) + cp.sum(outgoing) == 0)

    incoming_start = [x[edge_index[(u, start_index)]] for u in range(vertex_count) if (u, start_index) in edge_index]
    outgoing_start = [x[edge_index[(start_index, u)]] for u in range(vertex_count) if (start_index, u) in edge_index]
    constraints.append(-cp.sum(incoming_start) + cp.sum(outgoing_start) == 1)

    incoming_goal = [x[edge_index[(u, goal_index)]] for u in range(vertex_count) if (u, goal_index) in edge_index]
    outgoing_goal = [x[edge_index[(goal_index, u)]] for u in range(vertex_count) if (goal_index, u) in edge_index]
    constraints.append(-cp.sum(incoming_goal) + cp.sum(outgoing_goal) == -1)

    prob = cp.Problem(objective, constraints)
    prob.solve()

    path_edges = [edges[i] for i in range(edge_count) if x.value[i] > 0.5]

    path_points = []

    for edge in path_edges:
        path_points.extend(vertices[list(edge)])

    path_points = np.array(path_points)
    path_polydata = pv.PolyData(path_points)
    return path_polydata


def callback(point, picker):
    nonlocal clicked_points
    clicked_points.append(point)
    plotter.add_mesh(pv.PolyData(point), color='red', point_size=10)
    if len(clicked_points) == 2:
        start_vertex_index, _ = closest_vertex(clicked_points[0], pv_mesh)
        goal_vertex_index, _ = closest_vertex(clicked_points[1], pv_mesh)
        path_polydata = get_path(start_vertex_index, goal_vertex_index, pv_mesh)
        plotter.add_mesh(path_polydata, color='white', line_width=5)
        clicked_points = []


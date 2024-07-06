import time

import numpy as np
import pyvista as pv
import cvxpy as cp
from scipy.spatial import distance


def find_vertex_index_dict(vertices):
    return {tuple(vertex): idx for idx, vertex in enumerate(vertices)}


def mixed_integer_kinodynamic_planner(start_point: np.ndarray,
                                      goal_point: np.ndarray,
                                      _: pv.Plotter,
                                      mesh: pv.DataSet) -> list[np.ndarray] | None:
    start_time = time.time()
    edges_polydata = mesh.extract_all_edges()
    edge_cells = edges_polydata.lines.reshape(-1, 3)[:, 1:]
    max_acceleration = 0.5

    vertices = edges_polydata.points
    num_vertices = len(vertices)
    vertex_idx_dict = find_vertex_index_dict(vertices)

    edge_dict = {}
    d = []

    # Vectorized edge processing
    p1_indices, p2_indices = edge_cells[:, 0], edge_cells[:, 1]
    p1_vertices, p2_vertices = vertices[p1_indices], vertices[p2_indices]
    distances = np.linalg.norm(p1_vertices - p2_vertices, axis=1)

    for idx, (p1, p2, dist) in enumerate(zip(p1_vertices, p2_vertices, distances)):
        edge_dict[idx * 2] = (p1, p2)
        edge_dict[idx * 2 + 1] = (p2, p1)
        d.extend([dist, dist])

    d = np.array(d)

    distances_to_start = distance.cdist([start_point], vertices).flatten()
    closest_start_index = np.argmin(distances_to_start)
    start = vertices[closest_start_index]

    distances_to_goal = distance.cdist([goal_point], vertices).flatten()
    closest_goal_index = np.argmin(distances_to_goal)
    goal = vertices[closest_goal_index]

    num_edges = len(edge_dict)
    x = cp.Variable(num_edges, boolean=True)
    v = cp.Variable(num_vertices)
    objective = cp.Minimize(cp.sum(cp.multiply(d, x)))

    constraints = []
    vertex_indices = set(range(num_vertices))
    start_idx = vertex_idx_dict[tuple(start)]
    goal_idx = vertex_idx_dict[tuple(goal)]

    for vertex_idx in vertex_indices - {start_idx, goal_idx}:
        incoming_edges = [i for i, (p1, p2) in edge_dict.items() if vertex_idx_dict[tuple(p2)] == vertex_idx]
        outgoing_edges = [i for i, (p1, p2) in edge_dict.items() if vertex_idx_dict[tuple(p1)] == vertex_idx]
        constraints.append(cp.sum(x[outgoing_edges]) - cp.sum(x[incoming_edges]) == 0)

    start_edge_incoming = [i for i, (p1, p2) in edge_dict.items() if vertex_idx_dict[tuple(p2)] == start_idx]
    start_edges_outgoing = [i for i, (p1, p2) in edge_dict.items() if vertex_idx_dict[tuple(p1)] == start_idx]
    constraints.append(cp.sum(x[start_edges_outgoing]) - cp.sum(x[start_edge_incoming]) == 1)

    goal_edges_incoming = [i for i, (p1, p2) in edge_dict.items() if vertex_idx_dict[tuple(p2)] == goal_idx]
    goal_edges_outgoing = [i for i, (p1, p2) in edge_dict.items() if vertex_idx_dict[tuple(p1)] == goal_idx]
    constraints.append(cp.sum(x[goal_edges_incoming]) - cp.sum(x[goal_edges_outgoing]) == 1)

    for edge, (p1, p2) in edge_dict.items():
        p1_idx = vertex_idx_dict[tuple(p1)]
        p2_idx = vertex_idx_dict[tuple(p2)]
        constraints.append(cp.abs(v[p1_idx] - v[p2_idx]) <= max_acceleration)

    constraints.append(v[closest_start_index] == 0)
    constraints.append(v[closest_goal_index] == 0)

    problem = cp.Problem(objective, constraints)
    print(f"took {time.time() - start_time} seconds to create")

    start_time = time.time()
    problem.solve()
    print(f"took {time.time() - start_time} seconds to solve")

    if problem.status == cp.OPTIMAL:
        path_points = [point for edge, (p1, p2) in edge_dict.items() if x.value[edge] == 1 for point in (p1, p2)]
        return path_points
    return None

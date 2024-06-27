import numpy as np
import pyvista as pv
import cvxpy as cp
from pyvista import Plotter
from scipy.spatial import distance


def find_vertex_index(vertex, vertices):
    for i, v in enumerate(vertices):
        if np.array_equal(v, vertex):
            return i
    raise ValueError("Vertex not found")


def optimal_planner(start_point: np.ndarray, goal_point: np.ndarray, mesh: pv.PolyData, plotter: Plotter):
    edges_polydata = mesh.extract_all_edges()
    edge_cells = edges_polydata.lines.reshape(-1, 3)[:, 1:]
    max_acceleration = 0.5
    edge_dict = {}
    vertices = []
    d = []
    # Add the edges
    index = 0
    for edge in edge_cells:
        points = edges_polydata.points[edge]
        edge_dict[index] = (points[0], points[1])
        edge_dict[index + 1] = (points[1], points[0])
        vertices.extend([points[0], points[1]])
        d.append(np.linalg.norm(points[0] - points[1]))
        d.append(np.linalg.norm(points[0] - points[1]))
        index += 2

    distances_to_start = distance.cdist([start_point], np.array(vertices))
    closest_start_index = np.argmin(distances_to_start)
    start = vertices[closest_start_index]

    distances_to_goal = distance.cdist([goal_point], np.array(vertices))
    closest_goal_index = np.argmin(distances_to_goal)
    goal = vertices[closest_goal_index]

    x = cp.Variable(len(edge_dict), boolean=True)
    d = np.array(d)
    v = cp.Variable(len(vertices))
    objective = cp.Minimize(x @ d)

    constraints = []
    for vertex in vertices:
        if not np.array_equal(vertex, start) and not np.array_equal(vertex, goal):
            incoming_edges = [i for i, (p1, p2) in edge_dict.items() if np.array_equal(p2, vertex)]
            outgoing_edges = [i for i, (p1, p2) in edge_dict.items() if np.array_equal(p1, vertex)]
            constraint = -cp.sum(x[incoming_edges]) + cp.sum(x[outgoing_edges]) == 0
            constraints.append(constraint)

    start_edge_incoming = [i for i, (p1, p2) in edge_dict.items() if np.array_equal(p2, start)]
    start_edges_outgoing = [i for i, (p1, p2) in edge_dict.items() if np.array_equal(p1, start)]
    constraints.append(-cp.sum(x[start_edge_incoming]) + cp.sum(x[start_edges_outgoing]) == 1)

    goal_edges_incoming = [i for i, (p1, p2) in edge_dict.items() if np.array_equal(p2, goal)]
    goal_edges_outgoing = [i for i, (p1, p2) in edge_dict.items() if np.array_equal(p1, goal)]
    constraints.append(-cp.sum(x[goal_edges_incoming]) + cp.sum(x[goal_edges_outgoing]) == -1)

    for edge, (p1, p2) in edge_dict.items():
        p1_idx = find_vertex_index(p1, vertices)
        p2_idx = find_vertex_index(p2, vertices)
        constraints.append(cp.abs(v[p1_idx] - v[p2_idx]) <= max_acceleration)

    constraints.append(v[closest_start_index] == 0)
    constraints.append(v[closest_goal_index] == 0)
    problem = cp.Problem(objective, constraints)
    problem.solve()

    # Print the results
    # print("Status:", problem.status)
    # print("Optimal value:", problem.value)
    # print("Edge flows:", x.value)
    for edge, (p1, p2) in edge_dict.items():
        if x.value[edge] == 1:
            plotter.add_points(p1, color='white', point_size=10, )
            plotter.add_points(p2, color='white', point_size=10, )

    # Add start and goal points
    plotter.add_points(start, color='red', point_size=10, label='Start Point')
    plotter.add_points(goal, color='blue', point_size=10, label='Goal Point')

    plotter.show()
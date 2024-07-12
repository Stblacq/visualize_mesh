import numpy as np
import pyvista as pv
from pyvista.plotting import Plotter

from visualize import mixed_integer_kinodynamic_planner


def check_mesh_size(mesh: pv.DataSet):
    num_points = mesh.n_points
    num_cells = mesh.n_cells
    print(f"Number of points in the mesh: {num_points}")
    print(f"Number of cells in the mesh: {num_cells}")


def get_sorted_boundary_points(mesh: pv.DataSet,
                               start_point: np.ndarray,
                               goal_point: np.ndarray,
                               radius: float,
                               num_samples: int = 100):
    distances = np.linalg.norm(mesh.points - start_point, axis=1)

    tolerance = 0.1
    mask = np.abs(distances - radius) < tolerance

    boundary_points = mesh.points[mask]

    if len(boundary_points) > num_samples:
        sampled_indices = np.random.choice(len(boundary_points), num_samples, replace=False)
        boundary_points = boundary_points[sampled_indices]

    boundary_points = sorted(boundary_points, key=lambda point: np.linalg.norm(point - goal_point))

    return boundary_points


def dummy_planner(start_point: np.ndarray,
                  goal_point: np.ndarray,
                  mesh: pv.DataSet) -> np.ndarray | None:
    return [start_point, goal_point]


def geodesic_planner(start_point: np.ndarray, goal_point: np.ndarray, mesh: pv.PolyData, plotter: Plotter):
    start_id = mesh.find_closest_point(start_point)
    goal_id = mesh.find_closest_point(goal_point)

    path = mesh.geodesic(start_id, goal_id)
    plotter.add_mesh(path, color='white', line_width=5)


# def move_towards_goal(goal_point: np.ndarray, current_position: np.ndarray, radius: float):
#     direction_vector = goal_point - current_position
#     direction_vector = direction_vector / np.linalg.norm(direction_vector)  # Normalize
#     new_position = current_position + direction_vector * radius
#     return new_position


def is_point_in_mesh(mesh: pv.DataSet, point: np.ndarray, tolerance: float = 1e-6) -> bool:
    closest_point_index = mesh.find_closest_point(point)
    closest_point = mesh.points[closest_point_index]
    distance_to_closest_point = np.linalg.norm(closest_point - point)
    return distance_to_closest_point < tolerance


def extract_sub_mesh(mesh, radius, start_point):
    distances = np.linalg.norm(mesh.points - start_point, axis=1)
    mask = distances <= radius
    extracted_mesh = mesh.extract_points(mask)
    return extracted_mesh


def sequential_submesh_planner(start_point: np.ndarray,
                               goal_point: np.ndarray,
                               plotter,
                               mesh: pv.DataSet,
                               max_iterations: int = 1000):
    radius = 0.3
    path = []
    current_start = start_point
    iteration = 0

    while iteration < max_iterations:
        extracted_mesh = extract_sub_mesh(mesh, radius, current_start)
        plotter.add_mesh(extracted_mesh, color='white')
        boundary_points = get_sorted_boundary_points(extracted_mesh, current_start, goal_point, radius)
        # plotter.add_points(np.array(boundary_points), color='blue')

        if is_point_in_mesh(extracted_mesh, goal_point):
            sub_path = mixed_integer_kinodynamic_planner(current_start, goal_point, None, extracted_mesh)
            if sub_path:
                path.extend(sub_path)
            break  # Path to goal found, exit loop
        else:
            progress_made = False
            for closest_point in boundary_points:
                sub_path = mixed_integer_kinodynamic_planner(current_start, closest_point, None, extracted_mesh)
                if sub_path:
                    path.extend(sub_path)
                    current_start = closest_point
                    progress_made = True
                    break

            if not progress_made:
                print("No progress made, stopping the planner.")
                break
        iteration += 1

    if iteration == max_iterations:
        print("Reached maximum iterations, stopping the planner.")
    plotter.add_lines(np.array(path), color='blue', label='Path')
    return path

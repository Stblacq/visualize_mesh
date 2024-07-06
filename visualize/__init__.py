import os
import sys
import time
from typing import Callable

import numpy as np
import pyvista as pv

from visualize.miki_planner import mixed_integer_kinodynamic_planner
from visualize.ssp import sequential_submesh_planner


def visualizer(mesh_file_path: str, planner: Callable):
    if not os.path.isfile(mesh_file_path):
        print(f"The specified file was not found: {mesh_file_path}")
        return

    pv_mesh = pv.read(mesh_file_path)
    pv_mesh["Elevation"] = pv_mesh.points[:, 2]
    plotter = pv.Plotter()

    plotter.add_mesh(pv_mesh, scalars="Elevation", cmap="terrain", color='lightblue', show_edges=True)
    print(f"Number of points in the mesh: {pv_mesh.n_points}")
    plotter.add_axes(interactive=True)
    clicked_points = []

    def callback(point, _):
        nonlocal clicked_points
        clicked_points.append(point)
        plotter.add_mesh(pv.PolyData(point), color='red', point_size=10)

        if len(clicked_points) == 2:
            start_time = time.time()  # Start time
            path_points = planner(clicked_points[0], clicked_points[1], plotter, pv_mesh)
            plotter.add_lines(np.array(path_points), color='blue', label='Path')
            print(f"Function execution time: {time.time() - start_time:.4f} seconds")

            plotter.add_points(clicked_points[0], color='red', point_size=10, label='Start Point')
            plotter.add_points(clicked_points[1], color='green', point_size=10, label='Goal Point')

            plotter.add_legend()
            plotter.show()
            clicked_points = []

    plotter.enable_point_picking(callback=callback, use_picker=True)
    plotter.show()


def run_example(example: str):
    if example == 'simple':
        mesh_filename = 'simple_terrain.obj'
        planner = mixed_integer_kinodynamic_planner
    elif example == 'large':
        mesh_filename = 'desert.obj'
        planner = sequential_submesh_planner
    else:
        print("Invalid example specified. Use 'simple' or 'large'.")
        return

    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), mesh_filename)
    print(f"Running {example} example with file: {file_path}")
    visualizer(file_path, planner)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python -m visualize.run_example <example>")
        print("Example options: 'simple', 'large'")
    else:
        run_example(sys.argv[1])

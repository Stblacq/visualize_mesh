import os
import time
from typing import Callable

import numpy as np
import pyvista as pv

from visualize.miki_planner import mixed_integer_kinodynamic_planner
from visualize.ssp import sequential_submesh_planner


class Visualizer:
    def __init__(self, mesh_file_path: str, planner: Callable):
        self.clicked_points = []
        self.mesh_file_path = mesh_file_path
        self.planner = planner
        self.visualize()

    def visualize(self):
        if not os.path.isfile(self.mesh_file_path):
            print(f"The specified file was not found: {self.mesh_file_path}")
            return
        pv_mesh = pv.read(self.mesh_file_path)
        pv_mesh["Elevation"] = pv_mesh.points[:, 2]
        plotter = pv.Plotter()

        plotter.add_mesh(pv_mesh, scalars="Elevation", cmap="terrain", color='lightblue', show_edges=True)
        print(f"Number of points in the mesh: {pv_mesh.n_points}")
        plotter.add_axes(interactive=True)

        def callback(point, _):
            self.clicked_points.append(point)
            plotter.add_mesh(pv.PolyData(point), color='red', point_size=10)
            print(">>>>>>>", len(self.clicked_points))

            if len(self.clicked_points) == 2:
                start_time = time.time()
                start, goal = self.clicked_points
                self.clicked_points.clear()
                path_points = self.planner(start, goal, plotter, pv_mesh)
                plotter.add_lines(np.array(path_points), color='blue', label='Path')
                print(f"Function execution time: {time.time() - start_time:.4f} seconds")

                plotter.add_points(start, color='red', point_size=10, label='Start Point')
                plotter.add_points(goal, color='green', point_size=10, label='Goal Point')

                plotter.add_legend()
                plotter.show()

        plotter.enable_point_picking(callback=callback, use_picker=True)
        plotter.show()


def run_example(example: str):
    if example == 'simple':
        file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'simple_terrain.obj')
        visualizer = Visualizer(file_path, mixed_integer_kinodynamic_planner)
        visualizer.visualize()
    elif example == 'large':
        file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'desert.obj')
        visualizer = Visualizer(file_path, sequential_submesh_planner)
        visualizer.visualize()
    else:
        print("Invalid example specified. Use 'simple' or 'large'.")
        return


if __name__ == "__main__":
    # if len(sys.argv) != 2:
    #     print("Usage: python -m visualize.run_example <example>")
    #     print("Example options: 'simple', 'large'")
    # else:
    #     run_example(sys.argv[1])
    run_example('large')

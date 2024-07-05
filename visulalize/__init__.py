import time

import numpy as np
import pyvista as pv
from pyvista.plotting import Plotter


from visulalize.mcf import mixed_integer_kinodynamic_planner
from visulalize.pruned import mixed_integer_kinodynamic_planner_pruned


def geodesic_planner(start_point: np.ndarray, goal_point: np.ndarray, mesh: pv.PolyData, plotter: Plotter):
    start_id = mesh.find_closest_point(start_point)
    goal_id = mesh.find_closest_point(goal_point)

    path = mesh.geodesic(start_id, goal_id)
    plotter.add_mesh(path, color='white', line_width=5)


def main():
    try:
        pv_mesh = pv.read('/home/altair/PycharmProjects/visualize_mesh/visulalize/desert.obj')
    except FileNotFoundError:
        print("The specified file was not found.")
        return

    pv_mesh["Elevation"] = pv_mesh.points[:, 2]
    plotter = pv.Plotter()

    plotter.add_mesh(pv_mesh, scalars="Elevation", cmap="terrain", color='lightblue', show_edges=True)
    print(pv_mesh.n_points)
    # plotter.add_mesh(pv_mesh)
    plotter.add_axes(interactive=True)
    clicked_points = []

    def callback(point, _):
        nonlocal clicked_points
        clicked_points.append(point)
        plotter.add_mesh(pv.PolyData(point), color='red', point_size=10)

        if len(clicked_points) == 2:
            start_time = time.time()  # Start time
            path_points = mixed_integer_kinodynamic_planner_pruned(clicked_points[0],
                                                                   clicked_points[1], plotter, pv_mesh)
            plotter.add_lines(np.array(path_points), color='blue', label='Path')
            print(f"Function execution time: {time.time() - start_time:.4f} seconds")

            plotter.add_points(clicked_points[0], color='red', point_size=10, label='Start Point')
            plotter.add_points(clicked_points[1], color='green', point_size=10, label='Goal Point')

            plotter.add_legend()
            plotter.show()
            clicked_points = []

    plotter.enable_point_picking(callback=callback, use_picker=True)
    plotter.show()


if __name__ == "__main__":
    main()

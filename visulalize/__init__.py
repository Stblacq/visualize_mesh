import pyvista as pv

if __name__ == "__main__":
    pv_mesh = pv.read('/home/altair/PycharmProjects/visualize_mesh/visulalize/desert.obj')

    pv_mesh["Elevation"] = pv_mesh.points[:, 2]
    plotter = pv.Plotter()

    plotter.add_mesh(pv_mesh, scalars="Elevation", cmap="terrain", color='lightblue', show_edges=True)
    plotter.add_axes(interactive=True)
    clicked_points = []

    def callback(point, picker):
        global clicked_points
        clicked_points.append(point)
        plotter.add_mesh(pv.PolyData(point), color='red', point_size=10)
        if len(clicked_points) == 2:
            line = pv.Line(clicked_points[0], clicked_points[1])
            plotter.add_mesh(line, color='green', line_width=5)
            clicked_points = []

    plotter.enable_point_picking(callback=callback, use_picker=True)

    plotter.show()

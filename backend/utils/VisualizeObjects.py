import pyvista
import numpy as np

import os
from pathlib import Path

from .config import PRODUCTION_DATA


def visualize_dif_obj(file):
    points = np.loadtxt(file)
    plotter = pyvista.Plotter(window_size=(1000, 700))
    plotter.add_points(points, color='blue', point_size=3, label='Объект (точки)')
    plotter.add_legend()
    plotter.set_background('white')
    plotter.add_axes()
    plotter.show(interactive=True)

if __name__ == '__main__':
    files = [file for file in os.listdir(PRODUCTION_DATA)]
    files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
    for file in files:

        print(file)
        visualize_dif_obj(Path(PRODUCTION_DATA)/Path(file))